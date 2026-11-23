"""Run an isolated WALL and GPT episode on a frozen RoboDojo layout."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from .cards import fetch, load_card
from .robodojo.evaluation import case_identity, read_panel, verify_assets
from .robodojo.io import write_json

PACKAGE = Path(__file__).resolve().parent
PANEL = PACKAGE/'benchmarks/unseen210.json'


def select(panel, case_id):
    for case in panel['cases']:
        if case['case_id'] == case_id:
            return case
    raise ValueError(f'Unknown case {case_id}. Use "priora cases" to list layout IDs.')


def launch_sim(args, panel, case, run, stack):
    """Use a private child process and wait for its own ready message without consuming its RPC connection."""
    native = args.robodojo.resolve(strict=True)
    verify_assets(panel, native, [case])
    fixture = run/'layout.json'
    write_json(fixture, dict(case=case, identity=case_identity(panel, case)))
    scratch = run/'scratch'
    scratch.mkdir()
    python = args.sim_python or sys.executable
    env = os.environ.copy()
    for key in list(env):
        if 'API_KEY' in key or key.startswith(('CODEX_', 'ROLLOUT_', 'WALL_')):
            del env[key]
    env.update(PYTHONPATH=os.pathsep.join(map(str, [PACKAGE.parent,native,native/'XPolicyLab',native/'third_party/curobo'])),
               WALL_NATIVE_IK_ONLY='1', WALL_SHARED_CUROBO='1', WALL_MOTION_SMOOTHING='1',
               WALL_IK_CUDA_GRAPH='0' if case['task'] in ('imitate_sorting_sequence','make_kong') else '1',
               ROLLOUT_MAIN_CAMERA_INPUT='rgbd_all', ROLLOUT_RPC_IDLE_TIMEOUT_SECONDS='86400',
               ROLLOUT_PI05_GRIPPER_MODE='hold_closed', PYTHONUNBUFFERED='1')
    command = [python, '-m', 'priora.robodojo.robodojo_server.server',
               '--kit_args=--/app/hangDetector/enabled=false','--student-backend','wall_oss',
               '--task',case['runtime_task'],'--eval-seed',str(case['eval_seed']),
               '--eval-manifest',str(args.panel.resolve()),'--case-file',str(fixture),
               '--output',str(run/'sim'),'--port',str(args.sim_port)]
    logfile = run/'sim.log'
    logger = stack.enter_context(logfile.open('w'))
    process = subprocess.Popen(command, cwd=scratch, env=env, stdout=logger, stderr=subprocess.STDOUT)

    def stop():
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    stack.callback(stop)
    deadline = time.monotonic()+900
    with logfile.open() as stream:
        while time.monotonic() < deadline:
            line = stream.readline()
            if line.startswith('{'):
                try:
                    if json.loads(line).get('event') == 'ready':
                        return
                except ValueError:
                    pass
            if not line:
                if process.poll() is not None:
                    raise RuntimeError(f'Simulator exited ({process.returncode}). See {logfile}')
                time.sleep(0.25)
    raise TimeoutError(f'Simulator startup exceeded 900 seconds. See {logfile}')


def run_episode(args, panel, case):
    from .robodojo.wall_oss.client import WallTools, WallCodexPolicy, WallPolicyClient
    if args.sim_port == args.policy_port:
        raise ValueError('Give the simulator and policy separate ports')
    codex = shutil.which(args.codex)
    if not codex:
        raise ValueError('Codex CLI executable missing')
    if os.getenv('PRIORA_BASE_URL'):
        if not os.environ.get('PRIORA_API_KEY'):
            raise ValueError('PRIORA_API_KEY is required for the chosen Responses endpoint')
    elif not (Path(os.environ.get('CODEX_HOME', Path.home()/'.codex'))/'auth.json').is_file():
        raise ValueError('Sign in to the Codex CLI first')
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    try:
        card = args.card
        if card is None:
            card = run/'official_card.json'
            fetch(case['task'], card)
        load_card(card, case['task'])
        controller = run/'controller'
        os.environ.update(PRIORA_TASK_CARD=str(card.resolve()), PRIORA_TASK=case['task'],
                          WALL_PRIVATE_CONTROLLER=str(controller), ROLLOUT_MAIN_CAMERA_INPUT='rgbd_all',
                          ROLLOUT_PI05_GRIPPER_MODE='hold_closed',
                          WALL_REQUIRE_WHOLE_TASK_TRIAL='1' if args.whole_task_first else '0')
        with ExitStack() as resources:
            state = resources.enter_context(tempfile.TemporaryDirectory(prefix='priora-codex-'))
            os.environ['ROLLOUT_CODEX_STATE_DIR'] = state
            student = WallPolicyClient(args.policy_port)
            resources.callback(student.close)
            launch_sim(args, panel, case, run, resources)
            policy = WallCodexPolicy(controller/'codex_workspace', codex)
            resources.callback(policy.close)
            tools = WallTools(controller, case['runtime_task'], student, sim_port=args.sim_port,
                              seed=case['layout_id'], max_decisions=0, prompt_sha256=policy.prompt_sha256)
            resources.callback(tools.close)

            def keepalive():
                student.rpc.request('metadata')
                if tools.sim is not None:
                    tools.sim.request('metadata')
            policy.funding_keepalive = keepalive
            policy.run(tools)
            result = json.loads((controller/'result.json').read_text())
            if not result.get('complete') or result.get('evaluation_case') != case_identity(panel, case):
                raise RuntimeError('Missing terminal result for the requested layout')
            print(json.dumps({k:result[k] for k in ('success','step_id','reason','evaluation_case')}, indent=2))
    except BaseException as exc:
        write_json(run/'failure.json', dict(type=type(exc).__name__, message=str(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    for name in ('cases','check','run'):
        p = subs.add_parser(name)
        p.add_argument('--panel', type=Path, default=PANEL)
        if name in ('check','run'):
            p.add_argument('--case', required=True)
            p.add_argument('--robodojo', required=True, type=Path)
        if name == 'run':
            p.add_argument('--card', type=Path, help='Local pinned official card, otherwise fetch from RoboDojo')
            p.add_argument('--sim-python', help='Isaac Sim 5.1 interpreter (default: current Python)')
            p.add_argument('--output', required=True, type=Path, help='Fresh episode directory')
            p.add_argument('--codex', default='codex')
            p.add_argument('--sim-port', type=int, default=19113)
            p.add_argument('--policy-port', type=int, default=19660)
            p.add_argument('--whole-task-first', action='store_true', help='Enforce one full-task proposal even for multi-object tasks')
    args = parser.parse_args()
    panel = read_panel(args.panel)
    if args.command == 'cases':
        print('\n'.join(c['case_id'] for c in panel['cases']))
        return
    case = select(panel, args.case)
    if args.command == 'check':
        verify_assets(panel, args.robodojo, [case])
        print(json.dumps(case_identity(panel, case), indent=2))
    else:
        run_episode(args, panel, case)
