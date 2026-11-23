"""Launch RoboDojo's native evaluation environment behind the existing RPC."""
import argparse
import importlib
import json
import os
import shutil
import traceback
from pathlib import Path


class NoPolicyConnection:
    """Native reset callback only; inference is owned by the Codex tool host."""
    def __init__(self, **kwargs):
        pass

    def call(self, func_name, **kwargs):
        if func_name != 'reset':
            raise RuntimeError('Native autonomous policy loop is disabled')

    def close(self):
        pass


def guard_headless_viewport(simulation_context, get_active_viewport):
    """IsaacLab can request its UI camera before Kit creates it on a headless stage."""
    set_view = simulation_context.set_camera_view

    def set_camera_view(eye, target, camera_prim_path="/OmniverseKit_Persp", viewport_api=None):
        if camera_prim_path == "/OmniverseKit_Persp":
            if viewport_api is None:
                viewport_api = get_active_viewport()
            stage = viewport_api.stage if viewport_api is not None else None
            if stage is None or not stage.GetPrimAtPath(camera_prim_path).IsValid():
                print(json.dumps(dict(event="headless_viewport_camera_unavailable",
                                      camera=camera_prim_path)), flush=True)
                return
            # Kit can expose a valid viewport prim before its camera commands are ready.
            try:
                return set_view(eye, target, camera_prim_path, viewport_api=viewport_api)
            except AttributeError as error:
                if error.name != "do" or error.obj is not None:
                    raise
                print(json.dumps(dict(event="headless_viewport_camera_unavailable",
                                      camera=camera_prim_path, reason="kit_command_unavailable")), flush=True)
                return
        return set_view(eye, target, camera_prim_path, viewport_api=viewport_api)

    simulation_context.set_camera_view = set_camera_view


def main():
    import cv2  # Load before Isaac's extension dependency paths.
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', default='build_tower')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port', type=int, default=19113)
    p.add_argument('--eval-seed', type=int, default=0, help='Published layout group, NOT the reset layout ID')
    p.add_argument('--eval-manifest', type=Path)
    p.add_argument('--case-file', type=Path)
    p.add_argument('--student-backend', choices=('pi05', 'wall_oss'), default='pi05')
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args(); args.headless = True; args.enable_cameras = True
    from ..evaluation import read_panel, verify_assets, case_identity
    from ..io import require
    from env.global_configs import ROOT_DIR, BENCHMARK
    registry = importlib.import_module(f'task.{BENCHMARK}.task_registry')
    evaluation_case = None
    if args.eval_manifest:
        panel = read_panel(args.eval_manifest, os.environ.get('ROLLOUT_EVAL_MANIFEST_SHA256'))
        selected = json.loads(args.case_file.read_text())
        evaluation_case = selected['case']
        require(evaluation_case in panel['cases'] and selected['identity'] == case_identity(panel, evaluation_case),
                'Selected case differs from the frozen panel')
        require((args.task, args.eval_seed) == (evaluation_case['runtime_task'], evaluation_case['eval_seed']),
                'CLI task/eval_seed differs from selected case')
        verify_assets(panel, ROOT_DIR, [evaluation_case])
        args.output.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(ROOT_DIR)/evaluation_case['layout']['path'], args.output/'scene_layout.json')
    # Match native main.py: monitor Articulation tasks before AppLauncher.
    import yaml
    task_config_path = registry.task_config_path(str(Path(ROOT_DIR)/'task'/BENCHMARK/'config'), args.task)
    with open(task_config_path) as stream:
        task_values = yaml.safe_load(stream) or {}
    enable_monitor = bool(task_values.get('Articulation'))
    if enable_monitor:
        from src.eval_client.physx_warning_monitor import get_monitor
        get_monitor().start(enabled=True)
    # Match the selected native entrypoint. Historical report code predates
    # this camera synchronization helper and keeps its original Kit settings.
    if (Path(ROOT_DIR)/'env/camera_manager/capture/render_sync.py').is_file():
        from env.camera_manager.capture.render_sync import add_zero_delay_kit_args
        add_zero_delay_kit_args(args)
    app = AppLauncher(args).app
    if args.student_backend == 'wall_oss':
        import carb.settings
        detector = carb.settings.get_settings().get_as_bool('/app/hangDetector/enabled')
        print(json.dumps(dict(event='kit_hang_detector', enabled=detector)), flush=True)
    env = session = None
    try:
        from omegaconf import OmegaConf
        from env.global_configs import ENV_CONFIG_PATH, ROOT_DIR, BENCHMARK
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization
        from src.eval_client import eval_env
        from .rpc import serve
        from .session import RoboDojoSession
        if args.student_backend == 'wall_oss':
            if args.headless:
                from isaaclab.sim import simulation_context
                try:
                    from omni.kit.viewport.utility import get_active_viewport
                except ImportError:
                    pass  # Isaac Sim already skips viewport setup without this extension.
                else:
                    guard_headless_viewport(simulation_context, get_active_viewport)
            from ..wall_oss.session import WallSession as RoboDojoSession
            if os.environ.get('WALL_NATIVE_IK_ONLY') == '1':
                from ..wall_oss.ik_only import install as install_native_ik
                print(json.dumps(dict(event='native_ik_configuration', **install_native_ik())), flush=True)
        # JustRUA's opt-in spawn hook must precede robot creation/physics reset.
        from ..grasp_feedback import install as install_grasp_feedback
        grasp_setup = install_grasp_feedback()
        root = Path(ENV_CONFIG_PATH)
        evaluation = load_yaml(str(root/'arx_x5.yml'))
        evaluation.update(task_name=args.task, num_envs=1, device_id=0, eval_batch=False,
            policy_name='Pi_05', additional_info='codex_hybrid', seed=args.eval_seed,
            physx_monitor_enabled=enable_monitor)
        values = {key: load_yaml(str(root/key/(evaluation['config'][key]+'.yml')))
                  for key in ('sim', 'scene', 'camera', 'robot')}
        values.update(eval_cfg=evaluation, deploy_cfg=dict(port=1, policy_name='Pi_05'),
            task_env=load_yaml(registry.task_config_path(str(Path(ROOT_DIR)/'task'/BENCHMARK/'config'), args.task)))
        cfg = OmegaConf.create(values)
        cfg.sim.scene.num_envs = 1
        cfg = process_randomization(cfg)
        cfg, _ = process_config(cfg, task_name=args.task)
        cfg.eval_cfg.eval_num = 1
        cfg.camera.default_frequency = cfg.eval_cfg.observation.collect_freq
        from ..rgbd import configure_depth
        configure_depth(cfg)
        cfg.sim.seed = [0]  # Native startup; the controller later resets the selected layout_id.
        # WALL controls the policy arms. Native support arms may only replay
        # joint demonstrations and must retain their own planner requirement.
        from ..wall_oss.planner_roles import configure_planner_roles
        configure_planner_roles(cfg.robot.robots, args.student_backend)
        print(json.dumps(dict(event='robot_planner_roles', robots=[
            dict(name=robot.get('robot_name'), role=robot.get('type', 'policy'),
                 need_planner=bool(robot.get('need_planner')))
            for robot in cfg.robot.robots])), flush=True)
        original = eval_env.WsModelClient
        try:
            eval_env.WsModelClient = NoPolicyConnection
            env = eval_env.create_eval_env(cfg, app)
        finally:
            eval_env.WsModelClient = original
        if evaluation_case:
            resolved = env.seed_manager.seed_info[evaluation_case['layout_id']]['scene_layout']
            require(Path(resolved).resolve() == (Path(ROOT_DIR)/evaluation_case['layout']['path']).resolve(),
                    'Native SeedManager resolved a different layout')
        from ..io import write_json
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output/'grasp_sensor_setup.json', grasp_setup)
        write_json(args.output/'resolved_config.json', OmegaConf.to_container(cfg, resolve=True))
        session = RoboDojoSession(env, args.output, args.task,
            evaluation_identity=selected['identity'] if evaluation_case else None)
        if args.student_backend == 'wall_oss' and os.environ.get('WALL_INIT_LOCK_FD'):
            import fcntl
            lock_fd = int(os.environ.pop('WALL_INIT_LOCK_FD'))
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        serve(session, args.port)
    except BaseException:
        # SimulationApp.close may exit before Python prints an uncaught error.
        # Persist the original failure before native cleanup can obscure it.
        traceback.print_exc()
        raise
    finally:
        if session:
            try:
                session._write_summary('server_close')
            except Exception:
                traceback.print_exc()
            finally:
                if session.realtime is not None:
                    try:
                        session.realtime.finalize(dict(reason='server_close', step_id=session.step_id,
                            terminated=session.terminated, truncated=session.truncated,
                            success=session.success))
                    except Exception:
                        traceback.print_exc()  # Preserve uncertain execution; never replay at shutdown.
        try:
            if env:
                env.close()
        finally:
            app.close()


if __name__ == '__main__':
    main()
