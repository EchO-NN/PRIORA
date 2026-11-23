"""Add three rollout services to a full, persistent Codex policy agent."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from ..io import InputError, write_json
from ..grasp_feedback import GRIPPER_GUIDE, teacher_packet as grasp_teacher_packet
from ..teacher_history import TeacherEpisodeHistory, GUIDE as EPISODE_HISTORY_GUIDE, append_dialogue
from ..prompt_context import CONTEXT_VERSION
from ..settings import MODEL, EFFORT, PROVIDER, DEFAULT_CODEX_IMAGE_MAX_EDGE
from ..robodojo_server.client import RoboDojoTools
from ..method import evaluation_method
from .schema import _obj, response_schema, direct_response_schema
from .transport import StdioAppServer
from .workspace import prepare_workspace
from .environment_memory import EnvironmentMemory, VERSION as ENVIRONMENT_MEMORY_VERSION
from .rigid_spatial import RigidSpatial, VERSION as SPATIAL_VERSION, CAMERAS as SpatialCameras
from .near_field_memory import VERSION as NEAR_FIELD_MEMORY_VERSION
from .image_preview import image_max_edge, prepare_image, IMAGE_MIME_TYPE, JPEG_QUALITY
from .numeric_preview import robot_numeric_preview
from .camera_preview import compact_camera_packet, GUIDE as CAMERA_INPUT_GUIDE
from .candidate_preview import add_candidate_images, prepare_candidate_image, GUIDE as CANDIDATE_IMAGE_GUIDE
from .prompt_summary import EFFICIENCY_GUIDE, SPATIAL_GUIDE, VERSION as PROMPT_VERSION
from .usage import write_usage
from .quota import Quota
from ..rgbd import main_camera_input, depth_cameras
from .network_recovery import (NETWORK_CONTINUE_DELAYS, CAPACITY_CONTINUE_DELAYS,
                               NETWORK_RECOVERY_VERSION,
                               ACTIVE_TURN_GRACE_WINDOWS, active_turn,
                               closed_terminal_turn, is_network_error, is_capacity_error)

SKILL_ROOT = Path(__file__).parent
CONTROLLER_VERSION = 'capacity_wait_until_available_v6'
NATIVE_WORK_ITEMS = frozenset(('commandExecution', 'fileChange', 'imageView',
    'webSearch', 'mcpToolCall', 'collabAgentToolCall'))


def rejected_input(error, rollout, arguments, count):
    """Feedback only for InputError's pre-execution rejection contract.

    Never change arguments, clip a target, retry a handler or reset simulation.
    Other exceptions (including uncertain RPC outcomes) remain fatal.
    """
    packet = dict(error=str(error), error_type='recoverable_tool_input',
        no_execution=True, retryable=True, controller_version=CONTROLLER_VERSION,
        rejected_calls=count, step_id=rollout.tick, next_call=rollout.next_call(),
        correction='Correct the arguments and retry next_call in this same episode. '
                   'Do not reset or replay already executed actions.')
    if getattr(rollout, 'observation_path', None):
        packet['observation_path'] = str(rollout.observation_path)
    if getattr(error, 'waypoint_reachability', None) is not None:
        packet['waypoint_reachability'] = error.waypoint_reachability
        packet['waypoint_preflight_path'] = error.waypoint_preflight_path
        packet['correction'] = (
            'No robot action executed. Inspect per-waypoint errors and the available step budget. '
            'Revise targets/orientations or distribute steps differently (total <=30), then retry '
            'the SAME proposal/request_id. Do not reset, consume steps, or claim the target is '
            'globally impossible merely because the bounded numerical check did not pass.')
    request = getattr(rollout, 'request', None)
    if isinstance(request, dict) and rollout.phase in ('act', 'execute'):
        packet['request_id'] = request.get('request_id')
        packet['current_eef'] = request.get('current_eef')
    return packet


def toml_value(value):
    """Codex -c values are TOML; JSON objects/quoted dotted keys are not equivalent."""
    if isinstance(value, dict):
        return '{' + ', '.join(json.dumps(k) + ' = ' + toml_value(v) for k, v in value.items()) + '}'
    return json.dumps(value)


def agent_config(audit, agent):
    # Keep Codex's normal tool/skill configuration. These two capabilities are
    # essential, even if an inherited profile had disabled them previously.
    config = dict(model=MODEL, model_provider=PROVIDER, model_reasoning_effort=EFFORT,
        sqlite_home=str(Path(os.environ.get('ROLLOUT_CODEX_STATE_DIR', os.environ.get('CODEX_HOME', str(audit))))/'runtime_db'),
        log_dir=str(Path(os.environ.get('ROLLOUT_CODEX_STATE_DIR', os.environ.get('CODEX_HOME', str(audit))))/'runtime_logs'),
        default_permissions='rollout_agent', **{
            'features.shell_tool': True, 'features.view_image': True,
            'shell_environment_policy.inherit': 'all',
            'permissions.rollout_agent.extends': ':workspace',
            # The parent contains host-owned RPC logs and experiment artifacts.
            # Only the agent directory is a writable runtime workspace root.
            'permissions.rollout_agent.filesystem': {
                str(audit.parent): 'read', str(agent): 'write'},
        })
    config.update({'features.memories': False, 'memories.generate_memories': False,
                   'memories.use_memories': False,
                   'shell_environment_policy.exclude': ['*API_KEY', '*TOKEN', '*PROXY', '*proxy']})
    if PROVIDER == 'openai':
        config['forced_login_method'] = 'chatgpt'
    else:
        config['model_providers.priora_endpoint'] = {
            'name': 'PRIORA endpoint', 'base_url': os.environ['PRIORA_BASE_URL'],
            'wire_api': 'responses', 'env_key': 'PRIORA_API_KEY',
            'requires_openai_auth': False, 'stream_max_retries': 1, 'request_max_retries': 1}
    return config


def tool_specs(method='wall_plus_gpt'):
    string = {'type': 'string'}
    def spec(name, description, **properties):
        return dict(type='function', name=name, description=description,
                    inputSchema=_obj(properties))
    if method == 'gpt_only':
        return [spec('robodojo_start', 'Start the requested episode once; return current observation and next_call.',
                     task=string, output_dir=string),
                spec('robodojo_act', 'Blocking: execute GPT-authored bounded dual-arm EEF targets '
                     '(1-5 steps), then return RGB/proprio and native outcome. '
                     'No pi05 inference, proposals or direct joint-action commands exist.',
                     observation_path=string, response=direct_response_schema(), output_dir=string)]
    return [
        spec('robodojo_start', 'Start the requested episode once; return current observation and next_call.',
             task=string, output_dir=string),
        spec('pi05_infer', 'Infer a fresh pi05 H50 proposal and robot-only FK from the latest observation. No contact/object prediction; follow next_call.',
             observation_path=string, output_dir=string),
        spec('robodojo_execute', 'Validate and execute the current student/edit/EEF decision; return acknowledged motion and fresh observation. Follow the action contract for waypoint timing and reachability. Rejected input executes nothing; partial execution must not be replayed.',
             proposal_path=string, response=response_schema(), output_dir=string),
        spec('spatial_record', 'Register a visible rigid gripper surface landmark at original RGB pixel [u,v]; not camera calibration or an object landmark. No motion.',
             observation_path=string, camera={'type': 'string', 'enum': list(SpatialCameras)},
             arm={'type': 'string', 'enum': ['left', 'right']}, landmark=string,
             pixel={'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}),
        spec('spatial_query', 'Query a visible surface at original pixel [u,v] using current raw depth/geometry; no prior record required. pixel=[] projects the chosen EEF origin/registered landmark. Not a grasp target. No motion.',
             observation_path=string, camera={'type': 'string', 'enum': list(SpatialCameras)},
             arm={'type': 'string', 'enum': ['left', 'right']}, landmark=string,
             pixel={'type': 'array', 'items': {'type': 'number'}, 'maxItems': 2}),
        spec('spatial_remember', 'Save a named current local RGB/depth/XYZ patch before occlusion; use original pixels and valid 0.1..10 m depth. Historical evidence, not a future grasp target. No motion.',
             observation_path=string, camera={'type': 'string', 'enum': list(SpatialCameras)},
             label=string,
             pixel={'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2},
             radius={'type': 'integer', 'minimum': 1, 'maximum': 32}),
        spec('spatial_recall', 'Retrieve a dated patch; memory_id="" lists entries. Check age and visual validity before use. No motion.',
             observation_path=string, memory_id=string),
    ]


def content_items(packet, *, images, max_image_edge=DEFAULT_CODEX_IMAGE_MAX_EDGE):
    # This boundary is downstream of observation capture and policy inference.
    # Copy metadata only: never change packet, source PNG/NPZ or policy arrays.
    visible_packet = packet
    attachments = []
    if images:
        visible_packet = dict(packet, images=[])
        for item in packet.get('images', []):
            if item.get('teacher_history'):
                # Already encoded once at capture. Keep per-frame chronology
                # adjacent to its actual JPEG, without altering camera pixels.
                payload = Path(item['path']).read_bytes()
                if hashlib.sha256(payload).hexdigest() != item['sha256']:
                    raise ValueError('Teacher history image changed after capture')
                visible_packet['images'].append(dict(item))
                attachments.append(dict(type='inputText', text=item['label']))
                attachments.append(dict(type='inputImage', imageUrl='data:image/jpeg;base64,' +
                    base64.b64encode(payload).decode('ascii')))
                continue
            if packet.get('memory_tool') == 'get_images' and 'content' in item:
                # Memory retrieval already enforced native resolution, PNG depth,
                # exact bytes and per-query limits. Do not re-encode those bytes.
                payload = item['content']
                descriptor = {key: value for key, value in item.items() if key != 'content'}
                if payload.get('type') != 'image' or payload.get('mimeType') not in ('image/jpeg', 'image/png'):
                    raise ValueError('Invalid host memory image payload')
                visible_packet['images'].append(descriptor)
                attachments.append(dict(type='inputImage', imageUrl='data:' +
                    payload['mimeType'] + ';base64,' + payload['data']))
                continue
            descriptor, payload = prepare_image(item, max_image_edge)
            visible_packet['images'].append(descriptor)
            data = base64.b64encode(payload).decode('ascii')
            if packet.get('gpt_episode_history'):
                attachments.append(dict(type='inputText', text='CURRENT observation | camera=' +
                    str(item.get('camera', 'unknown')) + ' | step=' + str(packet.get('step_id', 'unknown'))))
            attachments.append(dict(type='inputImage', imageUrl='data:' + IMAGE_MIME_TYPE + ';base64,' + data))
        if packet.get('candidate_trajectory_images'):
            visible_packet['candidate_trajectory_images'] = []
            candidate = packet['candidate_trajectory']
            for item in packet['candidate_trajectory_images']:
                descriptor, payload = prepare_candidate_image(item, max_image_edge)
                visible_packet['candidate_trajectory_images'].append(descriptor)
                attachments.append(dict(type='inputText', text='UNEXECUTED CANDIDATE | camera=' +
                    item['camera'] + ' | request=' + candidate['request_id'] +
                    ' | step=' + str(candidate['step_id']) +
                    ' | t=' + str(candidate['simulation_time_s']) + 's' +
                    ' | prefix_steps=' + str(candidate['prefix_steps'])))
                attachments.append(dict(type='inputImage', imageUrl='data:' + IMAGE_MIME_TYPE +
                    ';base64,' + base64.b64encode(payload).decode('ascii')))
    visible_packet = compact_camera_packet(robot_numeric_preview(visible_packet))
    return [dict(type='inputText', text=json.dumps(visible_packet, separators=(',', ':'))), *attachments]


class CodexPolicy:
    funding_keepalive = None

    def customize_prompt(self, prompt):
        return prompt

    def customize_tool_specs(self, specs):
        return specs

    def __init__(self, workspace, codex, *, timeout=900, transport_factory=StdioAppServer,
                 method='wall_plus_gpt', realtime=False, onejev=False, funding_keepalive=None):
        self.funding_keepalive = funding_keepalive
        self.method = evaluation_method(method)
        direct = self.method == 'gpt_only'
        if realtime or onejev or direct:
            raise ValueError('PRIORA requires the WALL hybrid controller')
        skill_root = SKILL_ROOT
        self.image_max_edge = image_max_edge(os.environ.get(
            'CODEX_IMAGE_MAX_EDGE', str(DEFAULT_CODEX_IMAGE_MAX_EDGE)))
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=False)
        self.episode_history = TeacherEpisodeHistory(self.workspace/'episode_history_2hz')
        self.dialogue_path = self.workspace/'gpt_public_dialogue.jsonl'
        write_json(self.workspace.parent/'controller_protocol.json', dict(
            controller_version=CONTROLLER_VERSION, context_version=CONTEXT_VERSION,
            recoverable_input_error_limit=None, validation_unchanged=direct,
            active_view_version=None,
            ordinary_correction_gate_unchanged=True, recovery_changes_validation=False,
            same_thread=True, simulator_reset_on_input_error=False,
            action_replay_on_input_error=False,
            active_turn_wait=dict(extra_idle_windows=ACTIVE_TURN_GRACE_WINDOWS,
                seconds_per_window=timeout, matching_active_turn_required=True,
                 provider_error_required=False, new_turn_started=False, physical_actions_replayed=0),
            normal_turn_continuation=dict(progress='acknowledged_physical_steps',
                max_consecutive_no_progress_turns=3, total_continuation_limit=None)))
        self.agent_workspace = prepare_workspace(self.workspace, skill_root, method=self.method)
        self.environment_memory = EnvironmentMemory(self.agent_workspace, method=self.method)
        environment_guidance = self.environment_memory.initialize()
        self.spatial = None if direct else RigidSpatial(self.workspace.parent)
        self.timeout = timeout
        skill = (skill_root/'SKILL.md').read_text()
        gate = '' if direct else (skill_root/'gate_prompt.md').read_text()
        context = (skill_root/'context/teacher_context.md').read_text() if direct else ''
        self.prompt = (environment_guidance + '\n\n' + skill + ('' if direct else '\n\n# Baseline correction gate\n\n' + gate)
                       + '\n\n' + context + '\n\nAgent working directory: '
                       + str(self.agent_workspace)
                       + '\nResolve context/ and workspace.json relative to that directory.\n')
        self.prompt += '\n\n' + EFFICIENCY_GUIDE
        if not direct:
            self.prompt += '\n\n' + SPATIAL_GUIDE
        self.prompt += '\n\n' + CAMERA_INPUT_GUIDE + '\n'
        if not direct:
            self.prompt += '\n\n' + CANDIDATE_IMAGE_GUIDE + '\n'
        self.prompt += '\n\n' + GRIPPER_GUIDE + '\n\n' + EPISODE_HISTORY_GUIDE
        self.prompt += ('\nEpisode image index: ' + str(self.episode_history.root/'index.json') +
                        '\nPublic dialogue archive: ' + str(self.dialogue_path) + '\n')
        self.prompt = self.customize_prompt(self.prompt)
        self.prompt_sha256 = hashlib.sha256(self.prompt.encode()).hexdigest()
        (self.workspace/'SKILL.md').write_text(skill)
        if not direct:
            (self.workspace/'gate_prompt.md').write_text(gate)
        (self.workspace/'PROMPT.md').write_text(self.prompt)
        specs = tool_specs(self.method)
        specs = self.customize_tool_specs(specs)
        write_json(self.workspace/'tools.json', specs)
        version = subprocess.run([codex, '--version'], capture_output=True, text=True, check=True).stdout.strip()
        config = agent_config(self.workspace, self.agent_workspace)
        self.codex, self.transport_factory, self.provider = codex, transport_factory, PROVIDER
        child_env = None
        self.launch_config = config
        argv = [codex, 'app-server', '--stdio', '--strict-config']
        for key, value in config.items():
            argv += ['-c', key + '=' + toml_value(value)]
        write_json(self.workspace/'launch.json', dict(argv=argv, config=config))
        self.transport = transport_factory(argv, self.workspace,
            **({'env': child_env} if child_env is not None else {}))
        try:
            self.transport.request('initialize', dict(
                clientInfo=dict(name='robodojo-hybrid-policy', version='1'),
                capabilities=dict(experimentalApi=True)), timeout)
            self.transport.notify('initialized', {})
            self.quota = Quota(self.transport, PROVIDER, os.getenv('PRIORA_QUOTA_FLOOR'))
            response = self._start_rpc('thread/start', dict(
                cwd=str(self.agent_workspace), model=MODEL, modelProvider=PROVIDER,
                config=dict(model_reasoning_effort=EFFORT), developerInstructions=self.prompt,
                dynamicTools=specs, ephemeral=False,
                allowProviderModelFallback=False, approvalPolicy='never',
                approvalsReviewer='auto_review', permissions='rollout_agent',
                runtimeWorkspaceRoots=[str(self.agent_workspace)]))
            if response.get('model') != MODEL or response.get('reasoningEffort') != EFFORT:
                raise RuntimeError(f'Codex model/effort differs from the required {MODEL}/{EFFORT}: '
                                   f'{response.get("model")}/{response.get("reasoningEffort")}')
            if response['thread'].get('modelProvider', PROVIDER) != PROVIDER:
                raise RuntimeError('Codex started a different provider than the selected account route')
            self.thread_id = response['thread']['id']
            write_json(self.workspace/'worker.json', dict(model=MODEL, model_provider=PROVIDER,
                controller_version=CONTROLLER_VERSION,
                prompt_version=PROMPT_VERSION if not direct else 'gpt_only_existing',
                evaluation_method=self.method,
                reasoning_effort=EFFORT, context_version=CONTEXT_VERSION,
                codex_version=version, thread_id=self.thread_id, pid=self.transport.process.pid,
                prompt_sha256=self.prompt_sha256, no_rollback=True, tools=[s['name'] for s in specs],
                service_tools_are_additive=True, agent_workspace=str(self.agent_workspace),
                environment_memory_version=ENVIRONMENT_MEMORY_VERSION,
                environment_memory_path=str(self.workspace/'ENVIRONMENT_MEMORY.md'),
                runtime_state_path=str(self.workspace/'runtime_state.json'),
                native_tools='Codex defaults; shell and image viewing explicitly enabled',
                network_recovery=dict(revision=NETWORK_RECOVERY_VERSION,
                    backoff_seconds=list(NETWORK_CONTINUE_DELAYS), restart_simulator=False,
                    capacity_backoff_seconds=list(CAPACITY_CONTINUE_DELAYS),
                    capacity_attempt_limit=None, repeat_last_capacity_delay=True,
                    new_thread=False, replay_actions=False),
                request_guard=os.environ.get('ROLLOUT_PACKY_GUARD_URL'),
                codex_image_max_edge=self.image_max_edge, policy_images_resized=False,
                teacher_image_format='JPEG', teacher_jpeg_quality=JPEG_QUALITY,
                spatial_memory_version=None if direct else SPATIAL_VERSION,
                near_field_memory_version=None if direct else NEAR_FIELD_MEMORY_VERSION,
                active_view_version=None,
                main_camera_input=main_camera_input(),
                gpt_episode_history=dict(hz=2, scope='whole_episode', camera='cam_high',
                    delivery='incremental_same_thread', clear_on_goal_change=False),
                baseline_full_conversation_available=False))
        except BaseException:
            self.close()
            raise

    def _turn(self, text):
        self.quota.wait(self.funding_keepalive)
        if getattr(self, 'dialogue_path', None):
            append_dialogue(self.dialogue_path, dict(role='user', text=text,
                                                   thread_id=self.thread_id))
        params = dict(threadId=self.thread_id,
            model=MODEL, effort=EFFORT, approvalPolicy='never', approvalsReviewer='auto_review',
            permissions='rollout_agent', cwd=str(self.agent_workspace),
            runtimeWorkspaceRoots=[str(self.agent_workspace)], input=[dict(type='text', text=text)])
        return self._start_rpc('turn/start', params)['turn']['id']

    def _start_rpc(self, method, params):
        return self.transport.request(method, params, self.timeout)

    def _network_continue(self, rollout, old_turn, error, consecutive, serial):
        capacity = is_capacity_error(error)
        delays = CAPACITY_CONTINUE_DELAYS if capacity else NETWORK_CONTINUE_DELAYS
        if not is_network_error(error) or (not capacity and consecutive >= len(delays)):
            raise RuntimeError(f'Codex network continue exhausted or ineligible: {error}')
        delay = delays[min(consecutive, len(delays)-1)]
        record = dict(revision=NETWORK_RECOVERY_VERSION, thread_id=self.thread_id,
            previous_turn_id=old_turn, error=error, step_id=rollout.tick,
            phase=rollout.phase, next_call=rollout.next_call(), delay_seconds=delay,
            consecutive_attempt=consecutive+1, attempt_limit=None if capacity else len(delays),
            status='waiting', physical_actions_replayed=0)
        record['cause'] = 'provider_temporary_capacity' if capacity else 'network'
        path = self.workspace/f'network_continue_{serial:04d}.json'
        write_json(path, record)
        print(json.dumps(dict(event='codex_network_continue_wait', **record)), flush=True)
        # Keep Python/the simulator/observation alive. SIGTERM still exits normally.
        for _ in range(delay):
            time.sleep(1)
        text = ('Continue the same rollout. The previous turn ended because of a network error. '
                'The simulator and recorded actions are unchanged; do not reset or repeat executed actions. '
                f'Current step_id={rollout.tick}. Use the latest recorded observation and next_call: '
                + json.dumps(rollout.next_call()))
        if getattr(rollout, 'observation_path', None):
            text += '\nLatest observation: ' + str(rollout.observation_path)
        record['status'] = 'turn_start_requested'
        write_json(path, record)  # Ambiguous RPC must never be reissued blindly.
        new_turn = self._turn(text)
        record.update(status='continue_sent', new_turn_id=new_turn)
        write_json(path, record)
        print(json.dumps(dict(event='codex_network_continue_sent', thread_id=self.thread_id,
            old_turn_id=old_turn, turn_id=new_turn, step_id=rollout.tick)), flush=True)
        return new_turn

    def _environment_state(self, rollout, *, tool=None, packet=None):
        memory = getattr(self, 'environment_memory', None)
        if memory is None:
            return None
        try:
            return memory.refresh(rollout, tool=tool, packet=packet)
        except Exception as error:
            # Reporting is downstream of acknowledged execution. It must never
            # turn a completed action into a safe-to-retry InputError.
            return dict(version=ENVIRONMENT_MEMORY_VERSION, available=False,
                error=f'{type(error).__name__}: {error}', no_action_retry=True,
                phase=rollout.phase, step_id=rollout.tick,
                next_call=rollout.next_call() if rollout.phase != 'done' else None,
                instruction='Environment memory could not be refreshed. The service '
                    'receipt remains authoritative. Do not replay completed actions or '
                    'use an older runtime_state.json as current state.')

    def run(self, rollout):
        environment_state = self._environment_state(rollout)
        turn_id = self._turn('Act as the autonomous policy agent for this single simulation rollout. '
            'Use your normal file, image, shell/code and planning tools as useful, alongside '
            'the rollout service tools. Read workspace.json for paths and interpreter; '
            'keep working notes and analysis in your agent workspace. Use English for every '
            'public explanation, assessment, progress update, note, and final report. '
            'The user authorizes sending this episode\'s three RGB images, proprio, robot-only FK, '
            'task text and same-episode history to the configured teacher service for online decisions. '
            'First call: ' + json.dumps(rollout.next_call())
            + ('\nHost-maintained environment_state: ' + json.dumps(environment_state)
               if environment_state is not None else ''))
        call_index = 0
        errors = 0
        continuations = 0
        no_progress_turns = 0
        last_completed_tick = rollout.tick
        network_error = None
        capacity_retry_in_progress = False
        active_grace_windows = 0
        idle_probe_serial = 0
        network_consecutive = network_serial = 0
        completed_turns = 0
        token_limit = int(os.environ.get('CODEX_MAX_TOTAL_TOKENS', '0'))
        deadline = time.monotonic() + self.timeout
        account_interrupted_turn = None
        while True:
            held = None
            if (rollout.phase != "done" and account_interrupted_turn != turn_id
                    and self.quota.exhausted()):
                self.transport.request("turn/interrupt", {"threadId": self.thread_id, "turnId": turn_id}, 30)
                held = turn_id
                account_interrupted_turn=held
                deadline=time.monotonic()+self.timeout
            remaining = deadline - time.monotonic()
            try:
                if remaining <= 0:
                    raise TimeoutError('Codex made no completed service or native tool call within the configured timeout')
                try:
                    event = self.transport.next_message(min(remaining,2))
                except TimeoutError:
                    if time.monotonic() < deadline:
                        continue
                    raise
            except TimeoutError:
                # Silence is not evidence that a model turn ended. Always probe
                # the same thread/turn; this never invokes a robot service.
                response = self.transport.request('thread/read', dict(
                    threadId=self.thread_id, includeTurns=True), min(30, self.timeout))
                idle_probe_serial += 1
                if rollout.phase == 'done':
                    return  # Native completion requires no additional paid turn.
                turn = closed_terminal_turn(response, self.thread_id, turn_id)
                if turn is None and account_interrupted_turn == turn_id:
                    thread = response.get('thread') or {}
                    turns = thread.get('turns') or []
                    if (thread.get('id') == self.thread_id and (thread.get('status') or {}).get('type') == 'idle'
                            and turns and turns[-1].get('id') == turn_id and turns[-1].get('status') == 'interrupted'):
                        turn = turns[-1]
                if turn is None:
                    if active_turn(response, self.thread_id, turn_id):
                        if active_grace_windows >= ACTIVE_TURN_GRACE_WINDOWS:
                            raise TimeoutError('Codex turn remained active after '
                                'the bounded extra wait; no new turn or robot action was issued')
                        active_grace_windows += 1
                        capacity = capacity_retry_in_progress and is_capacity_error(network_error)
                        prefix = 'capacity_wait' if capacity else 'active_wait'
                        record = dict(event='codex_capacity_wait_active' if capacity else 'codex_active_turn_wait',
                            thread_id=self.thread_id, turn_id=turn_id, step_id=rollout.tick,
                            window=active_grace_windows, max_extra_windows=ACTIVE_TURN_GRACE_WINDOWS,
                            cause='provider_capacity_retry' if capacity else 'silent_active_turn',
                            wait_seconds=self.timeout, new_turn_started=False, physical_actions_replayed=0)
                        write_json(self.workspace/f'{prefix}_{idle_probe_serial:04d}_{active_grace_windows}.json', record)
                        print(json.dumps(record), flush=True)
                        deadline = time.monotonic() + self.timeout
                        continue
                    raise
                write_json(self.workspace/f'terminal_probe_{idle_probe_serial:04d}.json', dict(
                    thread_id=self.thread_id, turn_id=turn_id, status=turn['status'],
                    step_id=rollout.tick, recovered_terminal_notification=True,
                    physical_actions_replayed=0))
                event = dict(method='turn/completed', params=dict(threadId=self.thread_id, turn=turn))
            method, params = event.get('method'), event.get('params', {})
            if method == 'account/rateLimits/updated':
                self.quota.update(params)
            elif method == 'thread/tokenUsage/updated' and params.get('threadId') == self.thread_id:
                usage = params.get('tokenUsage', {})
                total_usage = write_usage(self, usage, token_limit, MODEL, EFFORT)
                print(json.dumps(dict(event='token_usage', total=total_usage,
                    provider=getattr(self, 'provider', PROVIDER))), flush=True)
                if token_limit and total_usage.get('totalTokens', 0) >= token_limit:
                    raise RuntimeError('Configured total-token budget reached; stopping this episode')
            elif method == 'item/tool/call':
                if params.get('threadId') != self.thread_id or params.get('turnId') != turn_id:
                    raise RuntimeError('Tool call belongs to another thread or turn')
                if account_interrupted_turn==turn_id:
                    self.transport.reply(event['id'],dict(success=False,contentItems=[dict(type='inputText',text='Interrupted before execution; no action ran.')]))
                    continue
                name, arguments = params['tool'], params['arguments']
                print(json.dumps(dict(event='tool_start', tool=name, call=call_index,
                                      phase=rollout.phase, step_id=rollout.tick)), flush=True)
                audit = dict(call_id=params['callId'], tool=name, arguments=arguments)
                write_json(self.workspace/f'call_{call_index:04d}_request.json', audit)
                handlers = (dict(robodojo_start=rollout.start, robodojo_act=rollout.act)
                    if self.method == 'gpt_only' else
                    dict(robodojo_start=rollout.start, pi05_infer=rollout.infer, robodojo_execute=rollout.execute))
                handlers.update(getattr(rollout, 'service_handlers', {}))
                spatial = getattr(self, 'spatial', None)
                if spatial is not None:
                    handlers.update(spatial_record=lambda **kw: spatial.record(rollout, **kw),
                                    spatial_query=lambda **kw: spatial.query(rollout, **kw),
                                    spatial_remember=lambda **kw: spatial.remember(rollout, **kw),
                                    spatial_recall=lambda **kw: spatial.recall(rollout, **kw))
                try:
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError as error:
                            raise InputError(f'Tool arguments must be a JSON object: {error}') from error
                    if name not in handlers:
                        raise InputError('Unknown host service; native Codex tools are executed by app-server')
                    if not isinstance(arguments, dict):
                        raise InputError('Tool arguments must be a JSON object')
                    packet = handlers[name](**arguments)
                    if spatial is not None and name in ('robodojo_start', 'robodojo_execute') and 'current_eef' in packet:
                        # A reporting failure after an acknowledged action must never be
                        # reported as a safe-to-retry action rejection.
                        try:
                            packet = dict(packet, spatial_status=spatial.summary(packet))
                        except Exception as error:
                            packet = dict(packet, spatial_status=dict(usable=False,
                                error=str(error), no_action_retry=True))
                except InputError as error:
                    errors += 1
                    packet = rejected_input(error, rollout, arguments, errors)
                    success = False
                else:
                    success = True
                    # Reset only network wakeups; rejected-call counts remain audit-only.
                    network_consecutive = 0
                    network_error = None
                if success:
                    packet = add_candidate_images(rollout, packet, tool=name)
                # Persist every rejection, including the fifth and later, before
                # replying. A failed transport reply must not repeat a handler.
                environment_state = self._environment_state(rollout, tool=name, packet=packet)
                if environment_state is not None:
                    # Keep the compact host state first even in large packets.
                    packet = dict(environment_state=environment_state,
                                  **{k: v for k, v in packet.items() if k != 'environment_state'})
                # After a tool has completed, backfill all autonomous execution
                # images into the same GPT thread. Failure here is not InputError:
                # already acknowledged actions must never be repeated.
                history = getattr(self, 'episode_history', None)
                history_cursor = None
                if success and history is not None:
                    packet, history_cursor = history.prepare(rollout, packet)
                    if getattr(rollout, 'obs', None) is not None:
                        packet = dict(packet, grasp_feedback=grasp_teacher_packet(rollout.obs))
                write_json(self.workspace/f'call_{call_index:04d}_result.json', packet)
                if getattr(self, 'dialogue_path', None):
                    append_dialogue(self.dialogue_path, dict(role='assistant_tool_call',
                        step_id=rollout.tick, **audit))
                    append_dialogue(self.dialogue_path, dict(role='tool_result', tool=name,
                        step_id=rollout.tick, success=success, content=packet))
                self.transport.reply(event['id'], dict(success=success,
                    contentItems=content_items(packet, images=success and
                        (name != 'pi05_infer' or bool(getattr(rollout, 'service_handlers', {}))
                         or bool(packet.get('candidate_trajectory_images'))
                         or bool(packet.get('gpt_episode_history', {}).get('new_frames'))),
                                               max_image_edge=self.image_max_edge)))
                if history_cursor is not None:
                    history.commit(history_cursor)
                call_index += 1
                print(json.dumps(dict(event='tool_done', tool=name, step_id=rollout.tick,
                    phase=rollout.phase, counters=rollout.counters)), flush=True)
                deadline = time.monotonic() + self.timeout
                capacity_retry_in_progress = False
                active_grace_windows = 0
            elif method in ('item/started', 'item/completed'):
                if params.get('threadId') != self.thread_id or params.get('turnId') != turn_id:
                    continue
                item = params.get('item', {})
                kind = item.get('type')
                if kind in NATIVE_WORK_ITEMS or (kind == 'agentMessage' and method == 'item/completed'):
                    # Public outputs only: never copy private reasoning items.
                    with (self.workspace/'agent_events.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(method=method, **params), ensure_ascii=False)+'\n')
                    event_record = dict(event='agent_activity', phase=method.split('/')[-1],
                        item_type=kind, item_id=item.get('id'), step_id=rollout.tick,
                        status=item.get('status'))
                    if kind == 'agentMessage':
                        event_record['text'] = item.get('text', '')
                    elif kind == 'commandExecution':
                        event_record['command'] = item.get('command')
                        event_record['exit_code'] = item.get('exitCode')
                    if method == 'item/completed' and getattr(self, 'dialogue_path', None):
                        append_dialogue(self.dialogue_path, dict(role='assistant',
                            step_id=rollout.tick, item=item))
                    print(json.dumps(event_record, ensure_ascii=False), flush=True)
                    if kind in NATIVE_WORK_ITEMS:
                        # Real analysis is progress, not a missing rollout decision.
                        deadline = time.monotonic() + self.timeout
                        capacity_retry_in_progress = False
                        active_grace_windows = 0
            elif method == 'turn/completed' and params.get('threadId') == self.thread_id:
                turn = params.get('turn', {})
                if turn.get('id') != turn_id:
                    continue
                write_json(self.workspace/f'turn_{completed_turns:02d}_completed.json', turn)
                completed_turns += 1
                failure = turn.get('error') or network_error
                if account_interrupted_turn == turn_id:
                    if rollout.phase=='done':
                        return
                    account_interrupted_turn=None
                    turn_id=self._turn(f'Continue the same episode from step {rollout.tick}. No acknowledged action should be repeated. Next call: '+json.dumps(rollout.next_call()))
                    deadline=time.monotonic()+self.timeout
                    continue
                if turn.get('status') == 'failed' and is_network_error(failure):
                    if rollout.phase == 'done':
                        return  # Native completion needs no further paid model response.
                    turn_id = self._network_continue(rollout, turn_id, failure,
                        network_consecutive, network_serial)
                    network_serial += 1
                    network_consecutive += 1
                    network_error = None
                    capacity_retry_in_progress = False
                    # A provider retry is not tool progress and must not earn
                    # another unbounded sequence of idle grace windows.
                    deadline = time.monotonic() + self.timeout
                    continue
                if turn.get('status') != 'completed':
                    raise RuntimeError(f'Codex turn ended: {turn.get("status")}')
                if rollout.phase == 'done':
                    return
                # A completed model turn is not a completed robot episode.
                # Bound consecutive stalls, not the number of productive turns.
                progressed = rollout.tick > last_completed_tick
                no_progress_turns = 0 if progressed else no_progress_turns + 1
                record = dict(revision=CONTROLLER_VERSION, thread_id=self.thread_id,
                    previous_turn_id=turn_id, previous_step_id=last_completed_tick,
                    step_id=rollout.tick, physical_progress=progressed,
                    consecutive_no_progress_turns=no_progress_turns,
                    next_call=rollout.next_call(), physical_actions_replayed=0,
                    status='stalled' if no_progress_turns >= 3 else 'continue_requested')
                path = self.workspace/f'progress_continue_{continuations:04d}.json'
                write_json(path, record)
                if no_progress_turns >= 3:
                    raise RuntimeError('Codex ended 3 consecutive turns without physical execution progress')
                last_completed_tick = rollout.tick
                # Context-only continuation, never a human action judgement.
                continuations += 1
                turn_id = self._turn('The same episode is unfinished. Continue the skill; next call: ' +
                                     json.dumps(rollout.next_call()))
                record.update(status='continue_sent', new_turn_id=turn_id)
                write_json(path, record)
                print(json.dumps(dict(event='codex_progress_continue_sent', **record)), flush=True)
                deadline = time.monotonic() + self.timeout
            elif method in ('error', 'turn/failed'):
                if (params.get('threadId') not in (None, self.thread_id)
                        or params.get('turnId') not in (None, turn_id)):
                    continue
                provider_error = params.get('error', params)
                if is_network_error(params.get('error', params)):
                    network_error = params.get('error', params)
                if method == 'error' and params.get('willRetry') is True:
                    capacity_retry_in_progress = is_capacity_error(params.get('error', params))
                    # App-server owns model transport retry; this does not
                    # repeat inference, a tool handler, or physical execution.
                    print(json.dumps(dict(event='codex_transport_retry',
                        error=params.get('error'), step_id=rollout.tick)), flush=True)
                    continue
                if network_error and is_network_error(params.get('error', params)):
                    if rollout.phase == 'done':
                        return
                    # Wait for turn/completed, or verify idle/failed via the timeout probe.
                    # Do not send continue while Codex still owns an active turn.
                    deadline = min(deadline, time.monotonic() + 30)
                    print(json.dumps(dict(event='codex_network_wait_terminal',
                        error=network_error, step_id=rollout.tick)), flush=True)
                    continue
                raise RuntimeError(f'Codex error: {params}')
            elif 'id' in event and 'method' in event:
                raise RuntimeError(f'Unexpected Codex capability request: {method}')

    def close(self):
        self.transport.close()


