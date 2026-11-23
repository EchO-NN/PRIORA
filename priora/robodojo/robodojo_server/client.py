"""File-backed blocking rollout API, using the copied native recorder/EEF contract.

One owner holds the simulator connection. Codex, not this class, schedules every
start -> infer -> execute cycle. No inference, retry or correction runs implicitly.
"""
import hashlib
from pathlib import Path
import time
import uuid

import numpy as np
from PIL import Image

from ..io import InputError, write_json
from ..settings import MODEL, EFFORT, PROVIDER
from .protocol import RPCClient
from .gate_assessment import GATE_INSTRUCTION, previous_observation, validate_assessment
from .validation import angles, validate_response
from .action_edit_kinematics import edited_targets
from .eef_waypoints import validated_waypoints, expanded_waypoints
from .waypoint_guard import verify_report
from .debug_recorder import DebugVideoRecorder
from .decision_log import decision_event, emit
from .proposal_diagnostics import action_diagnostics
from ..rgbd import main_camera_input, depth_artifact, depth_cameras, depth_key, sensor_input_note
from ..pi05_server.gripper_hold import GripperHold
from ..execution_limits import HYBRID_STEP_LIMITS
from ..grasp_feedback import teacher_packet as grasp_teacher_packet
from ..camera_intrinsics import from_observation, key as intrinsic_key, INTRINSICS_NOTE
from ..camera_geometry import (from_observation as geometry_from_observation,
                               OBSERVATION_KEYS as GEOMETRY_KEYS, VERSION as GEOMETRY_VERSION)
from ..dense_geometry import write_dense_geometry
from .active_view import reject_active_view


ARMS = ('left', 'right')


def validate_dual_response(response, request):
    reject_active_view(response)
    validated_waypoints(response, request)
    if response.get('mode') == 'eef':
        return  # All stages and both arms were checked before any execution.
    if response.get('mode') == 'edit':
        field = 'edit'
        if set(response.get(field, {})) != set(ARMS):
            raise ValueError(f'{field} requires explicit left and right arms')
        for arm in ARMS:
            local = dict(response, **{field: response[field][arm]})
            local_request = dict(request, current_eef=request['current_eef'][arm],
                student_eef_trajectory=[p[arm] for p in request['student_eef_trajectory']])
            validate_response(local, local_request)
    else:
        validate_response(response, request)


class RoboDojoTools:
    profile = 'robodojo'
    camera_keys = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
    observation_keys = (*camera_keys, 'states', 'eef_positions', 'eef_quaternions_wxyz')
    validate = staticmethod(validate_dual_response)

    def current_eef(self):
        return {arm: angles(dict(position=self.obs['eef_positions'][i].tolist(),
            quaternion_wxyz=self.obs['eef_quaternions_wxyz'][i].tolist(),
            frame='environment_origin', link='link6',
            gripper_opening_command=float(self.obs['states'][i*7+6])))
            for i, arm in enumerate(ARMS)}

    def trajectory(self):
        return [dict(index=i, **{arm: angles(p[arm]) for arm in ARMS})
                for i, p in enumerate(self.fk['trajectory'])]

    def preview_student_grippers(self, actions):
        return self.gripper_hold.preview(actions)

    def preview_student_trajectory(self):
        return self._rpc('fk_preview', actions=self.actions)

    def student_diagnostics(self):
        return action_diagnostics(self.actions)

    def execute_student(self, steps):
        return self._rpc('chunk_step', actions=self.actions[:steps],
                         student_prediction_id=self.prediction['prediction_id'])

    def correction_targets(self, response):
        if response['mode'] == 'eef':
            return [row['target'] for row in expanded_waypoints(response)]
        targets = {}
        for arm in ARMS:
            trajectory = [p[arm] for p in self.fk['trajectory']]
            targets[arm] = edited_targets(trajectory, response['steps'], response['edit'][arm])
            if response['edit'][arm]['gripper'] == 'keep':
                for target, original in zip(targets[arm], trajectory):
                    target['gripper_opening'] = original['gripper_opening']
        return [{arm: targets[arm][i] for arm in ARMS} for i in range(response['steps'])]

    def target_proposal(self, target):
        return self._rpc('eef_joint_target', targets=target)

    def verify_eef_report(self, report, **kwargs):
        return verify_report(report, **kwargs)

    def gripper_history(self, executed):
        return [{arm: bool(row[i*7+6] < .5) for i, arm in enumerate(ARMS)} for row in executed]

    def __init__(self, output, task, student, *, sim_port=19113, seed=0,
                 max_decisions=180, prompt_sha256='', rpc_factory=RPCClient):
        self.output = Path(output).resolve()
        self.main_camera_input = main_camera_input()
        if depth_cameras():
            self.observation_keys = (*self.observation_keys, *(depth_key(c) for c in depth_cameras()))
        self.task, self.student = task, student
        self.gripper_hold = GripperHold.from_env()
        self.sim_port, self.seed = sim_port, seed
        if max_decisions < 0:
            raise ValueError('max_decisions must be non-negative; 0 uses native termination only')
        self.max_decisions, self.prompt_sha256 = max_decisions, prompt_sha256
        self.rpc_factory = rpc_factory
        self.sim = None
        self.episode = None
        self.require_native_termination = False
        self.tick = 0
        self.phase = 'start'
        self.history = []
        self.preceding = None
        self.counters = dict(student_steps=0, edited_steps=0, recovery_steps=0, predictions=0)
        self.source = 'student'
        self.started = time.monotonic()
        self.recorder = DebugVideoRecorder(self.output)

    def next_call(self):
        index = len(self.history)
        if self.phase == 'start':
            return dict(tool='robodojo_start', task=self.task,
                        output_dir=str(self.output/'observations'/'000'))
        if self.phase == 'infer':
            return dict(tool='pi05_infer', observation_path=str(self.observation_path),
                        output_dir=str(self.output/'proposals'/f'{index:03d}'))
        if self.phase == 'execute':
            return dict(tool='robodojo_execute', proposal_path=str(self.proposal_path),
                        output_dir=str(self.output/'observations'/f'{index+1:03d}'),
                        response=('Supply your gate assessment and student/edit/eef decision; '
                                  'this benchmark waits for native termination'
                                  if self.require_native_termination else
                                  'Supply your gate assessment and student/edit/eef/stop decision'))
        return None

    def _check(self, name, arguments):
        expected = self.next_call()
        if expected is None or name != expected['tool']:
            raise InputError(f'Wrong order or completed rollout; next call: {expected}')
        for key, value in expected.items():
            if key not in ('tool', 'response') and arguments.get(key) != value:
                raise InputError(f'{key} must equal {value!r}')
        destination = Path(arguments['output_dir'])
        if destination.exists() or not destination.resolve().is_relative_to(self.output):
            raise InputError('Output must be a fresh directory under this rollout root')
        return destination

    def _rpc(self, op, **kwargs):
        return self.sim.request(op, episode_id=self.episode, step_id=self.tick, **kwargs)

    def _observation(self, directory, *, result=None):
        directory.mkdir(parents=True)
        self.obs = self._rpc('teacher_observation')
        calibration = from_observation(self.obs, required=self.meta.get('provides_camera_intrinsics', False))
        geometry = geometry_from_observation(self.obs, self.meta.get('camera_calibration'),
            episode_id=self.episode, step_id=self.tick,
            required=self.meta.get('provides_camera_geometry', False))
        if geometry:
            calibration = {camera: dict(value, extrinsics_provided=True,
                extrinsics_field='camera_geometry.cameras.'+camera)
                for camera, value in calibration.items()}
        np.savez_compressed(directory/'observation.npz', **{
            k: self.obs[k] for k in (*self.observation_keys, *(intrinsic_key(c) for c in calibration),
                                   *(GEOMETRY_KEYS if geometry else ()))})
        dense = (write_dense_geometry(directory, self.obs, calibration, geometry)
                 if geometry and all(depth_key(c) in self.obs for c in self.camera_keys) else {})
        images = []
        for camera in self.camera_keys:
            path = directory/f'{camera}.png'
            Image.fromarray(self.obs[camera]).save(path)
            images.append(dict(camera=camera, path=str(path)))
            if camera in depth_cameras():
                images.append(depth_artifact(self.obs[depth_key(camera)], directory, camera))
        if calibration:
            for item in images:
                camera = item.get('paired_rgb_camera', item['camera'])
                item['camera_intrinsics'] = calibration[camera]
                if geometry:
                    item['camera_geometry'] = geometry['cameras'][camera]
                if dense:
                    item['dense_geometry'] = dense[camera]
        self.observation_path = directory/'observation.json'
        packet = dict(episode_id=self.episode, step_id=self.tick, task=self.task,
            max_episode_steps=self.meta['max_episode_steps'],
            instruction=self.obs['instruction'], remaining_steps=self.obs['remaining_steps'],
            require_native_termination=self.require_native_termination,
            current_state=self.obs['states'].tolist(), images=images,
            current_eef=self.current_eef(), robot_profile=self.profile,
            observation_path=str(self.observation_path), arrays_path=str(directory/'observation.npz'),
            history_path=str(self.output/'history.json'),
            previous_result=self.history[-1] if self.history else None,
            counters=dict(self.counters), result=result, rollout_finished=self.phase == 'done',
            next_call=self.next_call())
        self.observation_packet = packet
        packet['grasp_feedback'] = grasp_teacher_packet(self.obs)
        if 'grasp_feedback_json' in self.obs:
            feedback_path = directory/'grasp_feedback.json'
            feedback_path.write_text(str(self.obs['grasp_feedback_json']))
            packet['grasp_feedback']['raw_path'] = str(feedback_path)
        if calibration:
            packet['camera_intrinsics'] = calibration
            packet['camera_intrinsics_note'] = INTRINSICS_NOTE
        if geometry:
            packet['camera_geometry'] = geometry
        if dense:
            packet['dense_geometry'] = dense
        if depth_cameras():
            packet['sensor_input_note'] = sensor_input_note()
        if self.tick == 0:
            from ..prompt_context import task_context
            packet['task_context'] = task_context(self.task)
        write_json(self.observation_path, packet)
        return packet

    def start(self, **arguments):
        directory = self._check('robodojo_start', arguments)
        self.sim = self.rpc_factory('127.0.0.1', self.sim_port, timeout=600)
        self.meta = self.sim.request('metadata')
        if depth_cameras() and self.meta.get('main_camera_input') != self.main_camera_input:
            raise RuntimeError('Teacher requests RGBD but simulator is not capturing depth')
        if self.meta['task'] != self.task:
            raise RuntimeError('Launched RoboDojo task differs from requested task')
        identity = self.student.metadata['checkpoint_sha256']
        config = self.student.metadata['config']
        backend = self.student.metadata.get('backend', 'OpenPI/JAX')
        version = f'{backend}/{config}/{identity[:16]}'
        reset = self.sim.request('reset', seed=self.seed, source='student', policy_version=version)
        # Fixed camera configuration is finalized at reset, after the initial
        # metadata handshake. Use that episode's declared calibration.
        for key in ('camera_calibration', 'camera_geometry_version', 'provides_camera_geometry'):
            if key in reset.get('metadata', {}):
                self.meta[key] = reset['metadata'][key]
        if reset.get('instruction'):
            self.meta['instruction'] = reset['instruction']
        self.episode, self.tick = reset['episode_id'], reset['step_id']
        # A frozen benchmark case must use its native horizon, not a model's
        # judgement that further attempts are unlikely to help. Operator and
        # infrastructure aborts still use finish() and remain incomplete.
        self.require_native_termination = bool(reset.get('metadata', {}).get('evaluation_case'))
        from ..prompt_context import CONTEXT_VERSION
        self._rpc('begin_combination', teacher_model=MODEL, teacher_model_provider=PROVIDER,
                  context_version=CONTEXT_VERSION,
                  prompt_sha256=self.prompt_sha256,
                  student_policy_version=version, student_policy_sha256=identity)
        self.run = dict(schema='robodojo_rollout.run.v1', teacher='codex_tools', evaluation_method='wall_plus_gpt',
            context_version=CONTEXT_VERSION,
            teacher_model=MODEL, teacher_model_provider=PROVIDER,
            teacher_reasoning_effort=EFFORT,
            task=self.task, instruction=self.meta['instruction'], seed=self.seed,
            evaluation_case=reset.get('metadata', {}).get('evaluation_case'),
            config=config, robot_profile=self.profile, checkpoint=self.student.metadata['checkpoint'],
            action_horizon=self.student.metadata.get('action_horizon', 50),
            execution_step_limits=dict(HYBRID_STEP_LIMITS),
            eef_waypoint_protocol='eef_waypoints_v1',
            eef_reachability_protocol='eef_reachability_v1',
            camera_geometry_version=self.meta.get('camera_geometry_version'),
            camera_calibration_rig_id=self.meta.get('camera_calibration', {}).get('rig_id'),
            action_dim=self.student.metadata.get('action_dim', 8),
            student_backend=backend, student_identity_sha256=identity,
            student_server_metadata=self.student.metadata, teacher_off=False, split=None,
            gate_policy='failure-or-intent', no_rollback=True,
            baseline_gate_instruction_sha256=hashlib.sha256(GATE_INSTRUCTION.encode()).hexdigest(),
            gate_instruction_sha256=hashlib.sha256(GATE_INSTRUCTION.encode()).hexdigest(),
            teacher_prompt_sha256=self.prompt_sha256, initial_state_hash=reset.get('initial_state_hash'),
            reset_metadata=reset.get('metadata'), max_episode_steps=self.meta['max_episode_steps'],
            control_dt=self.meta['control_dt'], max_decisions=self.max_decisions,
            require_native_termination=self.require_native_termination,
            training_steps=0, training_bundle_ready=False)
        self.run['pi05_gripper_policy'] = self.gripper_hold.metadata()
        write_json(self.output/'run.json', self.run)
        write_json(self.output/'history.json', self.history)
        self.phase = 'infer'
        return self._observation(directory)

    def infer(self, **arguments):
        directory = self._check('pi05_infer', arguments)
        return self._infer_proposal(directory)

    def _policy_observation(self):
        """Policy input may be scoped by a supervisor without altering native goals."""
        return self.obs

    def _infer_proposal(self, directory):
        """Generate a fresh proposal, also used when a supervisor changes its goal."""
        directory.mkdir(parents=True)
        decision = len(self.history)
        self.proposal_path = directory/'actions.npz'
        self.actions, self.prediction = self.student.infer(self._policy_observation(), self.proposal_path)
        self.actions, self.gripper_preview_states = self.preview_student_grippers(self.actions)
        if self.gripper_hold.mode == 'hold_closed':
            with np.load(self.proposal_path) as original:
                arrays = {key: original[key] for key in original.files}
            arrays['pi05_actions'] = arrays['actions'].copy()
            arrays['actions'] = self.actions
            arrays['gripper_held_after_each_step'] = self.gripper_preview_states
            np.savez_compressed(self.proposal_path, **arrays)
        # Preserve the colleague's downstream audit file layout without copying arrays.
        (self.output/f'proposal_{decision:03d}.npz').hardlink_to(self.proposal_path)
        self.counters['predictions'] += 1
        self.fk = self.preview_student_trajectory()
        self.request = dict(request_id=uuid.uuid4().hex, episode_id=self.episode,
            step_id=self.tick, decision=decision, gate_policy='failure-or-intent',
            execution_step_limits=dict(HYBRID_STEP_LIMITS),
            eef_waypoint_protocol='eef_waypoints_v1',
            eef_reachability_protocol='eef_reachability_v1',
            gate_instruction=GATE_INSTRUCTION, task=self.task, instruction=self.obs['instruction'],
            sim_time_s=self.tick*self.meta['control_dt'],
            max_episode_steps=self.meta['max_episode_steps'],
            remaining_steps=self.obs['remaining_steps'],
            require_native_termination=self.require_native_termination,
            images=self.observation_packet['images'],
            current_state=self.observation_packet['current_state'],
            current_eef=self.observation_packet['current_eef'],
            student_eef_trajectory=self.trajectory(), robot_profile=self.profile,
            action_diagnostics=self.student_diagnostics(),
            pi05_gripper_policy=self.gripper_hold.metadata(),
            fk_check=self.fk['measured_fk_check'],
            previous_observation=previous_observation(self.preceding, self.history[-1]) if self.history else None,
            previous_result=self.history[-1] if self.history else None,
            counters=dict(self.counters), **self.prediction)
        if depth_cameras():
            self.request['sensor_input_note'] = sensor_input_note()
        if self.observation_packet.get('camera_intrinsics'):
            self.request['camera_intrinsics'] = self.observation_packet['camera_intrinsics']
            self.request['camera_intrinsics_note'] = INTRINSICS_NOTE
        if self.observation_packet.get('camera_geometry'):
            self.request['camera_geometry'] = self.observation_packet['camera_geometry']
        if self.observation_packet.get('dense_geometry'):
            self.request['dense_geometry'] = self.observation_packet['dense_geometry']
        write_json(self.output/f'request_{decision:03d}.json', self.request)
        self.phase = 'execute'
        # The correction gate is in the persistent workspace.
        packet = {k: v for k, v in self.request.items() if k != 'gate_instruction'}
        packet.update(proposal_path=str(self.proposal_path), rollout_finished=False,
                      request_path=str(self.output/f'request_{decision:03d}.json'), next_call=self.next_call())
        write_json(directory/'proposal.json', packet)
        return packet

    def execute(self, **arguments):
        directory = self._check('robodojo_execute', arguments)
        response = arguments.get('response')
        try:
            self.validate(response, self.request)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise InputError(str(error)) from error
        if response['mode'] == 'stop' and self.require_native_termination:
            raise InputError(
                'This benchmark requires native termination; model stop is not allowed. '
                f"There are {self.obs['remaining_steps']} native control steps remaining. "
                'No action or stop was executed. Reassess the current observation and '
                'fresh proposal, then submit a student/edit/eef decision under the declared '
                'gate. Failed grasp attempts alone are not an episode termination. '
                'Do not reset or fabricate progress; operator/transport/physics aborts '
                'are handled separately by the runner.')
        decision = len(self.history)
        reachability = preflight_path = None
        nodes = []
        if response['mode'] == 'eef':
            nodes = validated_waypoints(response, self.request)
            reachability = self._rpc('check_eef_waypoints', waypoints=nodes)
            preflight_path = self.output/f'eef_preflight_{decision:03d}_{uuid.uuid4().hex[:12]}.json'
            write_json(preflight_path, dict(request_id=self.request['request_id'],
                step_id=self.tick, decision=decision, response=response,
                report=reachability, physical_steps=0))
            reachability = self.verify_eef_report(reachability,
                nodes=nodes, expected_remaining_steps=self.obs['remaining_steps'])
            if not reachability['accepted']:
                failed = [f"{row['index']} ({row['label']})" for row in reachability['waypoints']
                          if not row['passed']]
                error = InputError('EEF waypoint reachability not verified: '+
                    str(reachability.get('reason', 'step-budget or IK check failed'))+
                    ('; failed stages: '+', '.join(failed) if failed else '')+
                    '. No actions executed. Revise targets, orientation or stage step allocation '
                    'and retry this same proposal; do not reset.')
                error.waypoint_reachability = reachability
                error.waypoint_preflight_path = str(preflight_path)
                raise error
        begin_execution = getattr(self, '_begin_reviewed_execution', None)
        if begin_execution is not None:
            begin_execution(response)
        write_json(self.output/f'response_{decision:03d}.json', response)
        # RoboDojo persists every decision (including student/stop), not just
        # source transitions. A failed receipt aborts before physical execution.
        receipt = self._rpc('record_codex_decision', decision=decision,
                            prediction_id=self.prediction['prediction_id'], response=response)
        emit(dict(decision_event(episode_id=self.episode, step_id=self.tick,
            decision=decision, prediction_id=self.prediction['prediction_id'], response=response),
            simulator_receipt=receipt))
        if response['mode'] == 'stop':
            return self._observation(directory, result=self.finish('model_stop'))
        trigger = validate_assessment(response, self.request)
        wanted = 'student' if response['mode'] == 'student' else 'gpt_eef'
        if wanted != self.source:
            self._rpc('switch_control_source', source=wanted, reason=response['reason'])
            self.source = wanted
        trace = []
        arrival_checks = {}
        interruption = None
        infrastructure_fault = None
        motion_guard = None
        start_tick = self.tick
        if response['mode'] == 'student':
            result = self.execute_student(response['steps'])
            trace = result['steps']
            self.tick = result['step_id']
            infrastructure_fault = result.get('infrastructure_fault')
            motion_guard = result.get('motion_guard')
        else:
            targets = self.correction_targets(response)
            schedule = expanded_waypoints(response) if response['mode'] == 'eef' else []
            for control_index, target in enumerate(targets):
                proposal = self.target_proposal(target)
                result = self._rpc('chunk_step', actions=np.asarray([proposal['action']], np.float32),
                                   teacher_request_id=self.request['request_id'])
                for row in result['steps']:
                    row['edited_target'], row['ik_diagnostics'] = target, proposal['diagnostics']
                    if schedule:
                        row.update({key: value for key, value in schedule[control_index].items()
                                    if key != 'target'})
                trace.extend(result['steps'])
                self.tick = result['step_id']
                infrastructure_fault = result.get('infrastructure_fault')
                motion_guard = result.get('motion_guard')
                if motion_guard and motion_guard.get('blocked'):
                    interruption = dict(reason='joint_motion_guard', report=motion_guard)
                    break
                if infrastructure_fault:
                    break
                if any(r['terminated'] or r['truncated'] for r in result['steps']):
                    break
                if schedule:
                    stage = schedule[control_index]
                    last_in_stage = (control_index+1 == len(schedule) or
                        schedule[control_index+1]['waypoint_index'] != stage['waypoint_index'])
                    if last_in_stage:
                        check = self.verify_eef_report(self._rpc('check_eef_targets', targets=target))
                        arrival_checks[stage['waypoint_index']] = check
                        if not check['passed']:
                            interruption = dict(reason='waypoint_not_reached',
                                index=stage['waypoint_index'], label=stage['waypoint_label'], check=check)
                            break  # Already executed: return partial ACKs and a fresh observation.
        valid = [r for r in trace if r['valid']]
        self.gripper_hold.commit(self.gripper_preview_states, len(valid), response)
        if response['mode'] != 'student':
            write_json(self.output/f'edit_{decision:03d}.json', [dict(
                target=r['edited_target'], diagnostics=r['ik_diagnostics'], valid=r['valid'],
                executed_action=np.asarray(r['executed_action']).tolist(),
                **{key: r[key] for key in ('waypoint_index', 'waypoint_label', 'waypoint_step')
                   if key in r}) for r in trace])
        countkey = dict(student='student_steps', edit='edited_steps', eef='recovery_steps')[response['mode']]
        self.counters[countkey] += len(valid)
        executed = np.asarray([r['executed_action'] for r in valid], np.float32)
        np.savez_compressed(self.output/f'execution_{decision:03d}.npz', actions=executed,
                            states=np.asarray([r['obs']['states'] for r in valid], np.float32))
        done = bool(valid and (valid[-1]['terminated'] or valid[-1]['truncated']))
        record = dict(decision=decision, start_tick=start_tick, end_tick=self.tick, response=response,
            simulator_decision_receipt=receipt,
            executed_steps=len(valid), source=self.source, source_detail=response['mode'],
            prediction_id=self.prediction['prediction_id'],
            selected_student_steps=response['steps'] if response['mode'] == 'student' else 0,
            discarded_student_steps=len(self.actions)-len(valid) if response['mode'] == 'student' else len(self.actions),
            executed_gripper_closed=self.gripper_history(executed),
            pi05_gripper_policy=self.gripper_hold.metadata(),
            assessment=response['assessment'], takeover_trigger=trigger, terminal=done,
            native_success=bool(valid and valid[-1]['success']))
        if infrastructure_fault:
            record['infrastructure_fault'] = infrastructure_fault
        if motion_guard is not None:
            record['motion_guard'] = motion_guard
        if response['mode'] == 'eef':
            record['eef_waypoint_protocol'] = 'eef_waypoints_v1'
            record['waypoint_reachability'] = reachability
            record['waypoint_preflight_path'] = str(preflight_path)
            record['waypoint_interruption'] = interruption
            record['executed_waypoints'] = []
            cursor = start_tick
            for index, node in enumerate(nodes):
                count = sum(row.get('waypoint_index') == index for row in valid)
                check = arrival_checks.get(index)
                record['executed_waypoints'].append(dict(
                    index=index, label=node['label'], requested_steps=node['steps'],
                    executed_steps=count, start_tick=cursor if count else None,
                    end_tick=cursor+count if count else None, complete=count == node['steps'],
                    arrival_check=check, arrived=check['passed'] if check is not None else None))
                cursor += count
        self.history.append(record)
        self.preceding = self.request
        write_json(self.output/'history.json', self.history)
        write_json(self.output/'progress.json', dict(episode_id=self.episode, step_id=self.tick, **self.counters))
        complete_execution = getattr(self, '_complete_reviewed_execution', None)
        if complete_execution is not None:
            complete_execution(record)
        self.phase = 'infer'
        budget_exhausted = self.max_decisions > 0 and len(self.history) >= self.max_decisions
        final = (self.finish('infrastructure_fault' if infrastructure_fault else
                            'terminal' if done else 'decision_budget')
                 if infrastructure_fault or done or budget_exhausted else None)
        if infrastructure_fault:
            # Cached pre-fault images may be older than the acknowledged step.
            # Preserve counters/history without relabelling them as current RGBD.
            directory.mkdir(parents=True)
            packet = dict(episode_id=self.episode, step_id=self.tick,
                rollout_finished=True, result=final, infrastructure_fault=infrastructure_fault,
                observation_available=False, images=[], next_call=None,
                counters=dict(self.counters), history_path=str(self.output/'history.json'))
            write_json(directory/'observation_unavailable.json', packet)
            return packet
        return self._observation(directory, result=final)

    def finish(self, reason):
        final = self._rpc('finish_pilot', reason=reason)
        self.phase = 'done'
        from ..prompt_context import CONTEXT_VERSION
        final.update(self.counters, decisions=len(self.history), context_version=CONTEXT_VERSION,
                     complete=bool(final['terminated'] or final['truncated']),
                     wall_seconds=time.monotonic()-self.started, training_bundle_ready=False,
                     trajectory_path=str(self.output.parent/'sim'),
                     history_path=str(self.output/'history.json'),
                     debug_video_path=str(self.recorder.output/'debug_rollout.mp4'),
                     debug_video_manifest=str(self.recorder.output/'manifest.json'))
        write_json(self.output/'result.json', final)
        return final

    def close(self):
        try:
            if self.sim is not None:
                self.sim.close()
        finally:
            # Teardown of video evidence is independent of transport teardown.
            if self.episode is not None:
                self.recorder.finalize()
