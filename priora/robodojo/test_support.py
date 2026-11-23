import numpy as np
from .robodojo_server.client import RoboDojoTools
import copy

POSE = dict(position=[0., 0., 0.], quaternion_wxyz=[1., 0., 0., 0.], gripper_closed=False)


def response(request, mode='student'):
    return dict(request_id=request['request_id'], mode=mode, steps=2, reason='Synthetic contract test.',
        target={arm: copy.deepcopy(POSE) for arm in ('left', 'right')},
        edit={arm: dict(delta_position=[0, 0, 0], delta_rotation_vector=[0, 0, 0], gripper='keep')
              for arm in ('left', 'right')},
        assessment=dict(task_progress=dict(verified_completed=[], currently_attempting='Arrange digits', remaining=['Arrange']),
            current_subgoal='Arrange digits', execution_status='not_started', execution_evidence='Synthetic image.',
            expected_next_intent='Arrange', predicted_next_intent='Arrange',
            intent_status='aligned' if mode == 'student' else 'misaligned', intent_evidence='Synthetic FK.'))


class Student:
    metadata = dict(checkpoint='/test/checkpoint', checkpoint_sha256='a'*64,
        config="synthetic", action_horizon=50, action_dim=14)

    def infer(self, obs, output):
        actions = np.zeros((50, 14), np.float32)
        actions[:, [6, 13]] = .3
        np.savez_compressed(output, actions=actions, raw_actions=actions)
        return actions, dict(prediction_id='prediction', inference_index=0, inference_seconds=0)


class Sim:
    def __init__(self, *args, **kwargs):
        self.tick, self.calls = 0, []
        self.measured = {arm: copy.deepcopy(POSE) for arm in ('left', 'right')}
        self.pending_targets = None

    def _mock_limits(self):
        return dict(position_tolerance_m=.002, rotation_tolerance_rad=.02,
            joint_step_limit_rad=.05, joint_numerical_tolerance_rad=1e-6,
            max_total_steps=30)

    def _mock_move(self, start, targets, steps, *, actual=False):
        """Ideal Cartesian test robot; no real IK or simulator proof is claimed."""
        current = copy.deepcopy(start)
        speed = getattr(self, 'actual_translation_per_step' if actual else 'predicted_translation_per_step', .05)
        for _ in range(steps):
            for arm in ('left', 'right'):
                delta = np.asarray(targets[arm]['position']) - current[arm]['position']
                norm = np.linalg.norm(delta)
                current[arm]['position'] = (np.asarray(current[arm]['position']) +
                    delta*min(1., speed/norm if norm else 1.)).tolist()
                current[arm]['quaternion_wxyz'] = copy.deepcopy(targets[arm]['quaternion_wxyz'])
                current[arm]['gripper_closed'] = targets[arm]['gripper_closed']
        return current

    def _mock_target_check(self, measured, targets, *, joints=False):
        arms = {}
        for arm in ('left', 'right'):
            position_error = float(np.linalg.norm(np.asarray(measured[arm]['position']) - targets[arm]['position']))
            dot = abs(float(np.dot(measured[arm]['quaternion_wxyz'], targets[arm]['quaternion_wxyz'])))
            rotation_error = float(2*np.arccos(np.clip(dot, -1., 1.)))
            arms[arm] = dict(position_error_m=position_error, rotation_error_rad=rotation_error,
                within_tolerance=position_error <= .002 and rotation_error <= .02)
            if joints:
                arms[arm].update(joint_limits_ok=True, joint_step_limits_ok=True)
        return dict(schema='eef_target_check_v1', passed=all(a['within_tolerance'] for a in arms.values()),
                    physical_steps=0, limits=self._mock_limits(), arms=arms)

    def _ack_action(self, action):
        if self.pending_targets is not None:
            self.measured = self._mock_move(self.measured, self.pending_targets, 1, actual=True)

    def _mock_preflight(self, nodes):
        cursor = copy.deepcopy(self.measured)
        diagnostics = []
        for index, node in enumerate(nodes):
            cursor = self._mock_move(cursor, node['target'], node['steps'])
            check = self._mock_target_check(cursor, node['target'], joints=True)
            diagnostics.append(dict(index=index, label=node['label'], requested_steps=node['steps'],
                arms=check['arms'], passed=check['passed']))
        within_budget = sum(node['steps'] for node in nodes) <= getattr(self, 'limit', 2)-self.tick
        accepted = within_budget and all(row['passed'] for row in diagnostics)
        return dict(schema='eef_reachability_v1', accepted=accepted, physical_steps=0,
            limits=self._mock_limits(), waypoints=diagnostics,
            total_steps=sum(node['steps'] for node in nodes),
            remaining_steps=getattr(self, 'limit', 2)-self.tick,
            remaining_steps_sufficient=within_budget,
            reason=('accepted' if accepted else ('insufficient_remaining_native_steps'
                    if not within_budget else 'waypoint_not_reachable_in_requested_steps')),
            collision_checked=False, physical_tracking_guaranteed=False)

    def request(self, op, **args):
        self.calls.append(op)
        if op == 'metadata':
            return dict(task='arrange_largest_number', instruction='Arrange the digits.',
                        max_episode_steps=2, control_dt=.04)
        if op == 'reset':
            return dict(episode_id='episode', step_id=0)
        if op == 'teacher_observation':
            return dict(instruction='Arrange the digits.', remaining_steps=2-self.tick,
                states=np.zeros(14, np.float32),
                eef_positions=np.array([self.measured[arm]['position'] for arm in ('left', 'right')]),
                eef_quaternions_wxyz=np.array([self.measured[arm]['quaternion_wxyz'] for arm in ('left', 'right')]),
                **{k: np.zeros((8, 8, 3), np.uint8) for k in RoboDojoTools.camera_keys})
        if op == 'check_eef_waypoints':
            return self._mock_preflight(args['waypoints'])
        if op == 'check_eef_targets':
            return self._mock_target_check(self.measured, args['targets'])
        if op == 'fk_preview':
            return dict(trajectory=[{arm: dict(POSE, gripper_opening=.3) for arm in ('left', 'right')}
                                    for _ in range(50)], measured_fk_check=dict(passed=True))
        if op == 'eef_joint_target':
            self.pending_targets = copy.deepcopy(args['targets'])
            action = np.zeros(14)
            action[[6, 13]] = [not self.pending_targets[arm]['gripper_closed'] for arm in ('left', 'right')]
            return dict(action=action, diagnostics=dict(valid=True))
        if op == 'record_codex_decision':
            return dict(physical_steps=0, record_path='/test/decision.json')
        if op == 'chunk_step':
            rows = []
            for action in args['actions']:
                self.tick += 1
                self._ack_action(action)
                rows.append(dict(valid=True, executed_action=action, obs=dict(states=np.zeros(14)),
                    terminated=False, truncated=self.tick >= 2, success=False))
            return dict(step_id=self.tick, steps=rows)
        if op == 'finish_pilot':
            return dict(step_id=self.tick, success=False, terminated=False, truncated=self.tick >= 2)
        return {}


def invoke(tools, **extra):
    args = tools.next_call().copy()
    name = args.pop('tool'); args.update(extra)
    return dict(robodojo_start=tools.start, pi05_infer=tools.infer, robodojo_execute=tools.execute)[name](**args)


