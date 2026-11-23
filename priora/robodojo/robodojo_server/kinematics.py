"""Robot-only URDF FK and bounded local DLS; no planning or scene truth."""
import xml.etree.ElementTree as ET
from functools import cached_property
import numpy as np
from scipy.spatial.transform import Rotation
from ..robodojo_server.action_edit_kinematics import pose, transform
from . import reachability
from ..execution_limits import EEF_MAX_STEPS
from .fingertips import load_fingertip_geometry


class ArmFK:
    def __init__(self, urdf, joint_names, base='base_link', tip='link6'):
        self.names = list(joint_names)
        edges = {j.find('child').get('link'): j for j in ET.parse(urdf).getroot().findall('joint')}
        self.chain = []
        while tip != base:
            j = edges[tip]
            origin = j.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            t = np.eye(4)
            t[:3, :3], t[:3, 3] = Rotation.from_euler('xyz', rpy).as_matrix(), xyz
            axis = j.find('axis')
            axis = np.fromstring(axis.get('xyz'), sep=' ') if axis is not None else np.array([1., 0, 0])
            self.chain.insert(0, (j.get('name'), j.get('type'), t, axis))
            tip = j.find('parent').get('link')
        moving = [n for n, kind, _, _ in self.chain if kind != 'fixed']
        if set(moving) != set(self.names):
            raise ValueError(f'URDF chain/arm joint mismatch: {moving}, {self.names}')

    def matrix(self, q):
        q = np.asarray(q, float)
        if q.shape != (len(self.names),) or not np.isfinite(q).all():
            raise ValueError('Invalid arm joint vector')
        values, t = dict(zip(self.names, q)), np.eye(4)
        for name, kind, origin, axis in self.chain:
            motion = np.eye(4)
            if kind in ('revolute', 'continuous'):
                motion[:3, :3] = Rotation.from_rotvec(axis*values[name]).as_matrix()
            elif kind != 'fixed':
                raise ValueError('Only fixed/revolute arm chain supported')
            t = t @ origin @ motion
        return t

    def bounded_target(self, q, limits, root, target):
        q = np.asarray(q, float)
        now = root @ self.matrix(q)
        goal = transform(target['position'], target['quaternion_wxyz'])
        translation = goal[:3, 3] - now[:3, 3]
        rotation = Rotation.from_matrix(goal[:3, :3] @ now[:3, :3].T).as_rotvec()
        bounded = lambda x, cap: x*min(1., cap/max(np.linalg.norm(x), 1e-12))
        error = np.r_[bounded(translation, .02), bounded(rotation, .1)]
        jacobian = np.zeros((6, len(q)))
        for i in range(len(q)):
            shifted = q.copy(); shifted[i] += 1e-5
            nxt = root @ self.matrix(shifted)
            jacobian[:3, i] = (nxt[:3, 3]-now[:3, 3])/1e-5
            jacobian[3:, i] = Rotation.from_matrix(nxt[:3, :3] @ now[:3, :3].T).as_rotvec()/1e-5
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .05**2*np.eye(6), error)
        limits = np.asarray(limits, float)
        low, high = np.maximum(limits[:, 0], q-.05), np.minimum(limits[:, 1], q+.05)
        if np.any(low > high):
            raise ValueError('Joint outside bounded valid interval')
        result = np.clip(q+delta, low, high).astype(np.float32)
        return result, dict(target_position_error_m=float(np.linalg.norm(translation)),
            target_rotation_error_rad=float(np.linalg.norm(rotation)),
            proposed_joint_delta=(result-q).tolist(), bounded_task_delta=error.tolist(),
            method='robot_only_numerical_jacobian_dls', physical_tracking_verified=False)


class DualKinematics:
    def __init__(self, env):
        self.env, self.manager = env, env.robot_manager
        self.robots = {r.arm_name.split('_')[0]: r for r in self.manager.robot_list if r.type == 'target'}
        if set(self.robots) != {'left', 'right'}:
            raise ValueError('Expected left/right target X5 arms')
        self.fk = {arm: ArmFK(r.urdf_path, r.arm_joints_name, r.base_link, r.ee_link_name)
                   for arm, r in self.robots.items()}

    @cached_property
    def fingertip_geometry(self):
        return {arm: load_fingertip_geometry(robot) for arm, robot in self.robots.items()}

    def root(self, arm):
        robot = self.robots[arm]
        value = self.manager.get_link_pose(robot, robot.base_link, is_relative=True)[0]
        return transform(value[:3], value[3:])

    def check(self):
        checks = {}
        for arm, r in self.robots.items():
            q = self.manager.get_joint(r)[0]
            predicted = self.root(arm) @ self.fk[arm].matrix(q)
            measured = self.manager.get_real_endpose(r)[0]
            error = np.linalg.norm(predicted[:3, 3]-measured[:3])
            angle = Rotation.from_matrix(predicted[:3, :3] @ transform(measured[:3], measured[3:])[:3, :3].T).magnitude()
            checks[arm] = dict(position_error_m=float(error), rotation_error_rad=float(angle),
                passed=bool(error < .002 and angle < .01))
        if not all(c['passed'] for c in checks.values()):
            raise ValueError(f'Robot FK does not match measured link6: {checks}')
        return dict(arms=checks, passed=True, physical_steps=0)

    def preview(self, actions):
        a = np.asarray(actions)
        if a.shape != (50, 14) or not np.isfinite(a).all():
            raise ValueError('Expected H50 x 14 proposal')
        checks = self.check()
        roots = {arm: self.root(arm) for arm in ('left', 'right')}
        trajectory = [dict(index=i, **{arm: dict(
            **pose(roots[arm] @ self.fk[arm].matrix(row[offset:offset+6])),
            gripper_closed=bool(row[offset+6] < .5), gripper_opening=float(row[offset+6]))
            for arm, offset in (('left', 0), ('right', 7))}) for i, row in enumerate(a)]
        return dict(trajectory=trajectory, measured_fk_check=checks, physical_steps=0,
            frame='environment_origin', interpretation='kinematic_targets_not_object_future',
            fingertip_geometry=self.fingertip_geometry)

    def check_targets(self, targets):
        """Read measured link6 poses after an ACK; never substitute predicted FK."""
        targets = reachability.checked_targets(targets)
        arms = {}
        for arm in ('left', 'right'):
            measured = np.asarray(self.manager.get_real_endpose(self.robots[arm])[0])
            if measured.shape != (7,):
                raise ValueError('Measured EEF pose must contain position and wxyz')
            arms[arm] = reachability.pose_errors(measured[:3], measured[3:], targets[arm])
        passed = all(value['within_tolerance'] for value in arms.values())
        return dict(schema='eef_target_check_v1', passed=passed, arms=arms,
                    limits=reachability.limits(), physical_steps=0,
                    measurement='get_real_endpose', collision_checked=False,
                    physical_tracking_guaranteed=False,
                    reason='measured_pose_within_tolerance' if passed else 'measured_pose_outside_tolerance')

    def check_waypoints(self, waypoints, remaining_steps=None):
        """Predict the exact bounded-DLS sequence without stepping the simulator.

        The prediction assumes each joint target is reached before the next
        update. Failure means not verified by this local solver in the supplied
        step budget; it is not a proof of global mathematical unreachability.
        """
        if not isinstance(waypoints, list) or not 1 <= len(waypoints) <= EEF_MAX_STEPS:
            raise ValueError('A nonempty bounded EEF waypoint list is required')
        if remaining_steps is not None and (type(remaining_steps) is not int or remaining_steps < 0):
            raise ValueError('remaining_steps must be a nonnegative integer or None')
        nodes = []
        for node in waypoints:
            if not isinstance(node, dict) or set(node) != {'label', 'steps', 'target'}:
                raise ValueError('Each waypoint requires label, steps and target')
            if not isinstance(node['label'], str) or not node['label'].strip():
                raise ValueError('Nonempty waypoint label required')
            if type(node['steps']) is not int or not 1 <= node['steps'] <= EEF_MAX_STEPS:
                raise ValueError('Each waypoint needs a positive bounded integer step count')
            nodes.append(dict(label=node['label'], steps=node['steps'],
                              target=reachability.checked_targets(node['target'])))
        total = sum(node['steps'] for node in nodes)
        if total > EEF_MAX_STEPS:
            raise ValueError(f'Total EEF waypoint steps must not exceed {EEF_MAX_STEPS}')
        fk_check = self.check()
        states, soft_limits, roots = {}, {}, {}
        tolerance = reachability.JOINT_NUMERICAL_TOLERANCE_RAD
        for arm in ('left', 'right'):
            robot = self.robots[arm]
            states[arm] = np.asarray(self.manager.get_joint(robot)[0], float).copy()
            key = self.manager.robot_key[self.manager.robot_list.index(robot)]
            soft_limits[arm] = np.asarray(
                key.data.soft_joint_pos_limits[0, robot.arm_joint_indices].cpu().numpy(), float).copy()
            q, bounds = states[arm], soft_limits[arm]
            if (q.shape != (len(self.fk[arm].names),) or bounds.shape != (len(q), 2) or
                    not np.isfinite(q).all() or not np.isfinite(bounds).all() or
                    np.any(bounds[:, 0] > bounds[:, 1])):
                raise ValueError('Invalid measured joint state or soft limits')
            if np.any(q < bounds[:, 0]-tolerance) or np.any(q > bounds[:, 1]+tolerance):
                raise ValueError('Measured joints are outside their soft limits')
            roots[arm] = self.root(arm)
        reports = []
        for index, node in enumerate(nodes):
            arms = {}
            for arm in ('left', 'right'):
                q, bounds = states[arm], soft_limits[arm]
                joint_limits_ok = joint_step_limits_ok = True
                elbow_path_ok = True
                max_delta = 0.0
                for _ in range(node['steps']):
                    candidate, _ = self.fk[arm].bounded_target(q, bounds, roots[arm], node['target'][arm])
                    candidate = np.asarray(candidate, float)
                    if candidate.shape != q.shape or not np.isfinite(candidate).all():
                        raise ValueError('Bounded IK produced an invalid joint target')
                    delta = float(np.max(np.abs(candidate-q)))
                    max_delta = max(max_delta, delta)
                    joint_step_limits_ok &= delta <= reachability.JOINT_STEP_LIMIT_RAD+tolerance
                    joint_limits_ok &= bool(np.all(candidate >= bounds[:, 0]-tolerance) and
                                            np.all(candidate <= bounds[:, 1]+tolerance))
                    guard = getattr(self, 'joint_path_guards', {}).get(arm)
                    if guard is not None:
                        elbow_path_ok &= guard.check_path(q, candidate)['passed']
                    q = candidate
                states[arm] = q
                predicted = pose(roots[arm] @ self.fk[arm].matrix(q))
                errors = reachability.pose_errors(predicted['position'], predicted['quaternion_wxyz'],
                                                   node['target'][arm])
                arms[arm] = dict(errors, joint_limits_ok=bool(joint_limits_ok),
                                 joint_step_limits_ok=bool(joint_step_limits_ok),
                                 elbow_path_ok=bool(elbow_path_ok),
                                 maximum_joint_step_rad=max_delta,
                                 predicted_joint_positions=q.tolist(), predicted_pose=predicted)
            passed = all(item['within_tolerance'] and item['joint_limits_ok'] and
                         item['joint_step_limits_ok'] and item['elbow_path_ok'] for item in arms.values())
            reports.append(dict(index=index, label=node['label'], requested_steps=node['steps'],
                                arms=arms, passed=passed))
        budget_ok = remaining_steps is None or total <= remaining_steps
        accepted = budget_ok and all(report['passed'] for report in reports)
        reason = ('insufficient_remaining_native_steps' if not budget_ok else
                  'all_waypoints_verified_within_requested_steps' if accepted else
                  'not_verified_within_requested_steps_or_local_ik')
        return dict(schema='eef_reachability_v1', accepted=bool(accepted), physical_steps=0,
                    limits=reachability.limits(), waypoints=reports, reason=reason,
                    total_steps=total, remaining_steps=remaining_steps, remaining_steps_sufficient=budget_ok,
                    measured_fk_check=fk_check, collision_checked=False,
                    elbow_path_checked=bool(getattr(self, 'joint_path_guards', {})),
                    physical_tracking_guaranteed=False,
                    method='sequential_original_bounded_dls_v1',
                    assumption='Each predicted joint target is reached exactly before the next IK update.',
                    limitation='Local kinematic verification only; no collision, contact, grasp, '
                               'physical tracking or global unreachability guarantee.')

    def target(self, targets):
        if set(targets) != {'left', 'right'}:
            raise ValueError('Explicit left/right targets required')
        self.check()
        action, diagnostics = [], {}
        for arm in ('left', 'right'):
            robot, target = self.robots[arm], targets[arm]
            p, q = np.asarray(target['position'], float), np.asarray(target['quaternion_wxyz'], float)
            if p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p, q]).all() or abs(np.linalg.norm(q)-1) > 1e-4:
                raise ValueError('Finite pose and unit wxyz required')
            # The shared host checks the .35 rad rotation bound, without a
            # fixed Cartesian translation cap;
            # each real ACK recomputes this bounded robot-only IK update.
            key = self.manager.robot_key[self.manager.robot_list.index(robot)]
            limits = key.data.soft_joint_pos_limits[0, robot.arm_joint_indices].cpu().numpy()
            joints, diagnostics[arm] = self.fk[arm].bounded_target(
                self.manager.get_joint(robot)[0], limits, self.root(arm), target)
            if type(target['gripper_closed']) is not bool:
                raise ValueError('Explicit boolean gripper required')
            opening = target.get('gripper_opening', float(not target['gripper_closed']))
            if not np.isfinite(opening) or not 0 <= opening <= 1:
                raise ValueError('Invalid gripper opening')
            action.extend([*joints.tolist(), opening])
        return dict(action=action, diagnostics=diagnostics, physical_steps=0)
