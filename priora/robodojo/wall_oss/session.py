"""Native WALL EEF actions and robot observations only; no task truth RPC."""
import time
import shutil
import os
import numpy as np

from ..io import require, write_json
from ..robodojo_server.session import RoboDojoSession
from .ik_guard import VERSION


class WallSession(RoboDojoSession):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.metadata.update(student_backend='WALL-OSS/native-EEF',student_action_dim=16,
            student_action_horizon=32,student_max_steps=16,native_eef_ik_enabled=True)
        self.shared_curobo = os.environ.get('WALL_SHARED_CUROBO') == '1'
        self.motion_smoothing = os.environ.get('WALL_MOTION_SMOOTHING') == '1'
        if self.motion_smoothing and not self.shared_curobo:
            raise RuntimeError('Motion smoothing requires shared cuRobo preflight and execution')
        if self.motion_smoothing:
            from .motion_smoothing import metadata
            self.metadata['motion_smoothing'] = metadata()
        self.guarded_ik = self.shared_curobo or os.environ.get('WALL_IK_GUARD') == '1'
        self._pending_correction = None
        if self.shared_curobo:
            from .curobo_kinematics import VERSION as SHARED_VERSION, limits
            self.metadata.update(gpt_eef_solver=SHARED_VERSION,
                eef_reachability_limits=limits(self.metadata['control_dt']),
                collision_scope='native_self_collision_and_table_plus_elbow_mesh',
                collision_path_samples_per_control=9)
        if self.guarded_ik:
            for planner in self.env.robot_manager.ik_solver.values():
                planner.control_dt = self.metadata['control_dt']
            self.metadata['ik_guard'] = dict(version=VERSION,
                scope='link2/link3 mesh interference and joint-command continuity',
                maximum_joint_step_rad=5*self.metadata['control_dt']*.8,
                full_scene_collision_checked=False)
        self._release_idle_cache()

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        self._pending_correction = None
        if self.shared_curobo:
            from .curobo_kinematics import CuroboKinematics
            self.kinematics = CuroboKinematics(self.env, self.metadata['control_dt'])
        elif self.guarded_ik:
            self.kinematics.joint_path_guards = {arm:
                self.env.robot_manager.ik_solver[robot.robot_name].path_guard
                for arm, robot in self.kinematics.robots.items()}
        return result

    @staticmethod
    def _release_idle_cache():
        if os.environ.get('WALL_NATIVE_IK_ONLY') == '1':
            import torch
            torch.cuda.empty_cache()

    def dispatch(self, op, args):
        if self.shared_curobo and op == 'eef_joint_target':
            self._check_identity(args['episode_id'], args['step_id'])
            result = self.kinematics.target(args['targets'])
            self._pending_correction = dict(step_id=self.step_id, **result)
            return result
        if op not in ('wall_eef_step', 'wall_history', 'wall_robot_geometry', 'wall_storage'):
            return super().dispatch(op, args)
        self._check_identity(args['episode_id'], args['step_id'])
        if op == 'wall_storage':
            self._release_idle_cache()
            import torch
            return dict(free_bytes=shutil.disk_usage(self.output).free,physical_steps=0,
                        cuda_allocated_bytes=torch.cuda.memory_allocated(),
                        cuda_reserved_bytes=torch.cuda.memory_reserved())
        if op == 'wall_robot_geometry':
            return dict(fingertip_geometry=self.kinematics.fingertip_geometry,
                        measured_fk_check=self.kinematics.check(), physical_steps=0)
        if op == 'wall_history':
            after = args['after']
            require(type(after) is int and -1 <= after <= self.step_id, 'Invalid history cursor')
            require(self.step_id-after <= 48, 'WALL history advance exceeds protocol budget')
            frames = []
            for tick in range(after+1, self.step_id+1):
                with np.load(self.episode_dir/'observations'/f'{tick:06d}.npz') as row:
                    frames.append(dict(sim_frame=tick, face_view=row['cam_high'].copy()))
            return dict(frames=frames, physical_steps=0)
        return self.wall_eef_step(args['actions'], args.get('student_prediction_id'))

    def _joint_path_report(self, action):
        arms = {}
        for arm, offset in (('left', 0), ('right', 7)):
            robot = self.kinematics.robots[arm]
            planner = self.env.robot_manager.ik_solver[robot.robot_name]
            current = self.env.robot_manager.get_joint(robot)[0]
            arms[arm] = (planner.check_joint_path(current, action[offset:offset+6])
                         if self.shared_curobo else
                         planner.path_guard.check_path(current, action[offset:offset+6]))
        return dict(version=VERSION, blocked=not all(v['passed'] for v in arms.values()), arms=arms)

    def chunk_step(self, actions, **kwargs):
        if not self.guarded_ik:
            return super().chunk_step(actions, **kwargs)
        a = np.asarray(actions, np.float32)
        require(a.ndim == 2 and a.shape[1] == 14 and 1 <= len(a) <= 30 and
                np.isfinite(a).all(), 'Invalid guarded joint action')
        require(np.all((a[:, [6, 13]] >= 0) & (a[:, [6, 13]] <= 1)), 'Invalid gripper opening')
        rows = []
        pending = self._pending_correction
        if self.shared_curobo and self.source in ('gpt_eef', 'gpt_joint'):
            require(pending is not None and pending['step_id'] == self.step_id and len(a) == 1 and
                    np.array_equal(a[0], np.asarray(pending['action'], np.float32)),
                    'GPT correction must use the current shared-cuRobo target')
        self._pending_correction = None
        for requested in a:
            guard = self._joint_path_report(requested)
            if pending is not None:
                guard['solver_report'] = pending['motion_guard']
                guard['blocked'] |= pending['motion_guard']['blocked']
            actual = np.asarray(self.obs['states'], np.float32) if guard['blocked'] else requested
            result = super().chunk_step([actual], **kwargs)
            if self.shared_curobo and result['steps']:
                # Commit exactly the acknowledged command, including a forced
                # hold. Merely requesting an IK target must not advance filters.
                acknowledged=np.asarray(result['steps'][-1]['executed_action'])
                self.kinematics.commit_commands({arm:acknowledged[offset:offset+6]
                    for arm,offset in (('left',0),('right',7))})
            for row in result['steps']:
                row['motion_guard'] = guard
                write_json(self.episode_dir/f"action_{row['step_id']-1:06d}.json", dict(row,
                    executed_action=np.asarray(row['executed_action']).tolist(),
                    requested_joint_action=requested.tolist(),
                    obs=dict(states=np.asarray(row['obs']['states']).tolist())))
            rows.extend(result['steps'])
            if guard['blocked'] or any(r['terminated'] or r['truncated'] for r in result['steps']):
                break
        return dict(episode_id=self.episode_id, step_id=self.step_id, steps=rows, motion_guard=guard)

    def wall_eef_step(self, actions, prediction):
        require(not self.finished and not self.terminated and not self.truncated, 'Episode finished')
        a = np.asarray(actions, np.float32)
        require(a.ndim == 2 and a.shape[1] == 16 and 1 <= len(a) <= 16
                and np.isfinite(a).all(), 'Expected 1..16 finite absolute WALL EEF actions')
        require(np.all((a[:, [7, 15]] >= 0) & (a[:, [7, 15]] <= 1)), 'Invalid gripper opening')
        for columns in (slice(3, 7), slice(11, 15)):
            require(np.allclose(np.linalg.norm(a[:, columns], axis=1), 1, atol=1e-3), 'Invalid quaternion')
        rows = []
        for action in a:
            requested = action.copy()
            guard = None
            command = {key: value for arm, offset in (('left', 0), ('right', 8))
                for key, value in ((f'{arm}_ee_pose', action[offset:offset+7]),
                                  (f'{arm}_ee_joint_state', action[offset+7:offset+8]))}
            if self.guarded_ik:
                # Solve both arms before submitting either. Explicit joint hold
                # on failure avoids native EEF failure retaining an old target.
                if self.shared_curobo:
                    states=self.kinematics.measured_joints()
                    targets={arm:dict(position=action[offset:offset+3].tolist(),
                        quaternion_wxyz=action[offset+3:offset+7].tolist())
                        for arm,offset in (('left',0),('right',8))}
                    joint_commands,guard=self.kinematics.solve_step(states,targets)
                else:
                    solutions = {arm: self.env.robot_manager.solve_ik(
                        target_pose=command[f'{arm}_ee_pose'], env_idx=0, robot=self.kinematics.robots[arm])
                        for arm in ('left', 'right')}
                    guard = dict(version=VERSION, arms={arm: value['motion_guard']
                        for arm, value in solutions.items()})
                    guard['blocked'] = any(v['blocked'] for v in guard['arms'].values())
                    joint_commands={arm:value['joint_value'] for arm,value in solutions.items()}
                command = {}
                action = action.copy()
                for arm, offset, state_offset in (('left', 0, 0), ('right', 8, 7)):
                    q = (self.env.robot_manager.get_joint(self.kinematics.robots[arm])[0]
                         if guard['blocked'] else joint_commands[arm])
                    opening = self.obs['states'][state_offset+6] if guard['blocked'] else action[offset+7]
                    command[f'{arm}_arm_joint_state'] = np.asarray(q, np.float32)
                    command[f'{arm}_ee_joint_state'] = np.asarray([opening], np.float32)
                    # Audit the actual commanded FK target, separately from the
                    # student's desired EEF target. Neither claims real arrival.
                    from ..robodojo_server.action_edit_kinematics import pose
                    target = pose(self.kinematics.root(arm) @ self.kinematics.fk[arm].matrix(q))
                    action[offset:offset+7] = target['position']+target['quaternion_wxyz']
                    action[offset+7] = opening
            before = int(self.env.take_action_cnt[0]); begin = time.perf_counter_ns()
            self.env.take_action(command)
            require(int(self.env.take_action_cnt[0]) == before+1, 'Native action ACK mismatch')
            if self.shared_curobo:
                self.kinematics.commit_commands({arm:command[f'{arm}_arm_joint_state']
                                                for arm in ('left','right')})
            elapsed = time.perf_counter_ns()-begin
            self.step_id += 1
            ended = bool(self.env.end_flag[0])
            self.success = bool(ended and self.env.success[0])
            self.truncated = bool(ended and not self.success and self.step_id >= self.env.step_lim)
            self.terminated = bool(ended and not self.truncated)
            row = dict(valid=True, executed_action=action.copy(),
                action_format='left_xyz_wxyz_opening_right_xyz_wxyz_opening',
                native_action_elapsed_ns=elapsed, step_id=self.step_id,
                terminated=self.terminated, truncated=self.truncated, success=self.success,
                source=self.source, control_epoch=self.control_epoch, student_prediction_id=prediction,
                obs=dict(states=np.full(14, np.nan)), measurement_available=False)
            rows.append(row)
            if guard is not None:
                row.update(motion_guard=guard, requested_eef_action=requested.tolist(),
                    commanded_arm_joints={arm: command[f'{arm}_arm_joint_state'].tolist()
                                           for arm in ('left', 'right')})
            # A post-ACK error must not erase or replay physical execution.
            try:
                obs = self._observe(record=True)
                row.update(obs=dict(states=obs['states']), measurement_available=True)
                write_json(self.episode_dir/f'action_{self.step_id-1:06d}.json',
                    dict(row, executed_action=action.tolist(), obs=dict(states=obs['states'].tolist())))
            except Exception as error:
                self.poisoned = True
                fault = dict(error=str(error), after_ack=True, step_id=self.step_id, no_action_retry=True)
                write_json(self.output/'wall_capture_fault.json', fault)
                return dict(episode_id=self.episode_id, step_id=self.step_id,
                            steps=rows, infrastructure_fault=fault)
            if ended:
                self._write_summary('terminal')
                break
            if guard is not None and guard['blocked']:
                break  # Return the real hold ACK and fresh observation to GPT.
        result = dict(episode_id=self.episode_id, step_id=self.step_id, steps=rows)
        if self.guarded_ik:
            result['motion_guard'] = dict(version=VERSION,
                blocked=any(r['motion_guard']['blocked'] for r in rows),
                rate_limited_controls=sum(any(v.get('rate_limited', False)
                    for v in r['motion_guard']['arms'].values()) for r in rows),
                last=rows[-1]['motion_guard'],
                note='Unsafe IK holds both arms and grippers for one real control, then returns. '
                     'Use a different reachable correction; no earlier actions were rolled back.')
        return result
