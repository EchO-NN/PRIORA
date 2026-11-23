"""Use native cuRobo IK without unused trajectory-optimizer CUDA graphs.

Native evaluation's EEF branch calls solve_ik_to_joint; demonstration arms replay
recorded joint commands. Retain native 32-seed collision-aware optimization.
The opt-in guard selects a continuous branch and checks elbow mesh interference.
"""
def install():
    import os
    import numpy as np
    import yaml
    from copy import deepcopy
    from curobo.inverse_kinematics import InverseKinematics
    from curobo.types import DeviceCfg, GoalToolPose, JointState, ToolPoseCriteria
    from env.planner_manager.curobo_planner import CuroboPlanner
    from env.robot_manager import robot_manager
    from env.planner_manager.curobo_planner import calculate_target_pose
    from .ik_guard import JointPathGuard, VERSION

    class NativeIKOnly(CuroboPlanner):
        def __init__(self, robot_origin_pose, active_joints_name, all_joints,
                     dt=None, yml_path=None, table_height=0.74):
            if yml_path is None:
                raise ValueError('Native IK needs the robot configuration')
            self.yml_path = yml_path
            self.dt = dt
            self.robot_origin_pose = robot_origin_pose
            self.active_joints_name = active_joints_name
            self.all_joints = all_joints
            with open(yml_path) as stream:
                cfg = yaml.safe_load(stream)
            self.frame_bias = cfg['planner']['frame_bias']
            self.robot_cfg, self.scene_model = self._build_robot_and_scene_cfg(cfg, table_height)
            self.path_guard = (JointPathGuard(
                cfg['robot_cfg']['kinematics']['urdf_path'], active_joints_name)
                if os.environ.get('WALL_IK_GUARD') == '1' or
                   os.environ.get('WALL_SHARED_CUROBO') == '1' else None)
            self.control_dt = None  # Session supplies observation/control dt, not physics dt.
            cspace = self.robot_cfg['robot_cfg']['kinematics'].get('cspace', {})
            self.cspace_joint_names = list(cspace.get('joint_names', active_joints_name))
            retract = cspace.get('default_joint_position')
            self.cspace_retract_config = (np.zeros(len(self.cspace_joint_names), dtype=np.float32)
                if retract is None else np.asarray(retract, dtype=np.float32).reshape(-1))
            if len(self.cspace_retract_config) != len(self.cspace_joint_names):
                raise ValueError('Native IK cspace/retract size mismatch')
            self.device_cfg = DeviceCfg()
            self.use_cuda_graph = os.environ.get('WALL_IK_CUDA_GRAPH', '1') == '1'
            self.ik_solver = InverseKinematics(self._build_ik_cfg())
            names = list(self.ik_solver.joint_names)
            self.cspace_retract_config = np.asarray([
                self.cspace_retract_config[self.cspace_joint_names.index(n)]
                if n in self.cspace_joint_names else 0.0 for n in names], dtype=np.float32)
            self.cspace_joint_names = names
            self._active_to_cspace_idx = [names.index(n) for n in active_joints_name]
            self.tool_frames = list(self.ik_solver.tool_frames)
            if not self.tool_frames:
                raise ValueError('Native IK has no tool frames')
            self.ee_link = self.tool_frames[0]
            self._default_tool_pose_criteria = {
                f: ToolPoseCriteria(device_cfg=self.device_cfg) for f in self.tool_frames}
            # Same standalone IK warmup as the original planner; no TrajOpt warmup.
            retract = self.ik_solver.default_joint_state.clone().unsqueeze(0)
            js = JointState.from_position(retract.position, joint_names=names)
            kin = self.ik_solver.compute_kinematics(js)
            goal = GoalToolPose.from_poses(kin.tool_poses.to_dict(),
                ordered_tool_frames=self.tool_frames, num_goalset=1)
            self.ik_solver.solve_pose(goal_tool_poses=goal, current_state=js, return_seeds=1)

        def solve_ik_to_joint(self, curr_joint_pos, target_ee_pose, real_robot_pose, num_seeds=32):
            if self.path_guard is None:
                return super().solve_ik_to_joint(curr_joint_pos, target_ee_pose, real_robot_pose, num_seeds)
            if self.control_dt is None:
                raise RuntimeError('Guarded IK requires the real control period')
            target = calculate_target_pose(real_robot_pose, self.robot_origin_pose, deepcopy(target_ee_pose))
            p, q = self._trans_from_world_to_base(np.asarray(self.robot_origin_pose, np.float32),
                                                 np.asarray(target, np.float32))
            p += np.asarray(self.frame_bias)
            result = self.ik_solver.solve_pose(goal_tool_poses=self._build_goal_pose(p, q),
                current_state=self._build_joint_state(curr_joint_pos), return_seeds=32)
            values = np.asarray(result.js_solution.position.detach().cpu())
            names = list(result.js_solution.joint_names)
            values = values.reshape(-1, values.shape[-1])[:, [names.index(n) for n in self.active_joints_name]]
            success = np.asarray(result.success.detach().cpu()).reshape(-1)
            # Native X5 actuator max velocity is 5 rad/s; the environment
            # interpolates across the first 80% of each control period.
            selected, guard = self.path_guard.select(curr_joint_pos, values, success,
                                                     max_step=5.0*self.control_dt*0.8)
            if os.environ.get('WALL_SHARED_CUROBO') == '1' and not guard['blocked']:
                collision = self.check_joint_path(curr_joint_pos, selected)
                guard['collision_path'] = collision
                if not collision['passed']:
                    selected = np.asarray(curr_joint_pos).copy()
                    guard.update(blocked=True, reason='curobo_interpolated_path_rejected')
            return dict(status='Success', joint_names=self.active_joints_name,
                        joint_value=selected.astype(np.float32), motion_guard=guard)

        def check_joint_path(self, current, candidate):
            """Same native self/table constraints on the commanded joint segment.

            Sampled collision test, not a continuous swept-volume guarantee.
            Uses the independent auxiliary rollout, never steps physics or
            changes the optimization goal/constraints.
            """
            import torch
            q = np.linspace(np.asarray(current), np.asarray(candidate), 9)
            js = [self._build_cspace_joint_values(row) for row in q]
            samples = torch.as_tensor(np.asarray(js), device=self.device_cfg.device,
                                      dtype=self.device_cfg.dtype).unsqueeze(1)
            metrics = self.ik_solver.core.auxiliary_rollout.compute_metrics_from_action(samples)
            feasible = metrics.costs_and_constraints.get_feasible(
                include_all_hybrid=False, sum_horizon=True)
            passed = bool(feasible) if isinstance(feasible, bool) else bool(feasible.all().item())
            elbow = self.path_guard.check_path(current, candidate)
            return dict(passed=passed and elbow['passed'], native_collision_ok=passed,
                elbow_path_ok=elbow['passed'], samples=9,
                scope='native_self_collision_and_table_plus_elbow_mesh',
                full_scene_collision_checked=False, continuous_collision_guaranteed=False)

        def _unused(self, *args, **kwargs):
            raise RuntimeError('WALL evaluation uses native IK, not trajectory planning')
        plan_path = plan_batch = _unused

    robot_manager.CuroboPlanner = NativeIKOnly
    return dict(backend='native_curobo_ik_only', num_seeds=32,
                position_tolerance=0.001, orientation_tolerance=0.02,
                self_collision_check=True, trajectory_planners_loaded=False,
                continuity_guard=VERSION if os.environ.get('WALL_IK_GUARD') == '1' or
                    os.environ.get('WALL_SHARED_CUROBO') == '1' else None,
                shared_gpt_curobo=os.environ.get('WALL_SHARED_CUROBO') == '1',
                use_cuda_graph=os.environ.get('WALL_IK_CUDA_GRAPH', '1') == '1')
