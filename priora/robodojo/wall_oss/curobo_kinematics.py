"""GPT preview and real corrections share WALL's native cuRobo step solver."""
from copy import deepcopy
import os
import numpy as np

from ..robodojo_server.kinematics import DualKinematics
from ..robodojo_server.action_edit_kinematics import pose
from ..robodojo_server import reachability
from ..execution_limits import EEF_MAX_STEPS
from . import motion_smoothing

VERSION = 'wall_shared_curobo_v1'


def limits(control_dt):
    if not np.isfinite(control_dt) or not 0 < control_dt <= .1:
        raise ValueError('Invalid native control period')
    return dict(reachability.limits(), joint_step_limit_rad=float(5.0*control_dt*.8))


class CuroboKinematics(DualKinematics):
    def __init__(self, env, control_dt):
        super().__init__(env)
        self.control_dt = control_dt
        self.step_limits = limits(control_dt)
        self.smoothing_enabled = os.environ.get('WALL_MOTION_SMOOTHING') == '1'
        self.last_joint_commands = {}
        self.planners = {arm:self.manager.ik_solver[robot.robot_name]
                         for arm,robot in self.robots.items()}
        for planner in self.planners.values():
            if planner.path_guard is None:
                raise RuntimeError('Shared cuRobo requires the continuous joint guard')
            planner.control_dt = control_dt

    def measured_joints(self):
        return {arm:np.asarray(self.manager.get_joint(robot)[0],float).copy()
                for arm,robot in self.robots.items()}

    def commit_commands(self, commands):
        """Call only after a real native ACK; previews must not mutate this."""
        self.last_joint_commands = {arm: np.asarray(commands[arm], float).copy()
                                    for arm in ('left', 'right')}

    def solve_step(self, states, targets, *, filter_state=None):
        """No simulated step: same world/base transform and solver as WALL."""
        results = {}
        for arm,robot in self.robots.items():
            target = targets[arm]
            p,q = reachability.checked_pose(target['position'],target['quaternion_wxyz'])
            result = self.planners[arm].solve_ik_to_joint(states[arm], np.r_[p,q],
                real_robot_pose=deepcopy(robot.entity_origin_pose))
            if result.get('status') != 'Success' or 'motion_guard' not in result:
                raise RuntimeError('Shared cuRobo returned an incomplete guarded result')
            joints=np.asarray(result['joint_value'],float)
            if joints.shape!=states[arm].shape or not np.isfinite(joints).all():
                raise RuntimeError('Shared cuRobo produced invalid joints')
            results[arm]=dict(joints=joints, guard=result['motion_guard'])
        blocked=any(r['guard']['blocked'] for r in results.values())
        if self.smoothing_enabled and not blocked:
            previous = self.last_joint_commands if filter_state is None else filter_state
            for arm, result in results.items():
                planner = self.planners[arm]
                result['joints'], smoothing = motion_smoothing.filter_joint_command(
                    states[arm], result['joints'], previous.get(arm),
                    self.step_limits['joint_step_limit_rad'], planner.path_guard.limits)
                # Filtering changes the joint path: validate the ACTUAL command,
                # not only the raw IK solution. Never bypass a rejection.
                collision = planner.check_joint_path(states[arm], result['joints'])
                result['guard'] = dict(result['guard'], smoothing=smoothing,
                    collision_path=collision, blocked=not collision['passed'])
                if not collision['passed']:
                    result['guard']['reason'] = 'smoothed_joint_path_rejected'
            blocked = any(r['guard']['blocked'] for r in results.values())
        commands={arm:states[arm].copy() if blocked else r['joints']
                  for arm,r in results.items()}
        guard=dict(version=VERSION, blocked=blocked,
                   arms={arm:r['guard'] for arm,r in results.items()})
        return commands,guard

    def target(self, targets):
        if set(targets)!={'left','right'}:raise ValueError('Explicit dual EEF targets required')
        self.check()
        states=self.measured_joints();commands,guard=self.solve_step(states,targets)
        action=[];diagnostics={}
        for arm in ('left','right'):
            t=targets[arm]
            if type(t['gripper_closed']) is not bool:raise ValueError('Boolean gripper required')
            opening=t.get('gripper_opening',float(not t['gripper_closed']))
            if not np.isfinite(opening) or not 0<=opening<=1:raise ValueError('Invalid gripper opening')
            action.extend([*commands[arm].tolist(),opening])
            diagnostics[arm]=dict(method=VERSION, motion_guard=guard['arms'][arm],
                proposed_joint_delta=(commands[arm]-states[arm]).tolist())
        return dict(action=action,diagnostics=diagnostics,motion_guard=guard,physical_steps=0)

    def check_targets(self, targets):
        report=super().check_targets(targets)
        report.update(limits=self.step_limits,solver=VERSION)
        return report

    def check_waypoints(self, waypoints, remaining_steps=None):
        if not isinstance(waypoints,list) or not 1<=len(waypoints)<=EEF_MAX_STEPS:
            raise ValueError('Nonempty bounded waypoint list required')
        if type(remaining_steps) is not int or remaining_steps<0:
            raise ValueError('Native remaining control budget required')
        nodes=[]
        for row in waypoints:
            if not isinstance(row,dict) or set(row)!={'label','steps','target'}:
                raise ValueError('Waypoint requires label, steps and target')
            if not isinstance(row['label'],str) or not row['label'].strip():raise ValueError('Empty label')
            if type(row['steps']) is not int or not 1<=row['steps']<=EEF_MAX_STEPS:
                raise ValueError('Invalid waypoint steps')
            nodes.append(dict(row,target=reachability.checked_targets(row['target'])))
        total=sum(n['steps'] for n in nodes)
        if total>EEF_MAX_STEPS:raise ValueError('Total EEF controls exceed budget')
        fk_check=self.check();states=self.measured_joints()
        roots={a:self.root(a) for a in self.robots};reports=[]
        start_poses={a:pose(roots[a]@self.fk[a].matrix(states[a])) for a in self.robots}
        schedule=(motion_smoothing.eef_schedule(nodes,start_poses) if self.smoothing_enabled
                  else [deepcopy(n['target']) for n in nodes for _ in range(n['steps'])])
        filter_state=deepcopy(self.last_joint_commands)
        cursor=0
        for index,node in enumerate(nodes):
            okay={a:dict(ik_solution_ok=True,collision_path_ok=True,elbow_path_ok=True,
                         joint_limits_ok=True,joint_step_limits_ok=True) for a in self.robots}
            maximum={a:0. for a in self.robots}; last_guard=None
            for _ in range(node['steps']):
                new,last_guard=self.solve_step(states,schedule[cursor],filter_state=filter_state)
                cursor+=1
                for arm in self.robots:
                    g=last_guard['arms'][arm];path=g.get('collision_path',{})
                    okay[arm]['ik_solution_ok'] &= not g['blocked']
                    okay[arm]['collision_path_ok'] &= not g['blocked'] and path.get('passed',False)
                    okay[arm]['elbow_path_ok'] &= not g['blocked'] and path.get('elbow_path_ok',False)
                    bounds=self.planners[arm].path_guard.limits
                    maximum[arm]=max(maximum[arm],float(np.max(np.abs(new[arm]-states[arm]))))
                    okay[arm]['joint_limits_ok'] &= bool(np.all(new[arm]>=bounds[:,0]-1e-6) and
                                                       np.all(new[arm]<=bounds[:,1]+1e-6))
                    okay[arm]['joint_step_limits_ok'] &= maximum[arm]<=self.step_limits['joint_step_limit_rad']+1e-6
                states=new
                filter_state={a:q.copy() for a,q in new.items()}
                if last_guard['blocked']:
                    # Keep following waypoint schedules aligned after rejection.
                    cursor=sum(n['steps'] for n in nodes[:index+1])
                    break
            arms={}
            for arm in self.robots:
                predicted=pose(roots[arm]@self.fk[arm].matrix(states[arm]))
                errors=reachability.pose_errors(predicted['position'],predicted['quaternion_wxyz'],node['target'][arm])
                arms[arm]=dict(errors,**okay[arm],maximum_joint_step_rad=maximum[arm],
                    predicted_joint_positions=states[arm].tolist(),predicted_pose=predicted)
            passed=all(a['within_tolerance'] and all(okay[arm].values()) for arm,a in arms.items())
            reports.append(dict(index=index,label=node['label'],requested_steps=node['steps'],
                                passed=passed,arms=arms,last_motion_guard=last_guard))
        budget_ok=total<=remaining_steps;accepted=budget_ok and all(n['passed'] for n in reports)
        report=dict(schema='eef_reachability_v1',solver=VERSION,method=VERSION,
            accepted=accepted,physical_steps=0,limits=self.step_limits,waypoints=reports,
            total_steps=total,remaining_steps=remaining_steps,remaining_steps_sufficient=budget_ok,
            measured_fk_check=fk_check,elbow_path_checked=True,collision_checked=True,
            collision_scope='native_self_collision_and_table_plus_elbow_mesh',
            collision_path_samples_per_control=9,full_scene_collision_checked=False,
            continuous_collision_guaranteed=False,physical_tracking_guaranteed=False,
            reason='insufficient_remaining_native_steps' if not budget_ok else
                   'all_waypoints_verified_with_shared_curobo' if accepted else
                   'shared_curobo_no_safe_arrival_within_requested_controls',
            assumption='Preview assumes each command is reached; real execution recomputes '
                       'the same solver from measured joints after every ACK.')
        if self.smoothing_enabled:
            report['motion_smoothing']=dict(motion_smoothing.metadata(),start_poses=start_poses,
                nodes_sha256=motion_smoothing.digest(nodes),
                schedule_sha256=motion_smoothing.digest(schedule), eef_schedule=schedule)
        return report
