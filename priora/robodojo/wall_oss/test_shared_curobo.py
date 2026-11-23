from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from ..test_eef_kinematic_reachability import native_fk, target_for
from ..robodojo_server.waypoint_guard import verify_report
from .curobo_kinematics import CuroboKinematics, VERSION, limits
from .client import curobo_instructions, CUROBO_GUIDE


def adapter(native_fk, blocked=False):
    kin,manager=native_fk; calls=[]
    class Planner:
        path_guard=SimpleNamespace(limits=np.array([[-2,2]]*6))
        def solve_ik_to_joint(self,current,target,real_robot_pose):
            calls.append((current.copy(),np.array(target),deepcopy(real_robot_pose)))
            desired=np.ones(6)*.24
            nxt=current+np.clip(desired-current,-.16,.16)
            return dict(status='Success',joint_value=current.copy() if blocked else nxt,
                motion_guard=dict(blocked=blocked,collision_path=dict(passed=not blocked,elbow_path_ok=not blocked)))
    manager.ik_solver={}
    for i,r in enumerate(manager.robot_list):
        r.robot_name='robot'+str(i);r.entity_origin_pose=[i,2,3,1,0,0,0]
        manager.ik_solver[r.robot_name]=Planner()
    return CuroboKinematics(kin.env,.04),manager,calls


def test_preview_and_actual_share_solver_transform_and_limits(native_fk):
    kin,manager,calls=adapter(native_fk);initial=deepcopy(manager.q)
    target=target_for(kin,np.ones(6)*.24);nodes=[dict(label='move',steps=2,target=target)]
    report=kin.check_waypoints(nodes,remaining_steps=20)
    verify_report(report,nodes=nodes,expected_remaining_steps=20,
                  expected_limits=limits(.04),expected_solver=VERSION)
    assert report['accepted'] and report['physical_steps']==0 and len(calls)==4
    for arm in manager.q:np.testing.assert_array_equal(manager.q[arm],initial[arm])
    preview_first=calls[:2];calls.clear()
    proposal=kin.target(target)
    assert len(calls)==2 and not proposal['motion_guard']['blocked']
    np.testing.assert_allclose(proposal['action'][:6],[.16]*6)
    for before,after in zip(preview_first,calls):
        for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
    for arm in manager.q:manager.q[arm]=np.ones(6)*.16
    proposal=kin.target(target)
    np.testing.assert_allclose(proposal['action'][:6],report['waypoints'][0]['arms']['left']['predicted_joint_positions'])


def test_collision_rejection_is_consistent_and_holds_both_arms(native_fk):
    kin,manager,_=adapter(native_fk,blocked=True)
    target=target_for(kin,np.ones(6)*.24);nodes=[dict(label='blocked',steps=3,target=target)]
    report=kin.check_waypoints(nodes,remaining_steps=20)
    verify_report(report,nodes=nodes,expected_remaining_steps=20,
                  expected_limits=limits(.04),expected_solver=VERSION)
    assert not report['accepted']
    action=kin.target(target)
    assert action['motion_guard']['blocked']
    np.testing.assert_array_equal(action['action'][:6],manager.q['left'])
    bad=deepcopy(report);bad['waypoints'][0]['arms']['left'].pop('collision_path_ok')
    with pytest.raises(RuntimeError):verify_report(bad,nodes=nodes,expected_remaining_steps=20,
        expected_limits=limits(.04),expected_solver=VERSION)


def test_new_system_and_manual_replace_legacy_increments():
    from pathlib import Path
    base=Path(__file__).parents[1]/'skill'
    for name in ['SKILL.md','context/eef_control.md']:
        text=curobo_instructions((base/name).read_text())
        assert CUROBO_GUIDE.strip() in text
        assert '0.05 rad' not in text and 'No cuRobo planning' not in text
