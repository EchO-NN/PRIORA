"""Trajectory, ACK ownership and post-filter collision regression checks."""
from copy import deepcopy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from . import motion_smoothing as smooth
from .client import WallTools, curobo_instructions
from .curobo_kinematics import VERSION, limits
from .test_shared_curobo import adapter
from ..test_eef_kinematic_reachability import native_fk, target_for
from ..robodojo_server.waypoint_guard import verify_report


def dual(x=0, degrees=0, closed=False):
    q=Rotation.from_euler('z',degrees,degrees=True).as_quat()[[3,0,1,2]].tolist()
    return {a:dict(position=[x,0,0],quaternion_wxyz=q.copy(),gripper_closed=closed)
            for a in ('left','right')}


def test_easing_starts_and_ends_gently_without_extra_controls_or_stages():
    begin,end=dual(),dual(.1,90)
    nodes=[dict(label='existing_target',steps=30,target=end)]
    seq=smooth.eef_schedule(nodes,begin)
    assert len(seq)==30 and seq[-1]==end
    dx=np.diff([0]+[s['left']['position'][0] for s in seq])
    assert np.all(dx>0) and dx[0]<dx[14]/20 and dx[-1]<dx[14]/20
    np.testing.assert_allclose(dx,dx[::-1],atol=1e-12)
    assert smooth.metadata()['extra_stages']==smooth.metadata()['extra_controls']==0
    assert begin==dual() and nodes[0]['target']==end


def test_quaternion_shortest_arc_and_original_gripper_timing():
    start=dual(0,179,False);end=dual(.1,-179,True)
    seq=smooth.eef_schedule([dict(label='existing',steps=9,target=end)],start)
    assert all(s['left']['gripper_closed'] for s in seq)
    rotations=[Rotation.from_quat(np.array(s['left']['quaternion_wxyz'])[[1,2,3,0]])
               for s in [start,*seq]]
    assert sum((a.inv()*b).magnitude() for a,b in zip(rotations,rotations[1:]))<.036
    assert all(abs(np.linalg.norm(s['left']['quaternion_wxyz'])-1)<1e-12 for s in seq)
    same=dual();same['left']['quaternion_wxyz']=[-1,0,0,0]
    assert np.isfinite(smooth.eef_schedule([dict(label='same',steps=4,target=same)],dual())[0]['left']['quaternion_wxyz']).all()


def test_existing_nodes_not_rounded_or_gripper_blended():
    first,last=dual(.1,30,False),dual(.1,30,True)
    seq=smooth.eef_schedule([dict(label='user_node_1',steps=8,target=first),
                             dict(label='user_node_2',steps=3,target=last)],dual())
    assert len(seq)==11 and seq[7]==first and seq[-1]==last
    assert not any(x['left']['gripper_closed'] for x in seq[:8])
    assert all(x['left']['gripper_closed'] for x in seq[8:])
    assert seq[8]['left']['position']==last['left']['position']


def test_joint_filter_reduces_alternating_command_jitter_and_respects_limits():
    q=np.zeros(6);previous=None;raw=[];filtered=[]
    for i in range(20):
        desired=np.ones(6)*(.08 if i%2 else -.08)
        q,report=smooth.filter_joint_command(q,desired,previous,.16,np.array([[-.1,.1]]*6))
        previous=q.copy();raw.append(desired[0]);filtered.append(q[0])
        assert report['maximum_joint_step_rad']<=.16+1e-12
    assert np.abs(np.diff(filtered)).sum()<np.abs(np.diff(raw)).sum()*.5
    q,_=smooth.filter_joint_command(np.zeros(6),np.ones(6)*10,None,.16,np.array([[-.05,.05]]*6))
    assert np.max(np.abs(q))<=.05
    with pytest.raises(ValueError):
        smooth.filter_joint_command(np.zeros(6),[np.nan]*6,None,.16,np.array([[-1,1]]*6))


def smooth_adapter(native_fk,monkeypatch):
    monkeypatch.setenv('WALL_MOTION_SMOOTHING','1')
    kin,manager,calls=adapter(native_fk)
    for planner in kin.planners.values():
        planner.check_joint_path=lambda *args:dict(passed=True,elbow_path_ok=True,native_collision_ok=True)
    return kin,manager,calls


def test_preview_and_execution_share_exact_eased_targets_and_filter_state(native_fk,monkeypatch):
    kin,manager,calls=smooth_adapter(native_fk,monkeypatch)
    target=target_for(kin,np.ones(6)*.24)
    nodes=[dict(label='move',steps=16,target=target)]
    initial=deepcopy(manager.q)
    report=kin.check_waypoints(nodes,remaining_steps=30)
    verify_report(report,nodes=nodes,expected_remaining_steps=30,
                  expected_limits=limits(.04),expected_solver=VERSION)
    assert report['accepted'] and not kin.last_joint_commands
    for a in initial:np.testing.assert_array_equal(initial[a],manager.q[a])
    preview_calls=deepcopy(calls);calls.clear()
    seq=smooth.eef_schedule(nodes,report['motion_smoothing']['start_poses'])
    assert smooth.digest(seq)==report['motion_smoothing']['schedule_sha256']
    for t in seq:
        before=deepcopy(kin.last_joint_commands)
        one=kin.target(t);two=kin.target(t)
        np.testing.assert_array_equal(one['action'],two['action'])
        for a in before:np.testing.assert_array_equal(before[a],kin.last_joint_commands[a])
        # Simulate a real ACK. Discarded target calls do not advance the filter.
        for a,offset in [('left',0),('right',7)]:manager.q[a]=np.array(one['action'][offset:offset+6])
        kin.commit_commands(manager.q)
    # Two solver calls per arm above: first proposal of each control must equal preview.
    actual=[value for i,value in enumerate(calls) if i%4<2]
    for before,after in zip(preview_calls,actual):
        for a,b in zip(before,after):np.testing.assert_allclose(a,b,atol=1e-12)
    for a in initial:
        np.testing.assert_allclose(manager.q[a],report['waypoints'][0]['arms'][a]['predicted_joint_positions'])
    before=deepcopy(kin.last_joint_commands)
    kin.check_waypoints(nodes,remaining_steps=30)
    for a in before:np.testing.assert_array_equal(before[a],kin.last_joint_commands[a])


def test_collision_checked_after_filter_and_rejection_holds_both(native_fk,monkeypatch):
    kin,manager,_=smooth_adapter(native_fk,monkeypatch)
    checked=[]
    def reject(current,candidate):
        checked.append(candidate.copy())
        return dict(passed=False,elbow_path_ok=False,native_collision_ok=False)
    kin.planners['left'].check_joint_path=reject
    target=target_for(kin,np.ones(6)*.24)
    proposal=kin.target(target)
    np.testing.assert_allclose(checked[0],np.ones(6)*.08)
    assert proposal['motion_guard']['blocked']
    np.testing.assert_array_equal(proposal['action'][:6],manager.q['left'])
    np.testing.assert_array_equal(proposal['action'][7:13],manager.q['right'])
    assert not kin.last_joint_commands
    report=kin.check_waypoints([dict(label='move',steps=4,target=target)],remaining_steps=10)
    assert not report['accepted']


def test_client_rejects_mismatched_or_stale_smoothing_schedule(native_fk,monkeypatch):
    kin,manager,_=smooth_adapter(native_fk,monkeypatch)
    manager.q={a:np.ones(6)*.20 for a in ('left','right')}
    target=target_for(kin,np.ones(6)*.24);nodes=[dict(label='target',steps=16,target=target)]
    report=kin.check_waypoints(nodes,remaining_steps=30)
    tools=WallTools.__new__(WallTools)
    tools.meta=dict(gpt_eef_solver=VERSION,control_dt=.04,motion_smoothing=smooth.metadata())
    tools.tick=0;tools.request=dict(request_id='test',current_eef=target_for(kin,manager.q['left']))
    response=dict(mode='eef',request_id='test',reason='move',steps=16,target=target,waypoints=[])
    tools.verify_eef_report(report,nodes=nodes,expected_remaining_steps=30)
    seq=tools.correction_targets(response)
    assert smooth.digest(seq)==report['motion_smoothing']['schedule_sha256']
    tools.tick=1
    with pytest.raises(RuntimeError,match='matching'):tools.correction_targets(response)
    bad=deepcopy(report);bad['motion_smoothing']['schedule_sha256']='wrong'
    with pytest.raises(RuntimeError,match='schedule'):tools.verify_eef_report(bad,nodes=nodes,expected_remaining_steps=30)


def test_prompt_states_no_added_stages_and_collision_scope():
    from pathlib import Path
    text=curobo_instructions((Path(__file__).parents[1]/'skill/SKILL.md').read_text())
    assert 'No new stage' in text and 'full extent' in text
    assert 'A cuRobo pass does NOT certify' in text


def test_numeric_json_representation_is_not_a_corrupt_smoothing_report(native_fk,monkeypatch):
    kin,manager,_=smooth_adapter(native_fk,monkeypatch)
    target=target_for(kin,np.ones(6)*.24)
    # JSON accepts integer coordinates, but server checked_targets converts them
    # to floats. A normal reachability rejection must still reach the caller.
    target['left']['position'][0]=0
    nodes=[dict(label='target',steps=16,target=target)]
    report=kin.check_waypoints(nodes,remaining_steps=30)
    assert smooth.digest(nodes)!=report['motion_smoothing']['nodes_sha256']
    tools=WallTools.__new__(WallTools)
    tools.meta=dict(gpt_eef_solver=VERSION,control_dt=.04,motion_smoothing=smooth.metadata())
    tools.tick=0
    verified=tools.verify_eef_report(report,nodes=nodes,expected_remaining_steps=30)
    assert not verified['accepted']
    assert smooth.digest(tools._verified_smooth_eef[2])==report['motion_smoothing']['schedule_sha256']


def test_execute_simulator_rounding_and_reject_changed_controls(native_fk, monkeypatch):
    kin, manager, _ = smooth_adapter(native_fk, monkeypatch)
    target = target_for(kin, np.ones(6) * .24)
    nodes = [dict(label='target', steps=16, target=target)]
    report = kin.check_waypoints(nodes, remaining_steps=30)
    detail = report['motion_smoothing']
    control = detail['eef_schedule'][0]['left']
    q = control['quaternion_wxyz'][0]
    control['quaternion_wxyz'][0] = float(np.nextafter(q, 1. if q < 1. else 0.))
    detail['schedule_sha256'] = smooth.digest(detail['eef_schedule'])
    tools = WallTools.__new__(WallTools)
    tools.meta = dict(gpt_eef_solver=VERSION, control_dt=.04, motion_smoothing=smooth.metadata())
    tools.tick = 0
    public = tools.verify_eef_report(report, nodes=nodes, expected_remaining_steps=30)
    assert 'eef_schedule' not in public['motion_smoothing']
    assert smooth.digest(tools._verified_smooth_eef[2]) == detail['schedule_sha256']
    control['position'][0] += 1e-5
    detail['schedule_sha256'] = smooth.digest(detail['eef_schedule'])
    with pytest.raises(RuntimeError, match='position'):
        tools.verify_eef_report(report, nodes=nodes, expected_remaining_steps=30)
