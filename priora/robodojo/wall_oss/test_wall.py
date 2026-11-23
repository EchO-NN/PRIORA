"""Test native EEF semantics and independent episodes without pretending to test physics."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
from .client import WallTools, WallCodexPolicy
from ..io import InputError
from ..test_support import Sim, response, invoke
from ..skill import run


class Student:
    metadata=dict(backend='WALL-OSS/native-EEF',config='wall',checkpoint='/test',
                  checkpoint_sha256='a'*64,action_horizon=32,action_dim=16)
    def infer(self,obs,path):
        self.prompt=obs['instruction']
        a=np.zeros((32,16),np.float32);a[:,[3,11]]=1;a[:,[7,15]]=.3
        np.savez_compressed(path,actions=a,raw_actions=a)
        return a,dict(prediction_id='wallprediction',inference_seconds=0)


class WallSim(Sim):
    def request(self,op,**args):
        if op=='wall_storage':return dict(free_bytes=100*2**30)
        if op=='wall_robot_geometry':return dict(measured_fk_check=dict(passed=True))
        if op=='wall_eef_step':
            assert args['actions'].shape[1]==16
            return super().request('chunk_step',**args)
        return super().request(op,**args)


def test_native16_ack_and_goal_scope(tmp_path,monkeypatch):
    monkeypatch.setenv('ROLLOUT_MAIN_CAMERA_INPUT','rgb')
    monkeypatch.setenv('WALL_MIN_FREE_BYTES','0')
    student=Student();r=WallTools(tmp_path,'arrange_largest_number',student,rpc_factory=WallSim,max_decisions=0)
    invoke(r);original=r.obs['instruction']
    assert r.set_subtask('Move the left digit into position.')['physical_steps']==0
    packet=invoke(r)
    assert student.prompt=='Move the left digit into position.'
    assert r.obs['instruction']==original and packet['execution_step_limits']['student']==16
    assert packet['student_eef_trajectory'][0]['left']['quaternion_wxyz']==[1,0,0,0]
    with pytest.raises(InputError):r.set_subtask('Change goal after inference')
    d=response(r.request);d['steps']=17
    with pytest.raises(InputError):invoke(r,response=d)
    assert r.tick==0
    result=invoke(r,response=response(r.request))
    assert result['rollout_finished'] and r.tick==2
    assert np.load(tmp_path/'execution_000.npz')['actions'].shape==(2,16)
    assert r.history[0]['executed_gripper_closed']==[dict(left=True,right=True)]*2


def test_whole_task_is_actual_default_without_initial_subtask_call(tmp_path,monkeypatch):
    monkeypatch.setenv('ROLLOUT_MAIN_CAMERA_INPUT','rgb')
    student=Student();r=WallTools(tmp_path,'arrange_largest_number',student,rpc_factory=WallSim,max_decisions=0)
    invoke(r)
    original=r.obs['instruction']
    assert r._policy_observation()['policy_instruction_context']['source']=='native_whole_task'
    invoke(r)
    assert student.prompt==original and r.subtask is None and not r.subtask_history
    assert r.obs['instruction']==original


def test_explicit_full_goal_and_decomposition_report_correct_scope(tmp_path,monkeypatch):
    monkeypatch.setenv('ROLLOUT_MAIN_CAMERA_INPUT','rgb')
    r=WallTools(tmp_path,'arrange_largest_number',Student(),rpc_factory=WallSim)
    invoke(r);original=r.obs['instruction']
    assert r.set_subtask(original)['policy_instruction_context']['source']=='native_whole_task'
    result=r.set_subtask('Move the left digit into position.')
    assert result['policy_instruction_context']['source']=='gpt_subtask'
    assert result['native_task_instruction']==original
    assert result['physical_steps']==0 and result['history_preserved']
    assert r.obs['instruction']==original


def test_required_whole_task_trial_prevents_premature_narrowing(tmp_path,monkeypatch):
    monkeypatch.setenv('ROLLOUT_MAIN_CAMERA_INPUT','rgb')
    monkeypatch.setenv('WALL_REQUIRE_WHOLE_TASK_TRIAL','1')
    student=Student();r=WallTools(tmp_path,'arrange_largest_number',student,rpc_factory=WallSim)
    invoke(r);original=r.obs['instruction']
    with pytest.raises(InputError,match='complete native task'):
        r.set_subtask('Move the left digit only.')
    assert r.subtask is None and not r.subtask_history
    assert r.set_subtask(original)['physical_steps']==0
    invoke(r)
    assert student.prompt==original and r.whole_task_trial_completed
    # Once back at the inference boundary, semantic-error recovery may narrow.
    r.phase='infer'
    assert r.set_subtask('Move the left digit only.')['policy_instruction']== 'Move the left digit only.'


def test_two_clients_never_share_goals_or_hold_state(tmp_path,monkeypatch):
    monkeypatch.setenv('ROLLOUT_MAIN_CAMERA_INPUT','rgb')
    monkeypatch.setenv('ROLLOUT_PI05_GRIPPER_MODE','hold_closed')
    for name in ('a','b'):(tmp_path/name).mkdir()
    a=WallTools(tmp_path/'a','arrange_largest_number',Student(),rpc_factory=WallSim)
    b=WallTools(tmp_path/'b','arrange_largest_number',Student(),rpc_factory=WallSim)
    assert a.gripper_hold is not b.gripper_hold and a.student.rollout is a and b.student.rollout is b
    invoke(a);invoke(b)
    a.set_subtask('Goal A');b.set_subtask('Goal B')
    assert a._policy_observation()['instruction']=='Goal A'
    assert b._policy_observation()['instruction']=='Goal B'
    x=np.zeros((32,16),np.float32);x[:,[3,11]]=1;x[:,[7,15]]=0.2
    y,_=a.preview_student_grippers(x)
    assert np.array_equal(y[:,[3,11]],x[:,[3,11]])
    assert np.all(y[:,[7,15]]==0)


def test_no_jev_and_account_thread_contract(tmp_path,monkeypatch):
    class Transport:
        def __init__(self,*a):self.process=SimpleNamespace(pid=0)
        def request(self,method,params,timeout):
            if method=='thread/start':
                self.params=params
                return dict(thread=dict(id='isolated-test'),model=run.MODEL,reasoningEffort=run.EFFORT)
            return {}
        def notify(self,*a):pass
        def close(self):pass
    monkeypatch.setenv('ROLLOUT_AUTH_PROFILE','desktop_chatgpt')
    monkeypatch.setattr(run.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='test'))
    monkeypatch.setattr('priora.robodojo.wall_oss.client.load_card',
                        lambda path, task: dict(text='Official task detail and scoring.'))
    monkeypatch.setenv('PRIORA_TASK_CARD', 'test.json')
    monkeypatch.setenv('PRIORA_TASK', 'build_tower')
    policy=WallCodexPolicy(tmp_path/'audit','unused',transport_factory=Transport)
    try:
        payload=policy.transport.params
        names=[t['name'] for t in payload['dynamicTools']]
        assert all(t.get('type')=='function' for t in payload['dynamicTools'])
        assert 'wall_set_subtask' in names and not any('onejev' in n or 'moss' in n for n in names)
        assert payload['allowProviderModelFallback'] is False
        assert 'WALL H32' in payload['developerInstructions'] and '1..16' in payload['developerInstructions']
        assert 'H50' not in payload['developerInstructions']
        instructions=payload['developerInstructions']
        assert 'For a single-object task, prioritize this whole-task input' in instructions
        assert 'Your internal task_progress/subgoals need not replace' in instructions
        assert 'not by itself evidence that WALL cannot follow the complete task' in instructions
        assert 'SEMANTIC' in instructions and 'uncertainty alone' in instructions
        assert 'After robodojo_start, use wall_set_subtask' not in instructions
        infer=next(t for t in payload['dynamicTools'] if t['name']=='pi05_infer')
        assert 'No initial wall_set_subtask call is required' in infer['description']
        assert 'For a single-object task' in (policy.workspace/'SKILL.md').read_text()
        assert json.loads((policy.workspace/'launch.json').read_text())['config']['forced_login_method']=='chatgpt'
    finally:policy.close()
