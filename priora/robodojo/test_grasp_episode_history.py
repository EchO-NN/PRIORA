import base64
import json
from types import SimpleNamespace

import numpy as np
import pytest

from .grasp_feedback import capture, teacher_packet
from .justrua_sensors.readings import render_fingers
from .teacher_history import EpisodeImageArchive, TeacherEpisodeHistory
from .skill.run import content_items


def make_archive(tmp_path, end=500):
    archive = EpisodeImageArchive(tmp_path/'server', 'episode1', .04)
    for step in range(end+1):
        archive.capture(step, np.full((16, 24, 3), step % 255, dtype=np.uint8))
    return archive


def test_2hz_real_samples_and_paginated_backfill(tmp_path):
    archive = make_archive(tmp_path)
    assert [x['step_id'] for x in archive.rows[:5]] == [0, 13, 25, 38, 50]
    assert len(archive.rows) == 41
    assert all(abs(x['simulation_time_s']-x['sampling_target_time_s']) <= .0200001
               for x in archive.rows)
    client = TeacherEpisodeHistory(tmp_path/'client')
    calls = []
    def rpc(op, **args):
        calls.append(op)
        return archive.page(args['cursor'], args['through_step'])
    rollout = SimpleNamespace(episode='episode1', tick=500,
        meta={'provides_teacher_episode_history':True}, _rpc=rpc)
    packet, cursor = client.prepare(rollout, {'images':[], 'step_id':500})
    assert len(packet['images']) == cursor == 41
    assert calls == ['teacher_history'] * 3
    items = content_items(packet, images=True)
    assert len(items) == 1+41*2
    for row, label, payload in zip(client.rows, items[1::2], items[2::2]):
        assert row['label'] == label['text']
        assert base64.b64decode(payload['imageUrl'].split(',')[1]) == open(row['path'],'rb').read()
    # Before delivery confirmation retry has every frame, after it none is repeated.
    assert client.prepare(rollout, {'images':[]})[0]['gpt_episode_history']['new_frames'] == 41
    client.commit(cursor)
    assert client.prepare(rollout, {'images':[]})[0]['images'] == []
    # Changing subtask / control source has no influence on episode history.
    rollout.stage = {'goal':'new goal'}
    rollout.source = 'gpt_eef'
    for step in range(501, 526):
        archive.capture(step, np.zeros((16, 24, 3), dtype=np.uint8))
    rollout.tick=525
    packet, cursor = client.prepare(rollout, {'images':[]})
    assert [i['step_id'] for i in packet['images']] == [513,525]
    assert packet['gpt_episode_history']['cumulative_frames'] == 43


def test_no_future_episode_mixing_or_missing_frame_substitution(tmp_path):
    archive=make_archive(tmp_path, 50)
    assert len(archive.page(0, 25)['images']) == 3
    with pytest.raises(ValueError, match='Missing actual'):
        archive.capture(65, np.zeros((16,24,3), dtype=np.uint8))
    client=TeacherEpisodeHistory(tmp_path/'client')
    rollout=SimpleNamespace(episode='wrong_episode', tick=50,
        meta={'provides_teacher_episode_history':True},
        _rpc=lambda op,**kw: archive.page(kw['cursor'],kw['through_step']))
    with pytest.raises(ValueError,match='Wrong episode'):
        client.prepare(rollout, {})


def test_unknown_is_not_zero_and_contacts_not_grasp_labels(tmp_path, monkeypatch):
    # Missing sensors remain unavailable, including a fully closed controller command.
    env=SimpleNamespace(robot_manager=SimpleNamespace(robot_list=[],robot_key=[]))
    value=capture(env,episode_id='e',step_id=25,control_dt=.04)
    raw=json.loads(value)
    packet=teacher_packet({'grasp_feedback_json':value,'states':np.zeros(14)})
    assert packet['status']=='unavailable'
    assert 'left_finger_position_mm=[unknown, unknown]' in packet['text']
    assert 'contact_ge1N=[unknown, unknown]' in packet['text']
    assert 'grasp_success' not in packet
    np.savez_compressed(tmp_path/'observation.npz', grasp_feedback_json=value)
    with np.load(tmp_path/'observation.npz',allow_pickle=False) as arrays:
        assert str(arrays['grasp_feedback_json'])==value
    raw['contact']=dict(frame='world',units='N',sample_window='latest_physics_step', fingers=[
        dict(arm_name='left_arm',joint_name='joint7',status='available',
             normal_force_w=[.999,0,0],resultant_magnitude=.999),
        dict(arm_name='left_arm',joint_name='joint8',status='available',
             normal_force_w=[0,2,0],resultant_magnitude=2)])
    text=render_fingers(raw)
    assert 'left_normal_N=[0.999, 2]; contact_ge1N=[no, yes]' in text


def test_unavailable_old_server_and_corrupted_image(tmp_path):
    client=TeacherEpisodeHistory(tmp_path/'client')
    packet,_=client.prepare(SimpleNamespace(episode='e',meta={}),{})
    assert packet['gpt_episode_history']['status']=='unavailable'
    archive=make_archive(tmp_path,0)
    rollout=SimpleNamespace(episode='episode1',tick=0,
        meta={'provides_teacher_episode_history':True},
        _rpc=lambda op,**kw:archive.page(kw['cursor'],kw['through_step']))
    packet,_=client.prepare(rollout,{})
    from pathlib import Path
    Path(packet['images'][0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed after capture'):
        content_items(packet,images=True)


