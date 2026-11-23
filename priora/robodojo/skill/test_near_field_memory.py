"""Local RGBD-history contracts; no RPC, model call or simulator needed."""
import copy
import json

import numpy as np
import pytest

from ..io import InputError
from ..test_rigid_spatial import CAMERAS, Episode
from .near_field_memory import MAX_SNAPSHOTS, NearFieldMemory


def test_summary_counts_79mm_without_claiming_rgb_invisible(tmp_path):
    env = Episode(tmp_path)
    packet = env.observation()
    def change(data):
        d = data['cam_right_wrist_depth_m']
        d[10:13, 10:13] = .079
        d[0, 0], d[0, 1], d[0, 2], d[0, 3] = 0, np.nan, np.inf, 11
    env.change_arrays(change)
    memory = NearFieldMemory(tmp_path)
    result = memory.summary(packet)
    counts = result['depth_counts']['cam_right_wrist']
    assert counts == dict(total=16384, valid=16371, below_min_positive=9,
                          above_max=1, nonpositive_or_nonfinite=3)
    assert 'RGB may still show' in result['warning']
    assert result['configured_depth_range_m'] == [.1, 10.]
    assert result['snapshots'][0]['current_measurement'] is False
    with pytest.raises(InputError, match='configured 0.1..10'):
        memory.remember(packet, camera='cam_right_wrist', label='cloth_corner', pixel=[11, 11])
    assert not list(memory.patches.iterdir())


def test_snapshot_references_are_bounded_and_same_step_is_idempotent(tmp_path):
    env, memory = Episode(tmp_path), NearFieldMemory(tmp_path)
    packet = env.observation()
    before = copy.deepcopy(packet)
    first = memory.summary(packet)
    mtime = memory.path.stat().st_mtime_ns
    assert first == memory.summary(packet)
    assert memory.path.stat().st_mtime_ns == mtime
    assert packet == before
    for step in range(1, MAX_SNAPSHOTS+3):
        result = memory.summary(env.observation(step))
    assert result['snapshot_count'] == MAX_SNAPSHOTS
    assert len(result['snapshots']) == 3
    assert result['snapshots'][-2]['age_control_steps'] == 1
    assert not list(memory.patches.iterdir())
    assert len(NearFieldMemory(tmp_path).state['snapshots']) == MAX_SNAPSHOTS


def test_clipped_patch_mixed_mask_recomputed_from_raw_depth(tmp_path):
    env = Episode(tmp_path)
    packet = env.observation()
    env.dense()
    # The archive reference may exist, but patch XYZ must derive from raw depth.
    env.change_dense(lambda d: d['xyz_environment'].fill(999))
    def change(data):
        d = data['cam_high_depth_m']
        d[0, 0], d[0, 1], d[0, 2] = .079, np.nan, 11
    env.change_arrays(change)
    memory = NearFieldMemory(tmp_path)
    result = memory.remember(packet, camera='cam_high', label='hem', pixel=[2, 2], radius=4)
    assert result['pixel_bounds_xyxy'] == [0, 0, 7, 7]
    assert result['valid_count'] == 46 and result['invalid_count'] == 3
    assert result['source_dense_geometry']['path'] == packet['dense_geometry']['cam_high']['path']
    with np.load(result['archive_path'], allow_pickle=False) as data:
        assert data['xyz_environment'].shape == (7, 7, 3)
        assert data['valid'].dtype == np.bool_
        assert np.isnan(data['xyz_environment'][0, :3]).all()
        assert data['raw_depth_m'][0, 0] == pytest.approx(.079)
        assert np.max(data['xyz_environment'][data['valid']]) < 999
        k, t = data['K'], data['T_environment_camera']
        camera_point = np.linalg.solve(k, [2, 2, 1])
        expected = t[:3, :3] @ camera_point + t[:3, 3]
        np.testing.assert_allclose(data['xyz_environment'][2, 2], expected, atol=1e-6)
    from PIL import Image
    assert Image.open(result['image_path']).size == (7, 7)


def test_recall_after_wrist_moves_keeps_historical_world_xyz_and_original_matrix(tmp_path):
    env, memory = Episode(tmp_path), NearFieldMemory(tmp_path)
    old = env.observation()
    result = memory.remember(old, camera='cam_left_wrist', label='hem', pixel=[64, 64], radius=32)
    assert result['patch_shape'] == [65, 65, 3]
    assert result['historical_only'] and result['require_reobservation_before_contact']
    old_xyz, old_t = result['observed_center_environment_position_m'], result['T_environment_camera']
    new = env.observation(30, left_angle=1.2)
    recalled = NearFieldMemory(tmp_path).recall(new, memory_id=result['memory_id'])
    assert recalled['observed_center_environment_position_m'] == old_xyz
    assert recalled['T_environment_camera'] == old_t
    assert recalled['T_environment_camera'] != new['camera_geometry']['cameras']['cam_left_wrist']['T_environment_camera']
    assert recalled['age_control_steps'] == 30 and recalled['stale_after_controls']
    assert recalled['current_measurement'] is False and recalled['object_tracking'] is False
    assert 'current_position_m' not in recalled and 'grasp_target' not in recalled


def test_immutable_patch_repeat_and_label_identity(tmp_path):
    env, memory = Episode(tmp_path), NearFieldMemory(tmp_path)
    packet = env.observation()
    args = dict(camera='cam_high', label='hem', pixel=[64, 64])
    first = memory.remember(packet, **args)
    index_mtime = memory.path.stat().st_mtime_ns
    assert first == memory.remember(packet, **args)
    assert memory.path.stat().st_mtime_ns == index_mtime
    second = memory.remember(packet, **dict(args, label='hem_other'))
    assert second['memory_id'] != first['memory_id']
    assert memory.recall(packet)['memory_count'] == 2


@pytest.mark.parametrize('changed', ['episode', 'rig', 'backwards', 'same_step_matrix', 'same_step_arrays'])
def test_reject_changed_identity_and_stale_packets(tmp_path, changed):
    env, memory = Episode(tmp_path), NearFieldMemory(tmp_path)
    packet = env.observation(10)
    memory.summary(packet)
    packet = copy.deepcopy(packet)
    if changed == 'episode':
        packet['episode_id'] = packet['camera_geometry']['episode_id'] = 'other'
    elif changed == 'rig':
        packet['camera_geometry']['rig_id'] = 'other'
    elif changed == 'backwards':
        packet['step_id'] = packet['camera_geometry']['step_id'] = 9
    elif changed == 'same_step_matrix':
        packet['camera_geometry']['cameras']['cam_high']['T_environment_camera'][0][3] += .01
    else:
        env.change_arrays(lambda data: data['cam_high_depth_m'].fill(.5))
    with pytest.raises(InputError):
        NearFieldMemory(tmp_path).summary(packet)


@pytest.mark.parametrize('args', [dict(radius=0), dict(radius=33), dict(radius=True),
    dict(label='../escape'), dict(pixel=[0, 64]), dict(pixel=[127, 64]),
    dict(pixel=[np.nan, 64]), dict(camera='unknown')])
def test_invalid_patch_arguments(tmp_path, args):
    env, memory = Episode(tmp_path), NearFieldMemory(tmp_path)
    packet = env.observation()
    kwargs = dict(camera='cam_high', label='hem', pixel=[64, 64])
    kwargs.update(args)
    with pytest.raises(InputError):
        memory.remember(packet, **kwargs)
    assert not list(memory.patches.iterdir())


def test_depth_discontinuity_rejected(tmp_path):
    env, memory = Episode(tmp_path), NearFieldMemory(tmp_path)
    packet = env.observation()
    env.change_arrays(lambda data: data['cam_high_depth_m'].__setitem__((64, 65), 1.05))
    with pytest.raises(InputError, match='discontinuity'):
        memory.remember(packet, camera='cam_high', label='hem', pixel=[64, 64])


def test_unsafe_source_and_unsafe_ids_rejected(tmp_path):
    env, memory = Episode(tmp_path), NearFieldMemory(tmp_path)
    packet = env.observation()
    for value in ('../outside', 'nf_'+32*'z', '/tmp/absolute'):
        with pytest.raises(InputError, match='canonical'):
            memory.recall(packet, memory_id=value)
    packet['arrays_path'] = str(tmp_path/'outside.npz')
    (tmp_path/'outside.npz').write_bytes(env.arrays.read_bytes())
    with pytest.raises(InputError, match='episode observation'):
        memory.summary(packet)


def test_tampered_archive_and_index_are_rejected(tmp_path):
    env, memory = Episode(tmp_path), NearFieldMemory(tmp_path)
    packet = env.observation()
    result = memory.remember(packet, camera='cam_high', label='hem', pixel=[64, 64])
    memory.state['memories'][0]['observed_center_environment_position_m'][0] += 1
    with pytest.raises(InputError, match='identity was altered'):
        memory.recall(packet, memory_id=result['memory_id'])
    memory = NearFieldMemory(tmp_path)
    from pathlib import Path
    Path(result['archive_path']).write_bytes(b'changed')
    with pytest.raises(InputError, match='modified'):
        memory.recall(packet, memory_id=result['memory_id'])


def test_rgb_validation_once_per_observation_and_summary_status(tmp_path, monkeypatch):
    from . import near_field_memory as module
    calls = []
    def compare(rgb, xyz_environment, valid, **kwargs):
        calls.append(kwargs)
        status = 'same_observation' if kwargs['source_step'] == kwargs['current_step'] else 'consistent'
        return dict(status=status, matched_fraction=.9)
    monkeypatch.setattr(module, 'compare_patch', compare)
    env, memory = Episode(tmp_path), NearFieldMemory(tmp_path)
    packet = env.observation()
    item = memory.remember(packet, camera='cam_left_wrist', label='hem', pixel=[64, 64])
    assert len(calls) == 1
    assert item['visual_status'] == 'same_observation'
    assert item['usable_for_planning_hint'] is True
    assert item['current_measurement'] is False
    memory.summary(packet)
    memory.recall(packet, memory_id=item['memory_id'])
    assert len(calls) == 1
    packet = env.observation(7, left_angle=.1)
    summary = memory.summary(packet)
    assert len(calls) == 2
    check = summary['memories'][0]['visual_check']
    assert check['status'] == 'consistent' and check['checked_at_step'] == 7
    assert check['matched_fraction'] == .9 and not check['object_tracking']
    assert check['require_reobservation_before_contact']
    np.testing.assert_allclose(calls[-1]['T_environment_camera'],
        packet['camera_geometry']['cameras']['cam_left_wrist']['T_environment_camera'])
    restarted = NearFieldMemory(tmp_path)
    assert restarted.recall(packet, memory_id=item['memory_id'])['usable_for_planning_hint']
    assert len(calls) == 2


def test_rgb_change_invalidation_is_sticky_and_archive_immutable(tmp_path, monkeypatch):
    from . import near_field_memory as module
    calls = []
    def compare(rgb, xyz_environment, valid, **kwargs):
        calls.append((kwargs['source_step'], kwargs['current_step']))
        return dict(status='same_observation' if kwargs['source_step'] == kwargs['current_step']
                    else 'invalidated_rgb_change', rgb_change_fraction=.9)
    monkeypatch.setattr(module, 'compare_patch', compare)
    env, memory = Episode(tmp_path), NearFieldMemory(tmp_path)
    packet = env.observation()
    item = memory.remember(packet, camera='cam_high', label='hem', pixel=[64, 64])
    from pathlib import Path
    original_archive = Path(item['archive_path']).read_bytes()
    packet = env.observation(4)
    recalled = memory.recall(packet, memory_id=item['memory_id'])
    assert recalled['invalidated'] and recalled['visual_status'] == 'invalidated_rgb_change'
    assert recalled['visual_check']['invalidated_at_step'] == 4
    assert recalled['usable_for_planning_hint'] is False
    assert recalled['archive_path'] == item['archive_path']
    assert recalled['current_measurement'] is False
    assert Path(item['archive_path']).read_bytes() == original_archive
    with np.load(item['archive_path'], allow_pickle=False) as data:
        assert 'visual_check' not in json.loads(data['metadata_json'].item())
    # Even a future identical-looking view must not revive retired geometry.
    def no_revive(*args, **kwargs):
        raise AssertionError('Invalidated patches must not be compared again')
    monkeypatch.setattr(module, 'compare_patch', no_revive)
    restarted = NearFieldMemory(tmp_path)
    packet = env.observation(8)
    later = restarted.recall(packet, memory_id=item['memory_id'])
    assert later['invalidated'] and not later['usable_for_planning_hint']
    assert later['age_control_steps'] == 8
    assert later['visual_check']['invalidated_at_step'] == 4
    assert len(calls) == 2
    # Fresh explicit observation creates a separate record, leaving the old one invalid.
    monkeypatch.setattr(module, 'compare_patch', compare)
    fresh = restarted.remember(packet, camera='cam_high', label='hem', pixel=[64, 64])
    assert fresh['memory_id'] != item['memory_id']
    assert fresh['visual_status'] == 'same_observation' and fresh['usable_for_planning_hint']
    assert restarted.recall(packet, memory_id=item['memory_id'])['invalidated']


def test_unverified_view_cannot_supply_planning_hint_but_can_be_rechecked(tmp_path, monkeypatch):
    from . import near_field_memory as module
    def compare(rgb, xyz_environment, valid, **kwargs):
        step = kwargs['current_step']
        return dict(status='same_observation' if step == 0 else 'unverified' if step == 5 else 'consistent',
                    reason='example occlusion' if step == 5 else 'enough matched pixels')
    monkeypatch.setattr(module, 'compare_patch', compare)
    env, memory = Episode(tmp_path), NearFieldMemory(tmp_path)
    item = memory.remember(env.observation(), camera='cam_high', label='hem', pixel=[64, 64])
    unavailable = memory.recall(env.observation(5), memory_id=item['memory_id'])
    assert unavailable['visual_status'] == 'unverified'
    assert not unavailable['usable_for_planning_hint'] and not unavailable['invalidated']
    visible = memory.recall(env.observation(10), memory_id=item['memory_id'])
    assert visible['visual_status'] == 'consistent' and visible['usable_for_planning_hint']
    assert visible['current_measurement'] is False and visible['require_reobservation_before_contact']


def test_rgb_check_index_is_bounded_with_patch_index(tmp_path, monkeypatch):
    from . import near_field_memory as module
    monkeypatch.setattr(module, 'MAX_PATCHES', 2)
    monkeypatch.setattr(module, 'compare_patch', lambda *args, **kw: dict(status='same_observation'))
    env, memory = Episode(tmp_path), NearFieldMemory(tmp_path)
    packet = env.observation()
    records = [memory.remember(packet, camera='cam_high', label='hem_'+str(n), pixel=[64, 64])
               for n in range(3)]
    assert len(memory.state['memories']) == len(memory.state['visual_checks']) == 2
    assert records[0]['memory_id'] not in memory.state['visual_checks']
    from pathlib import Path
    assert Path(records[0]['archive_path']).exists()
