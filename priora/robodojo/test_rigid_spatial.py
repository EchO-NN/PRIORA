"""Rigid geometry contracts; numerical fixtures without simulator or private poses."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from .camera_intrinsics import descriptor, key as intrinsics_key
from .io import InputError
from .skill.rigid_spatial import (ARMS, CAMERAS, RigidSpatial, VERSION, inverse,
                                   point, rotation, transform)


def pose(position, angle=0., opening=.5):
    return dict(position=position, quaternion_wxyz=[float(np.cos(angle/2)), 0., 0., float(np.sin(angle/2))],
                gripper_opening_command=opening)


def matrix(position=(0., 0., 0.), angle=0.):
    t = np.eye(4)
    t[:3, :3] = rotation(pose([0, 0, 0], angle)['quaternion_wxyz'])
    t[:3, 3] = position
    return t


class Episode:
    def __init__(self, root):
        self.root = root
        self.rollout = SimpleNamespace(episode='episode-one', tick=0)
        self.spatial = RigidSpatial(root)
        self.k = np.array([[120., 0, 64], [0, 100., 64], [0, 0, 1]])
        self.mount = matrix([.02, -.01, .03], .25)
        self.packet = None

    def observation(self, step=0, *, left_angle=0., right_angle=0., opening=.5):
        poses = dict(left=pose([.1+step*.001, -.1, .2], left_angle, opening),
                     right=pose([-.1, .1, .3], right_angle, opening))
        cameras = {}
        for camera in CAMERAS:
            if camera == 'cam_high':
                t, mount, parent = matrix([0, 0, -.5], -.12), None, 'environment'
            else:
                arm = 'left' if camera == 'cam_left_wrist' else 'right'
                p = poses[arm]
                t = matrix(p['position'], left_angle if arm == 'left' else right_angle) @ self.mount
                mount, parent = self.mount.tolist(), arm+'_eef'
            cameras[camera] = dict(T_environment_camera=t.tolist(), T_camera_environment=inverse(t).tolist(),
                T_eef_camera=mount, parent_frame=parent, source='fixed known mechanical mounting')
        directory = self.root/'observations'/f'{step:03}'
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for camera in CAMERAS:
            arrays[camera] = np.zeros((128, 128, 3), dtype=np.uint8)
            arrays[camera+'_depth_m'] = np.full((128, 128), 1., dtype=np.float32)
            arrays[intrinsics_key(camera)] = self.k
        self.arrays = directory/'observation.npz'
        np.savez_compressed(self.arrays, **arrays)
        self.rollout.tick = step
        self.rollout.observation_path = directory/'observation.json'
        self.packet = dict(episode_id=self.rollout.episode, step_id=step,
            observation_path=str(self.rollout.observation_path), arrays_path=str(self.arrays),
            current_eef=poses, camera_intrinsics={c: descriptor(self.k, [128, 128]) for c in CAMERAS},
            camera_geometry=dict(version='rigid_camera_geometry_v1', episode_id=self.rollout.episode,
                step_id=step, rig_id='known-fixture-rig', cameras=cameras))
        self.save()
        return self.packet

    def save(self):
        self.rollout.observation_path.write_text(json.dumps(self.packet))

    def query(self, camera='cam_high', arm='right', pixel=None, **extra):
        args = dict(observation_path=str(self.rollout.observation_path), camera=camera,
                    arm=arm, landmark='gripper_housing', pixel=[] if pixel is None else pixel)
        args.update(extra)
        return self.spatial.query(self.rollout, **args)

    def record(self, camera='cam_high', arm='right', pixel=None, **extra):
        args = dict(observation_path=str(self.rollout.observation_path), camera=camera,
                    arm=arm, landmark='gripper_housing', pixel=[64, 64] if pixel is None else pixel)
        args.update(extra)
        return self.spatial.record(self.rollout, **args)

    def change_arrays(self, callback):
        with np.load(self.arrays, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        callback(arrays)
        np.savez_compressed(self.arrays, **arrays)

    def dense(self):
        rows, columns = np.mgrid[:128, :128]
        rays = np.stack(((columns-64)/120, (rows-64)/100, np.ones_like(rows)), axis=-1)
        descriptors = {}
        for camera in CAMERAS:
            t = np.asarray(self.packet['camera_geometry']['cameras'][camera]['T_environment_camera'])
            xyz = (rays @ t[:3, :3].T + t[:3, 3]).astype(np.float32)
            path = self.arrays.parent/(camera+'_xyz.npz')
            identity = dict(version='dense_rgbd_xyz_v1', camera=camera,
                episode_id=self.packet['episode_id'], step_id=self.packet['step_id'],
                rig_id=self.packet['camera_geometry']['rig_id'])
            np.savez_compressed(path, xyz_environment=xyz, valid=np.ones((128, 128), dtype=bool),
                                K=self.k, T_environment_camera=t, **identity)
            descriptors[camera] = dict(identity, path=str(path), xyz_key='xyz_environment', valid_key='valid',
                shape=[128, 128, 3], original_size_wh=[128, 128], frame='environment_origin',
                units='meters', dtype='float32', depth_range_m=[.1, 10.],
                valid_count=128*128, invalid_count=0)
        self.packet['dense_geometry'] = descriptors
        self.save()
        return descriptors

    def change_dense(self, callback, camera='cam_high'):
        path = self.packet['dense_geometry'][camera]['path']
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        callback(arrays)
        np.savez_compressed(path, **arrays)


@pytest.mark.parametrize('camera', CAMERAS)
def test_visible_surface_backprojection_and_both_eef_frames(tmp_path, camera):
    env = Episode(tmp_path)
    packet = env.observation(left_angle=.4, right_angle=-.2)
    before = copy.deepcopy(packet)
    result = env.query(camera=camera, pixel=[70.2, 59.7])
    camera_point = np.linalg.solve(env.k, [70, 60, 1])
    expected = point(np.array(packet['camera_geometry']['cameras'][camera]['T_environment_camera']), camera_point)
    assert result['kind'] == 'visible_surface_point' and result['usable']
    assert result['sampled_pixel'] == [70, 60]
    np.testing.assert_allclose(result['environment_position_m'], expected)
    for arm, angle in [('left', .4), ('right', -.2)]:
        t = matrix(packet['current_eef'][arm]['position'], angle)
        np.testing.assert_allclose(result['eef_positions_m'][arm], point(inverse(t), expected))
    assert 'not a grasp' in result['warning']
    assert env.rollout.tick == 0 and env.packet == before
    assert 'reference_equivalent_environment_position' not in result


def test_auto_summary_updates_wrist_with_measured_pose_without_refitting(tmp_path):
    env = Episode(tmp_path)
    first = env.spatial.summary(env.observation())
    frozen = copy.deepcopy(env.spatial.state['calibration'])
    last = env.spatial.summary(env.observation(30, left_angle=1., right_angle=-.4))
    assert first['version'] == VERSION
    assert first['cameras']['cam_high']['T_environment_camera'] == last['cameras']['cam_high']['T_environment_camera']
    assert first['cameras']['cam_left_wrist']['T_environment_camera'] != last['cameras']['cam_left_wrist']['T_environment_camera']
    assert env.spatial.state['calibration'] == frozen
    assert 'candidate' not in env.spatial.state and 'maps' not in env.spatial.state
    for camera in CAMERAS:
        assert set(last['cameras'][camera]['measured_eef_origin_projections']) == set(ARMS)
    assert RigidSpatial(tmp_path).state == env.spatial.state


def test_empty_pixel_projects_origin_even_without_correspondences(tmp_path):
    env = Episode(tmp_path); env.observation()
    result = env.query()
    assert result['kind'] == 'measured_eef_origin_projection'
    assert result['usable']
    assert 'not a visible fingertip' in result['warning']
    assert result['visibility'].startswith('unknown')
    assert 'registered_landmark_projection' not in result
    assert result['environment_position_m'] == env.packet['current_eef']['right']['position']


def test_registered_rigid_surface_is_eef_local_and_opening_guarded(tmp_path):
    env = Episode(tmp_path); env.observation()
    recorded = env.record()
    item = recorded['recorded_landmark']
    assert item['K'] == env.k.tolist()
    assert item['intrinsics_source']
    np.testing.assert_allclose(item['local_eef_position_m'], [.1, -.1, .2])
    with pytest.raises(InputError, match='already recorded'):
        env.record()
    env.observation(30, right_angle=.7)
    result = env.query()
    projection = result['registered_landmark_projection']
    expected = point(matrix(env.packet['current_eef']['right']['position'], .7), np.array(item['local_eef_position_m']))
    np.testing.assert_allclose(projection['environment_position_m'], expected)
    assert projection['recorded_at_step'] == 0
    env.observation(60, right_angle=.7, opening=.6)
    assert not env.query()['registered_landmark_projection']['usable']
    events = list((tmp_path/'spatial').glob('event_*.json'))
    assert len(events) == 1 and json.loads(events[0].read_text())['kind'] == 'rigid_surface_registration'


@pytest.mark.parametrize('change,message', [
    (lambda p: p.pop('camera_geometry'), 'required'),
    (lambda p: p['camera_geometry'].update(step_id=99), 'stale'),
    (lambda p: p['camera_geometry'].update(episode_id='other'), 'mismatch'),
    (lambda p: p['camera_geometry']['cameras'].pop('cam_right_wrist'), 'All three'),
    (lambda p: p['camera_geometry']['cameras']['cam_left_wrist'].update(parent_frame='right_eef'), 'incorrect'),
    (lambda p: p['camera_geometry']['cameras']['cam_left_wrist']['T_environment_camera'][0].__setitem__(3, 9), 'stale'),
    (lambda p: p['camera_geometry']['cameras']['cam_high']['T_environment_camera'][0].__setitem__(0, 2), 'proper rigid'),
    (lambda p: p['camera_intrinsics']['cam_high'].update(K=[[0]*3]*3), 'intrinsics'),
    (lambda p: p['current_eef']['left'].update(quaternion_wxyz=[0, 0, 0, 0]), 'quaternion'),
])
def test_reject_missing_stale_or_invalid_geometry(tmp_path, change, message):
    env = Episode(tmp_path); env.observation()
    change(env.packet); env.save()
    with pytest.raises(InputError, match=message):
        env.query(pixel=[64, 64])
    assert env.spatial.state['calibration'] is None


@pytest.mark.parametrize('kind', ['rig_id', 'head', 'wrist_mount', 'intrinsics'])
def test_calibration_is_frozen_within_episode(tmp_path, kind):
    env = Episode(tmp_path); env.spatial.summary(env.observation())
    frozen = copy.deepcopy(env.spatial.state['calibration'])
    env.observation(30)
    if kind == 'rig_id':
        env.packet['camera_geometry']['rig_id'] = 'changed'
    elif kind == 'intrinsics':
        env.packet['camera_intrinsics']['cam_high']['K'][0][0] += 1
    else:
        camera = 'cam_high' if kind == 'head' else 'cam_left_wrist'
        d = env.packet['camera_geometry']['cameras'][camera]
        d['T_environment_camera'][0][3] += .01
        d['T_camera_environment'] = inverse(np.array(d['T_environment_camera'])).tolist()
        if kind == 'wrist_mount':
            d['T_eef_camera'][0][3] += .01
    env.save()
    with pytest.raises(InputError, match='changed within'):
        env.query()
    assert env.spatial.state['calibration'] == frozen


def test_latest_path_and_episode_guards(tmp_path):
    env = Episode(tmp_path); env.observation()
    old = str(env.rollout.observation_path)
    env.observation(30)
    with pytest.raises(InputError, match='latest'):
        env.query(observation_path=old)
    env.rollout.episode = 'other'
    with pytest.raises(InputError, match='mismatch'):
        env.query()
    env.rollout.episode = 'episode-one'
    env.packet['arrays_path'] = str(tmp_path/'elsewhere.npz'); env.save()
    with pytest.raises(InputError, match='current observation'):
        env.query()


def test_summary_rejects_older_or_changed_same_step_observations(tmp_path):
    env = Episode(tmp_path)
    original = env.observation()
    env.spatial.summary(original)
    changed = copy.deepcopy(original)
    changed['current_eef']['right']['gripper_opening_command'] = .9
    with pytest.raises(InputError, match='without a new observation'):
        env.spatial.summary(changed)
    env.spatial.summary(env.observation(30))
    with pytest.raises(InputError, match='Stale observation'):
        env.spatial.summary(original)
    other = copy.deepcopy(env.packet)
    other['episode_id'] = other['camera_geometry']['episode_id'] = 'other'
    with pytest.raises(InputError, match='cross episodes'):
        env.spatial.summary(other)


@pytest.mark.parametrize('kind,message', [('nan', 'Invalid/out-of-range'), ('too_near', 'Invalid/out-of-range'),
    ('too_far', 'Invalid/out-of-range'), ('edge', 'discontinuity'), ('missing', 'Raw aligned'),
    ('shape', 'geometry mismatch'), ('k', 'intrinsics disagree')])
def test_depth_validity_and_archived_calibration(tmp_path, kind, message):
    env = Episode(tmp_path); env.observation()
    def change(a):
        name = 'cam_high_depth_m'
        if kind in ('nan', 'too_near', 'too_far', 'edge'):
            a[name][65, 65] = dict(nan=np.nan, too_near=.09, too_far=10.1, edge=1.1)[kind]
        elif kind == 'missing': a.pop(name)
        elif kind == 'shape': a[name] = a[name][:-1]
        else: a[intrinsics_key('cam_high')][0, 0] += 1
    env.change_arrays(change)
    with pytest.raises(InputError, match=message):
        env.query(pixel=[64, 64])


@pytest.mark.parametrize('pixel', [[0, 64], [127, 64], [64, float('nan')], [1], [64, 64, 64], '64,64'])
def test_pixel_input_contract(tmp_path, pixel):
    env = Episode(tmp_path); env.observation()
    with pytest.raises(InputError):
        env.query(pixel=pixel)


def test_no_fallback_or_state_cross_version(tmp_path):
    env = Episode(tmp_path); env.spatial.summary(env.observation())
    state_path = tmp_path/'spatial/state.json'
    state = json.loads(state_path.read_text()); state['version'] = 'online_spatial_v2'
    state_path.write_text(json.dumps(state))
    with pytest.raises(InputError, match='fresh episode'):
        RigidSpatial(tmp_path)


def test_rigid_transform_rejects_reflection_and_bad_homogeneous_row():
    a = np.eye(4); a[0, 0] = -1
    with pytest.raises(InputError, match='proper rigid'):
        transform(a, 'test')
    a = np.eye(4); a[3, 3] = 2
    with pytest.raises(InputError, match='proper rigid'):
        transform(a, 'test')


@pytest.mark.parametrize('camera', CAMERAS)
def test_dense_pixel_uses_cached_matrix_and_exposes_only_descriptors(tmp_path, camera):
    env = Episode(tmp_path); env.observation(left_angle=.3, right_angle=-.2)
    baseline = env.query(camera=camera, pixel=[70, 60])
    assert baseline['geometry_source']['kind'] == 'raw_depth_backprojection'
    desc = env.dense()
    # A sub-tolerance float32 perturbation distinguishes reading the cache
    # from merely recomputing while claiming that the cache was used.
    env.change_dense(lambda d: d['xyz_environment'][60, 70].__iadd__(np.array([4e-6, 0, 0])), camera)
    result = env.query(camera=camera, pixel=[70, 60])
    assert result['geometry_source']['kind'] == 'cached_dense_xyz'
    assert result['geometry_source']['path'] == desc[camera]['path']
    assert result['geometry_source']['matrix_index_vu'] == [60, 70]
    assert result['environment_position_m'][0] != baseline['environment_position_m'][0]
    np.testing.assert_allclose(result['environment_position_m'], baseline['environment_position_m'], atol=1e-5, rtol=0)
    for arm in ARMS:
        np.testing.assert_allclose(result['eef_positions_m'][arm], baseline['eef_positions_m'][arm], atol=1e-5, rtol=0)
    summary = env.spatial.summary(env.packet)
    assert summary['dense_geometry'] == desc
    assert 'xyz_environment' not in summary['dense_geometry']['cam_high']


@pytest.mark.parametrize('kind,message', [
    ('archive_step', 'identity mismatch'), ('archive_episode', 'identity mismatch'),
    ('archive_camera', 'identity mismatch'), ('archive_rig', 'identity mismatch'),
    ('archive_version', 'identity mismatch'), ('K', 'calibration disagrees'),
    ('T', 'calibration disagrees'), ('invalid', 'surface pixel is invalid'),
    ('nan', 'surface pixel is invalid'), ('wrong_point', 'disagrees with current raw depth'),
    ('shape', 'shape or dtype'), ('dtype', 'shape or dtype'), ('mask_shape', 'shape or dtype'),
])
def test_dense_cache_rejects_stale_invalid_or_inconsistent_data(tmp_path, kind, message):
    env = Episode(tmp_path); env.observation(); env.dense()
    def change(a):
        if kind.startswith('archive_'):
            name = dict(archive_step='step_id', archive_episode='episode_id', archive_camera='camera',
                        archive_rig='rig_id', archive_version='version')[kind]
            a[name] = np.array(30 if name == 'step_id' else 'changed')
        elif kind == 'K': a['K'][0, 0] += 1
        elif kind == 'T': a['T_environment_camera'][0, 3] += .1
        elif kind == 'invalid': a['valid'][64, 64] = False
        elif kind == 'nan': a['xyz_environment'][64, 64, 0] = np.nan
        elif kind == 'wrong_point': a['xyz_environment'][64, 64, 0] += .01
        elif kind == 'shape': a['xyz_environment'] = a['xyz_environment'][:-1]
        elif kind == 'dtype': a['xyz_environment'] = a['xyz_environment'].astype(np.float64)
        elif kind == 'mask_shape': a['valid'] = a['valid'][:-1]
    env.change_dense(change)
    with pytest.raises(InputError, match=message):
        env.query(pixel=[64, 64])


def test_dense_cache_still_requires_raw_depth_neighborhood_checks(tmp_path):
    env = Episode(tmp_path); env.observation(); env.dense()
    env.change_arrays(lambda a: a['cam_high_depth_m'].__setitem__((65, 65), 2.))
    with pytest.raises(InputError, match='discontinuity'):
        env.query(pixel=[64, 64])


def test_dense_archive_path_and_descriptor_identity_guards(tmp_path):
    env = Episode(tmp_path); env.observation(); desc = env.dense()
    desc['cam_high']['path'] = str(tmp_path/'outside.npz'); env.save()
    with pytest.raises(InputError, match='current observation directory'):
        env.query(pixel=[64, 64])
    desc['cam_high']['path'] = desc['cam_left_wrist']['path']; env.save()
    with pytest.raises(InputError, match='identity mismatch'):
        env.query(pixel=[64, 64])
    desc['cam_high']['step_id'] = 100; env.save()
    with pytest.raises(InputError, match='descriptor identity'):
        env.query(pixel=[64, 64])


def test_declared_missing_dense_cache_is_not_silently_recomputed(tmp_path):
    env = Episode(tmp_path); env.observation(); env.dense()
    env.packet['dense_geometry'].pop('cam_high'); env.save()
    with pytest.raises(InputError, match='declared dense geometry is missing'):
        env.query(pixel=[64, 64])


def test_real_dense_writer_and_rigid_query_share_archive_contract(tmp_path):
    from .dense_geometry import write_dense_geometry
    env = Episode(tmp_path); env.observation(left_angle=.3, right_angle=-.2)
    with np.load(env.arrays, allow_pickle=False) as data:
        obs = {key: data[key] for key in data.files}
    env.packet['dense_geometry'] = write_dense_geometry(env.arrays.parent, obs,
        env.packet['camera_intrinsics'], env.packet['camera_geometry'])
    env.save()
    for camera in CAMERAS:
        result = env.query(camera=camera, pixel=[64, 64])
        assert result['geometry_source']['kind'] == 'cached_dense_xyz'
        assert result['geometry_source']['sample_checked_against_raw_depth']
    assert len(env.spatial.summary(env.packet)['dense_geometry']) == 3
