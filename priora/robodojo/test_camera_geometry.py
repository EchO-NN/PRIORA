"""Rigid geometry contracts, including independent convention checks."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from .camera_geometry import (VERSION, OBSERVATION_KEYS, USD_FROM_OPTICAL,
    backproject, capture, declared_pose, fixed_urdf_transform,
    from_environment_config, from_observation, invert_transform,
    pose_matrix, project, transform_points, validate_transform)


def rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def make_environment(tmp_path):
    urdf = tmp_path / 'X5A.urdf'
    urdf.write_text('''<robot name="x5"><link name="link6"/><link name="camera_base"/><link name="camera"/>
      <joint name="hand_to_camera_mount" type="fixed"><parent link="link6"/><child link="camera_base"/>
        <origin xyz="0.057 0 0" rpy="0 0 -3.1416"/></joint>
      <joint name="camera_joint" type="fixed"><parent link="camera_base"/><child link="camera"/>
        <origin xyz="-0.0275 0 0.05" rpy="0 0.3491 3.141592653589793"/></joint></robot>''')
    wrist = dict(name='cam_wrist', link='camera', pos=[0, 0, .001], ori=[0, -80, -90], mesh='pinhole')
    robots = [SimpleNamespace(type='target', arm_name=arm+'_arm', camera=[copy.deepcopy(wrist)],
                              urdf_path=urdf, ee_link_name='link6') for arm in ('left', 'right')]
    configs = {'cam_head': {'camera': dict(pos=[0, -.41, 1.308], ori=[30, 0, 0], mesh='pinhole')}}
    for i, arm in enumerate(('left', 'right')):
        configs['cam_'+arm+'_wrist'] = {'camera': dict(wrist, mount_link=f'robot{i}/camera')}
    class NoWorldReads:
        def get_world_pose(self, *a, **kw):
            raise AssertionError('Camera world pose must not be used')
        def get_local_pose(self, *a, **kw):
            raise AssertionError('Runtime local pose must not be used')
    manager = SimpleNamespace(camera_config=configs, random_config={},
                              cameras=[[NoWorldReads() for _ in range(3)]],
                              cameras_xform=[[NoWorldReads() for _ in range(3)]])
    env = SimpleNamespace(camera_manager=manager,
        robot_manager=SimpleNamespace(robot_list=robots, use_scene_cfg=[True, True]))
    visual = dict(pinhole=dict(position=[0, 0, 0], orientation=[1, 0, 0, 0]))
    return env, visual


def calibration(tmp_path):
    env, visual = make_environment(tmp_path)
    return from_environment_config(env, visual_templates=visual)


def test_config_intrinsic_xyz_differs_from_urdf_extrinsic(tmp_path):
    angles = [.23, -.65, .71]
    t = declared_pose(dict(pos=[.1, .2, .3], ori=np.rad2deg(angles)))
    expected = rx(angles[0]) @ ry(angles[1]) @ rz(angles[2])
    np.testing.assert_allclose(t[:3, :3], expected, atol=1e-15)
    assert not np.allclose(expected, rz(angles[2]) @ ry(angles[1]) @ rx(angles[0]))
    urdf = tmp_path / 'mixed.urdf'
    urdf.write_text('<robot><link name="eef"/><link name="cam"/>'
      '<joint name="fixed" type="fixed"><parent link="eef"/><child link="cam"/>'
      '<origin xyz="1 2 3" rpy=".23 -.65 .71"/></joint></robot>')
    fixed, _ = fixed_urdf_transform(urdf, 'eef', 'cam')
    np.testing.assert_allclose(fixed[:3, :3], rz(angles[2]) @ ry(angles[1]) @ rx(angles[0]), atol=1e-15)


def test_actual_x5_chain_config_and_optical_flip_without_pose_reads(tmp_path):
    value = calibration(tmp_path)
    wrist = value['cameras']['cam_left_wrist']
    first = np.eye(4); first[:3, :3] = rz(-3.1416); first[:3, 3] = [.057, 0, 0]
    second = np.eye(4); second[:3, :3] = rz(np.pi) @ ry(.3491); second[:3, 3] = [-.0275, 0, .05]
    mount = np.eye(4); mount[:3, :3] = ry(np.deg2rad(-80)) @ rz(np.deg2rad(-90)); mount[:3, 3] = [0, 0, .001]
    expected = first @ second @ mount @ np.diag([1, -1, -1, 1])
    np.testing.assert_allclose(wrist['fixed_transform'], expected, atol=1e-15)
    assert wrist['parent_frame'] == 'left_eef'
    assert wrist['source']['fixed_joints'] == ['hand_to_camera_mount', 'camera_joint']
    assert wrist['source']['runtime_camera_world_pose_used'] is False
    head = np.asarray(value['cameras']['cam_high']['fixed_transform'])
    np.testing.assert_allclose(head[:3, :3], rx(np.pi/6) @ np.diag([1, -1, -1]), atol=1e-15)
    # A forward optical point must go DOWN and away from the fixed main camera.
    np.testing.assert_allclose(transform_points(head, [0, 0, 1]), [0, .09, 1.308-np.sqrt(3)/2])


def test_poses_update_wrist_geometry_only_and_round_trip(tmp_path):
    value = calibration(tmp_path)
    p = np.array([[.2, -.3, .8], [-.2, .4, 1.1]])
    q = Rotation.from_euler('xyz', [[.4, -.2, .8], [-.5, .7, -.1]]).as_quat()[:, [3, 0, 1, 2]]
    arrays, desc = capture(value, p, q)
    assert set(arrays) == set(OBSERVATION_KEYS)
    original = copy.deepcopy(arrays)
    p[0] += [.3, .1, -.2]
    q[0] = Rotation.from_euler('xyz', [1., .4, -.7]).as_quat()[[3, 0, 1, 2]]
    new, _ = capture(value, p, q)
    np.testing.assert_array_equal(new['cam_high_T_environment_camera'], original['cam_high_T_environment_camera'])
    np.testing.assert_array_equal(new['cam_right_wrist_T_environment_camera'], original['cam_right_wrist_T_environment_camera'])
    np.testing.assert_array_equal(new['cam_left_wrist_T_eef_camera'], original['cam_left_wrist_T_eef_camera'])
    assert not np.allclose(new['cam_left_wrist_T_environment_camera'], original['cam_left_wrist_T_environment_camera'])
    k = np.array([[288.1, 0, 320], [0, 336.4, 240], [0, 0, 1]])
    pc = backproject(k, [412.25, 108.75], .723)
    t = arrays['cam_left_wrist_T_environment_camera']
    pe = transform_points(t, pc)
    np.testing.assert_allclose(transform_points(invert_transform(t), pe), pc)
    np.testing.assert_allclose(project(k, transform_points(invert_transform(t), pe)), [412.25, 108.75])
    np.testing.assert_allclose(t @ np.asarray(desc['cam_left_wrist']['T_camera_environment']), np.eye(4), atol=1e-15)


def test_observation_contract_detects_stale_poses_partial_arrays_and_tampered_mount(tmp_path):
    value = calibration(tmp_path)
    p, q = np.zeros((2, 3)), np.array([[1, 0, 0, 0]]*2)
    arrays, _ = capture(value, p, q)
    obs = dict(arrays, eef_positions=p.copy(), eef_quaternions_wxyz=q)
    packet = from_observation(obs, value, episode_id='a', step_id=7, required=True)
    assert packet['version'] == VERSION and packet['episode_id'] == 'a' and packet['step_id'] == 7
    assert packet['rig_id'] == value['rig_id']
    assert from_observation({}) == {}
    with pytest.raises(ValueError, match='All three'):
        from_observation({}, required=True)
    missing = dict(obs); del missing[OBSERVATION_KEYS[0]]
    with pytest.raises(ValueError, match='All three'):
        from_observation(missing, value)
    obs['eef_positions'][0, 0] += .01
    with pytest.raises(ValueError, match='stale'):
        from_observation(obs, value)
    bad = copy.deepcopy(value); bad['cameras']['cam_high']['fixed_transform'][0][3] += .01
    with pytest.raises(ValueError, match='changed'):
        capture(bad, p, q)


@pytest.mark.parametrize('mutation,match', [
    (lambda env, vis: env.camera_manager.random_config.update(common=dict(enabled=True, pos=dict(min=[0,0,0], max=[.01,0,0]))), 'randomized'),
    (lambda env, vis: env.camera_manager.random_config.update(common=dict(action_noise=dict(pos=dict(min=[0,0,0], max=[.01,0,0])))), 'randomized'),
    (lambda env, vis: env.camera_manager.camera_config['cam_head']['camera'].update(mount_link='robot0/camera'), 'Main camera'),
    (lambda env, vis: env.camera_manager.camera_config['cam_left_wrist']['camera'].update(mount_link='robot1/camera'), 'parent disagree'),
    (lambda env, vis: env.camera_manager.camera_config['cam_left_wrist']['camera'].update(pos=[1,0,0]), 'differs'),
    (lambda env, vis: vis['pinhole'].update(asset_path='unknown.usd'), 'Asset-backed'),
])
def test_unsupported_or_miswired_mounts_fail_closed(tmp_path, mutation, match):
    env, visual = make_environment(tmp_path)
    mutation(env, visual)
    with pytest.raises(ValueError, match=match):
        from_environment_config(env, visual_templates=visual)


def test_moving_urdf_mount_fails_closed(tmp_path):
    env, visual = make_environment(tmp_path)
    path = env.robot_manager.robot_list[0].urdf_path
    path.write_text(path.read_text().replace('type="fixed"', 'type="revolute"', 1))
    with pytest.raises(ValueError, match='moving joint'):
        from_environment_config(env, visual_templates=visual)


@pytest.mark.parametrize('bad', [np.diag([-1,1,1,1]), np.diag([2,1,1,1]), np.ones((4,4)), np.full((4,4), np.nan)])
def test_nonrigid_invalid_transforms_rejected(bad):
    with pytest.raises(ValueError, match='proper rigid'):
        validate_transform(bad)


def test_positive_depth_and_quaternion_contracts():
    for z in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError, match='positive optical'):
            backproject(np.eye(3), [1,2], z)
        with pytest.raises(ValueError, match='front'):
            project(np.eye(3), [1,2,z])
    with pytest.raises(ValueError, match='unit wxyz'):
        pose_matrix([0,0,0], [0,0,0,0])
    np.testing.assert_array_equal(USD_FROM_OPTICAL @ USD_FROM_OPTICAL, np.eye(4))
