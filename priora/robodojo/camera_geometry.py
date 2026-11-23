"""Declared rigid camera mounts composed with measured EEF poses.

T_A_B maps column vectors expressed in B into A. No runtime camera or object
world pose is read here. URDF joint origins use extrinsic xyz radians, whereas
RoboDojo camera ``ori`` uses intrinsic XYZ degrees (utils.rotations default).
The camera prim uses USD axes; optical points use ROS/OpenCV +X right,+Y down,
+Z forward, so the final fixed factor is diag(1,-1,-1,1).
"""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .camera_intrinsics import CAMERA_ALIASES, validate_k

VERSION = 'rigid_camera_geometry_v1'
USD_FROM_OPTICAL = np.diag([1., -1., -1., 1.])
ARMS = {'cam_left_wrist': ('left', 0), 'cam_right_wrist': ('right', 1)}
OBSERVATION_KEYS = tuple(c + '_T_environment_camera' for c in CAMERA_ALIASES) + tuple(
    c + '_T_eef_camera' for c in ARMS)


def validate_transform(value):
    t = np.asarray(value, dtype=np.float64)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-7, rtol=0)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1., atol=1e-7, rtol=0)):
        raise ValueError('Expected a finite proper rigid 4x4 transform')
    return t


def pose_matrix(position, quaternion_wxyz):
    position = np.asarray(position, dtype=np.float64)
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    if (position.shape != (3,) or not np.isfinite(position).all() or q.shape != (4,)
            or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-3):
        raise ValueError('Expected position meters and unit wxyz quaternion')
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    t[:3, 3] = position
    return validate_transform(t)


def invert_transform(value):
    t = validate_transform(value)
    inverse = np.eye(4)
    inverse[:3, :3] = t[:3, :3].T
    inverse[:3, 3] = -t[:3, :3].T @ t[:3, 3]
    return inverse


def transform_points(transform, points):
    t = validate_transform(transform)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
        raise ValueError('Expected finite Cartesian points with final dimension 3')
    return points @ t[:3, :3].T + t[:3, 3]


def backproject(k, pixel, depth):
    """One original-image (u,v) and optical-axis Z in meters, not ray distance."""
    pixel = np.asarray(pixel, dtype=np.float64)
    depth = float(depth)
    if pixel.shape != (2,) or not np.isfinite(pixel).all() or not np.isfinite(depth) or depth <= 0:
        raise ValueError('Backprojection requires a finite pixel and positive optical Z')
    return depth * np.linalg.solve(validate_k(k), np.r_[pixel, 1.])


def project(k, point_camera):
    p = np.asarray(point_camera, dtype=np.float64)
    if p.shape != (3,) or not np.isfinite(p).all() or p[2] <= 0:
        raise ValueError('Projection requires a point in front of the optical camera (Z > 0)')
    uvz = validate_k(k) @ p
    return uvz[:2] / uvz[2]


def _xyz(value, label):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError(f'{label}: expected 3 finite coordinates')
    return a


def declared_pose(config):
    """CameraManager.init_camera_pose: intrinsic XYZ degrees or wxyz."""
    position = config.get('pos')
    position = [0, 0, 1] if position is None else position
    orientation = config.get('ori')
    orientation = [0, 0, 0] if orientation is None else orientation
    a = np.asarray(orientation, dtype=np.float64)
    if a.shape == (3,):
        _xyz(a, 'camera orientation')
        xyzw = Rotation.from_euler('XYZ', a, degrees=True).as_quat()
        a = xyzw[[3, 0, 1, 2]]
    return pose_matrix(position, a)


def fixed_urdf_transform(path, parent, child):
    """Read only fixed joint origins from parent EEF link to mount link."""
    root = ET.parse(path).getroot()
    edges = {}
    for joint in root.findall('joint'):
        child_element = joint.find('child')
        if child_element is None or joint.find('parent') is None:
            raise ValueError('Malformed URDF joint')
        name = child_element.get('link')
        if name in edges:
            raise ValueError('URDF child link has multiple parents')
        edges[name] = joint
    links = {x.get('name') for x in root.findall('link')}
    if parent not in links or child not in links:
        raise ValueError('Requested EEF/mount link absent from URDF')
    chain, visited = [], set()
    while child != parent:
        if child in visited or child not in edges:
            raise ValueError('Camera mount is not a descendant of the requested EEF')
        visited.add(child)
        joint = edges[child]
        if joint.get('type') != 'fixed':
            raise ValueError('Camera mount contains a moving joint; fixed extrinsics invalid')
        origin = joint.find('origin')
        xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
        rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
        t = np.eye(4)
        t[:3, :3] = Rotation.from_euler('xyz', _xyz(rpy, 'URDF rpy')).as_matrix()
        t[:3, 3] = _xyz(xyz, 'URDF xyz')
        chain.insert(0, (joint.get('name'), t))
        child = joint.find('parent').get('link')
    result = np.eye(4)
    for _, t in chain:
        result = result @ t
    return validate_transform(result), [name for name, _ in chain]


def _no_pose_randomization(config, camera):
    # Reject unsupported time-varying mounts rather than presenting nominal
    # calibration as measured extrinsics. Config reads do not inspect a prim.
    config = config or {}
    for field in ('pos', 'ori', 'action_noise'):
        value = config.get(field)
        if value is None or (field != 'action_noise' and not config.get('enabled', False)):
            continue
        def nonzero(x):
            if hasattr(x, 'items'):
                return any(nonzero(v) for _, v in x.items())
            a = np.asarray(x, dtype=np.float64)
            return not np.isfinite(a).all() or bool(np.any(a != 0))
        if nonzero(value):
            raise ValueError(f'{camera}: randomized/moving camera mounts require explicit recalibration')


def _rig_id(cameras):
    stable = {name: dict(parent_frame=c['parent_frame'], fixed_transform=c['fixed_transform'])
              for name, c in sorted(cameras.items())}
    payload = json.dumps(stable, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def from_environment_config(env, *, visual_templates=None):
    """Capture static declared mounts after reset, without runtime camera poses.

    ``visual_templates`` is only a pure-test injection; real deployment reads
    CameraManager's declared VISUAL_MAP (env_cfg.camera.template.PINHOLE).
    """
    manager = env.camera_manager
    if visual_templates is None:
        module = sys.modules[type(manager).__module__]
        visual_templates = getattr(module, 'VISUAL_MAP')
    configs = manager.camera_config
    randomization = getattr(manager, 'random_config', {})
    robots = {r.arm_name.split('_')[0]: r for r in env.robot_manager.robot_list if r.type == 'target'}
    if set(robots) != {'left', 'right'}:
        raise ValueError('Expected exactly two target arms for rigid wrist calibration')
    cameras = {}
    for name, aliases in CAMERA_ALIASES.items():
        matches = [alias for alias in aliases if alias in configs]
        if len(matches) != 1:
            raise ValueError(f'{name}: expected one declared native camera')
        native = matches[0]
        config = configs[native]['camera']
        _no_pose_randomization(randomization.get(native, randomization.get('common', {})), name)
        mesh = config.get('mesh')
        if mesh not in ('pinhole', 'd435') or mesh not in visual_templates:
            raise ValueError(f'{name}: unsupported camera visual mount {mesh}')
        visual = visual_templates[mesh]
        if visual.get('asset_path') is not None:
            raise ValueError('Asset-backed camera visuals need an explicit validated local chain')
        visual_t = pose_matrix(visual['position'], visual['orientation'])
        local_t = declared_pose(config) @ visual_t @ USD_FROM_OPTICAL
        source = dict(native_camera=native, camera_config=dict(
            mesh=mesh, pos=list(config.get('pos', [0, 0, 1])), ori=list(config.get('ori', [0, 0, 0])),
            mount_link=config.get('mount_link')),
            config_euler='intrinsic XYZ degrees; wxyz when length 4',
            urdf_euler='extrinsic xyz radians',
            visual_local_usd=visual_t.tolist(), usd_from_optical=USD_FROM_OPTICAL.tolist(),
            runtime_camera_world_pose_used=False, object_scene_truth_used=False)
        if name in ARMS:
            arm, _ = ARMS[name]
            robot = robots[arm]
            mount = config.get('mount_link', '')
            if not mount or len(mount.split('/')) != 2:
                raise ValueError(f'{name}: expected declared robot/link mount')
            link = mount.split('/')[-1]
            declared = [entry for entry in robot.camera if entry.get('link') == link and entry.get('name') == 'cam_wrist']
            if len(declared) != 1:
                raise ValueError(f'{name}: mount does not match robot camera configuration')
            # Validate that the merged native camera really belongs to this arm.
            robot_index = env.robot_manager.robot_list.index(robot)
            scene_flags = getattr(env.robot_manager, 'use_scene_cfg', None)
            if scene_flags is not None:
                idx = 0
                expected_mount = None
                for i, candidate in enumerate(env.robot_manager.robot_list):
                    if candidate.type != 'target':
                        continue
                    expected_mount = f'robot{idx}' if scene_flags[i] else f'robot{idx-1}'
                    if scene_flags[i]:
                        idx += 1
                    if i == robot_index:
                        break
                if mount.split('/')[0] != expected_mount:
                    raise ValueError(f'{name}: arm and native camera parent disagree')
            robot_declared_t = declared_pose(declared[0])
            if not np.allclose(robot_declared_t, declared_pose(config), atol=1e-9, rtol=0):
                raise ValueError(f'{name}: merged camera pose differs from declared robot mount')
            chain, joints = fixed_urdf_transform(robot.urdf_path, robot.ee_link_name, link)
            fixed = chain @ local_t
            parent = arm + '_eef'
            source.update(urdf_sha256=hashlib.sha256(Path(robot.urdf_path).read_bytes()).hexdigest(),
                          urdf_path=str(robot.urdf_path), eef_link=robot.ee_link_name,
                          mount_link=link, fixed_joints=joints, T_eef_mount_link=chain.tolist())
        else:
            if config.get('mount_link') is not None:
                raise ValueError('Main camera must be fixed relative to environment origin')
            fixed, parent = local_t, 'environment'
        cameras[name] = dict(parent_frame=parent, fixed_transform=validate_transform(fixed).tolist(), source=source)
    return dict(version=VERSION, frame='environment_origin', optical_frame='+X right,+Y down,+Z forward',
                units='meters', rig_id=_rig_id(cameras), cameras=cameras,
                provenance='declared URDF fixed joints + camera mount config + USD optical conversion; no runtime camera poses')


def _validated_calibration(calibration):
    if (not isinstance(calibration, dict) or calibration.get('version') != VERSION
            or set(calibration.get('cameras', {})) != set(CAMERA_ALIASES)
            or calibration.get('rig_id') != _rig_id(calibration['cameras'])):
        raise ValueError('Missing, partial or changed rigid camera calibration')
    for name, camera in calibration['cameras'].items():
        expected = ARMS[name][0] + '_eef' if name in ARMS else 'environment'
        if camera['parent_frame'] != expected:
            raise ValueError('Invalid camera parent frame')
        validate_transform(camera['fixed_transform'])
    return calibration


def capture(calibration, eef_positions, eef_quaternions_wxyz):
    """Compose fixed extrinsics with measured link6 at this observation tick."""
    calibration = _validated_calibration(calibration)
    p, q = np.asarray(eef_positions), np.asarray(eef_quaternions_wxyz)
    if p.shape != (2, 3) or q.shape != (2, 4):
        raise ValueError('Expected measured left/right EEF positions and wxyz quaternions')
    poses = [pose_matrix(p[i], q[i]) for i in range(2)]
    arrays, descriptors = {}, {}
    for name, camera in calibration['cameras'].items():
        fixed = validate_transform(camera['fixed_transform'])
        if name in ARMS:
            _, index = ARMS[name]
            t = poses[index] @ fixed
            arrays[name + '_T_eef_camera'] = fixed.copy()
        else:
            t = fixed.copy()
        arrays[name + '_T_environment_camera'] = validate_transform(t).copy()
        descriptors[name] = dict(T_environment_camera=t.tolist(), T_camera_environment=invert_transform(t).tolist(),
            T_eef_camera=fixed.tolist() if name in ARMS else None,
            parent_frame=camera['parent_frame'], source=camera['source'],
            update='measured EEF pose at this observation; fixed mounting matrix' if name in ARMS else 'fixed environment camera',
            axes='+X right,+Y down,+Z forward', units='meters', transform_direction='p_environment = T_environment_camera @ [p_camera,1]')
    return arrays, descriptors


def from_observation(obs, calibration=None, episode_id=None, step_id=None, *, required=False):
    present = [key for key in OBSERVATION_KEYS if key in obs]
    if not present and calibration is None and not required:
        return {}
    if set(present) != set(OBSERVATION_KEYS):
        raise ValueError('All three rigid camera transforms and both fixed wrist mounts are required')
    arrays, descriptors = capture(calibration, obs['eef_positions'], obs['eef_quaternions_wxyz'])
    for key, expected in arrays.items():
        actual = validate_transform(obs[key])
        if not np.allclose(actual, expected, atol=1e-7, rtol=0):
            raise ValueError(f'{key}: stale/inconsistent transform for measured EEF observation')
    return dict(version=VERSION, episode_id=episode_id, step_id=step_id, rig_id=calibration['rig_id'],
                frame='environment_origin', cameras=descriptors)
