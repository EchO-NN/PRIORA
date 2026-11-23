"""Observation-only spatial geometry using known optics and fixed camera mounts.

No affine fitting, private scene poses, simulated calibration motions or action
execution occurs here. Every wrist transform is checked against the supplied
measured EEF pose and the fixed, episode-local camera mounting transform.
"""
import copy
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from ..camera_intrinsics import key as intrinsics_key, validate_k
from ..io import InputError, write_json
from .near_field_memory import NearFieldMemory

VERSION = 'rigid_spatial_v3'
GEOMETRY_VERSION = 'rigid_camera_geometry_v1'
DENSE_VERSION = 'dense_rgbd_xyz_v1'
CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
ARMS = ('left', 'right')
SURFACE_WARNING = (
    'This is the currently visible surface point sampled from raw RGBD, not a grasp '
    'target or an EEF target. Contact offsets, approach direction, object identity, '
    'occlusion and reachability must be checked separately.'
)
ORIGIN_WARNING = (
    'This projects the measured link6/EEF coordinate origin, not a visible fingertip '
    'or gripper surface. In-frame projection does not establish visibility: another '
    'surface can occlude it. It is a computed prediction, not a new observation.'
)


def rotation(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-3:
        raise InputError('Invalid measured unit wxyz quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def transform(value, name):
    """Validate an SE(3) transform rather than silently repair a bad matrix."""
    try:
        t = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise InputError(f'{name}: invalid rigid transform') from error
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise InputError(f'{name}: invalid rigid transform')
    r = t[:3, :3]
    if (not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(r.T @ r, np.eye(3), atol=2e-5, rtol=0)
            or not np.isclose(np.linalg.det(r), 1, atol=2e-5, rtol=0)):
        raise InputError(f'{name}: matrix must be a proper rigid transform')
    return t


def measured_eef(packet, arm):
    try:
        pose = packet['current_eef'][arm]
        p = np.asarray(pose['position'], dtype=float)
        r = rotation(pose['quaternion_wxyz'])
        opening = float(pose['gripper_opening_command'])
    except InputError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise InputError(f'{arm}: measured EEF pose is required') from error
    if p.shape != (3,) or not np.isfinite(p).all() or not np.isfinite(opening):
        raise InputError(f'{arm}: invalid measured EEF pose/opening')
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = r, p
    return t, opening


def inverse(t):
    result = np.eye(4)
    result[:3, :3] = t[:3, :3].T
    result[:3, 3] = -t[:3, :3].T @ t[:3, 3]
    return result


def point(t, p):
    return t[:3, :3] @ p + t[:3, 3]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class RigidSpatial:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.directory = self.output/'spatial'
        self.directory.mkdir(exist_ok=True)
        self.path = self.directory/'state.json'
        self.state = dict(version=VERSION, episode_id=None, calibration=None,
                          last_step=None, last_observation_fingerprint=None,
                          landmarks={}, event_count=0,
                          scope='known fixed camera mounts plus current measured EEF poses; no fitted extrinsics')
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state.get('version') != VERSION:
                raise InputError('Spatial state version mismatch; use a fresh episode directory')
        self.near_field = NearFieldMemory(self.output)

    def _save(self, event=None):
        if event is not None:
            self.state['event_count'] += 1
            write_json(self.directory/f"event_{self.state['event_count']:05d}.json", event)
        write_json(self.path, self.state)

    @staticmethod
    def _key(camera, arm, landmark):
        if camera not in CAMERAS or arm not in ARMS:
            raise InputError('Unknown camera or arm')
        if not isinstance(landmark, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', landmark):
            raise InputError('Use a stable 1..64 character landmark name (letters/digits/_/-)')
        return '/'.join((camera, arm, landmark))

    def _packet(self, rollout, observation_path):
        expected = getattr(rollout, 'observation_path', None)
        if expected is None or Path(observation_path).resolve() != Path(expected).resolve():
            raise InputError('Spatial tools require the latest acknowledged observation_path')
        path = Path(expected).resolve()
        if not path.is_relative_to(self.output/'observations'):
            raise InputError('Observation is outside this episode')
        packet = json.loads(path.read_text())
        if packet.get('episode_id') != rollout.episode or packet.get('step_id') != rollout.tick:
            raise InputError('Observation episode/step mismatch')
        arrays = Path(packet['arrays_path']).resolve()
        if arrays.parent != path.parent or not arrays.is_relative_to(self.output/'observations'):
            raise InputError('Arrays must belong to the current observation in this episode')
        return packet, arrays

    def _geometry(self, packet):
        """Freeze calibration, check pose freshness and recompute dynamic transforms."""
        episode, step = packet.get('episode_id'), packet.get('step_id')
        if not isinstance(episode, str) or not episode or type(step) is not int or step < 0:
            raise InputError('Valid observation episode/step identity is required')
        if self.state['episode_id'] not in (None, episode):
            raise InputError('Spatial memory cannot cross episodes')
        geometry = packet.get('camera_geometry')
        if not isinstance(geometry, dict) or geometry.get('version') != GEOMETRY_VERSION:
            raise InputError('Known rigid camera geometry is required; no affine fallback')
        if geometry.get('episode_id') != episode or geometry.get('step_id') != step:
            raise InputError('Camera geometry episode/step mismatch or stale transforms')
        rig_id = geometry.get('rig_id')
        if not isinstance(rig_id, str) or not rig_id:
            raise InputError('Camera rig identity is required')
        cameras = geometry.get('cameras', {})
        if set(cameras) != set(CAMERAS):
            raise InputError('All three rigid camera transforms are required')
        eef = {arm: measured_eef(packet, arm) for arm in ARMS}
        frames, calibration = {}, dict(rig_id=rig_id, cameras={})
        intrinsics = packet.get('camera_intrinsics', {})
        for camera in CAMERAS:
            desc = cameras[camera]
            try:
                kdesc = intrinsics[camera]
                k = validate_k(kdesc['K'])
                size = kdesc['original_size_wh']
                if (not isinstance(size, list) or len(size) != 2
                        or any(type(x) is not int or x <= 0 for x in size)):
                    raise ValueError('invalid original calibrated resolution')
            except (KeyError, TypeError, ValueError) as error:
                raise InputError(f'{camera}: valid calibrated intrinsics are required') from error
            supplied = transform(desc.get('T_environment_camera'), camera)
            parent = desc.get('parent_frame')
            source = desc.get('source')
            if not source:
                raise InputError(f'{camera}: calibration source is required')
            if camera == 'cam_high':
                if parent != 'environment' or desc.get('T_eef_camera') is not None:
                    raise InputError('Head camera must have a fixed environment mounting transform')
                fixed = supplied
                current = supplied
            else:
                arm = 'left' if camera == 'cam_left_wrist' else 'right'
                if parent != arm+'_eef':
                    raise InputError(f'{camera}: incorrect camera-bearing EEF frame')
                fixed = transform(desc.get('T_eef_camera'), camera+' fixed mount')
                current = eef[arm][0] @ fixed
                if not np.allclose(current, supplied, atol=2e-5, rtol=0):
                    raise InputError(f'{camera}: stale transform disagrees with current measured EEF and fixed mount')
            if 'T_camera_environment' in desc and not np.allclose(
                    transform(desc['T_camera_environment'], camera+' inverse'), inverse(current), atol=2e-5, rtol=0):
                raise InputError(f'{camera}: inconsistent inverse camera transform')
            calibration['cameras'][camera] = dict(parent_frame=parent, fixed_transform=fixed.tolist(),
                K=k.tolist(), original_size_wh=size, source=source,
                intrinsics_source=kdesc.get('source', 'supplied camera intrinsics'))
            frames[camera] = dict(T_environment_camera=current, T_camera_environment=inverse(current),
                                 K=k, size=size, source=source,
                                 intrinsics_source=kdesc.get('source', 'supplied camera intrinsics'))
        frozen = self.state['calibration']
        # Known mounts and optics are immutable within an episode. Tolerances
        # allow numerical serialization noise, never a changed rig or optics.
        if frozen is not None:
            if frozen['rig_id'] != rig_id:
                raise InputError('Camera rig identity changed within the episode')
            for camera in CAMERAS:
                old, new = frozen['cameras'][camera], calibration['cameras'][camera]
                if (any(old[k] != new[k] for k in ('parent_frame', 'original_size_wh', 'source', 'intrinsics_source'))
                        or not np.allclose(old['K'], new['K'], atol=1e-7, rtol=0)
                        or not np.allclose(old['fixed_transform'], new['fixed_transform'], atol=2e-6, rtol=0)):
                    raise InputError(f'{camera}: frozen camera calibration changed within the episode')
                # Always compute from frozen parameters, not the caller's copy.
                fixed = np.asarray(old['fixed_transform'])
                current = fixed if camera == 'cam_high' else eef['left' if camera == 'cam_left_wrist' else 'right'][0] @ fixed
                frames[camera].update(T_environment_camera=current, T_camera_environment=inverse(current), K=np.asarray(old['K']))
        observation_hash = fingerprint(dict(episode=episode, step=step,
            measured_eef=packet['current_eef'], calibration=calibration,
            matrices={camera: frames[camera]['T_environment_camera'].tolist() for camera in CAMERAS}))
        if self.state['last_step'] is not None:
            if step < self.state['last_step']:
                raise InputError('Stale observation step cannot update rigid spatial memory')
            if step == self.state['last_step'] and observation_hash != self.state['last_observation_fingerprint']:
                raise InputError('Measured geometry changed without a new observation step')
        self.state.update(episode_id=episode, calibration=frozen or copy.deepcopy(calibration),
                          last_step=step, last_observation_fingerprint=observation_hash)
        self._save()
        return frames, eef

    @staticmethod
    def _projection(frame, environment_point):
        camera_point = point(frame['T_camera_environment'], np.asarray(environment_point))
        z = float(camera_point[2])
        pixel = None
        inside = False
        if z > 0:
            p = frame['K'] @ camera_point
            pixel = (p[:2]/p[2]).tolist()
            inside = bool(0 <= pixel[0] < frame['size'][0] and 0 <= pixel[1] < frame['size'][1])
        return dict(pixel=pixel, optical_depth_m=z, in_front=z > 0,
                    within_depth_range=0.1 <= z <= 10, in_image=inside,
                    usable=inside and 0.1 <= z <= 10,
                    visibility='unknown; geometric projection does not prove lack of occlusion')

    @staticmethod
    def _dense_descriptor(packet, arrays, camera, frame):
        if 'dense_geometry' not in packet:
            return None
        dense = packet['dense_geometry']
        if not isinstance(dense, dict) or camera not in dense or not isinstance(dense[camera], dict):
            raise InputError(f'{camera}: declared dense geometry is missing')
        desc = dense[camera]
        expected = dict(version=DENSE_VERSION, camera=camera, episode_id=packet['episode_id'],
                        step_id=packet['step_id'], rig_id=packet['camera_geometry']['rig_id'],
                        xyz_key='xyz_environment', valid_key='valid', dtype='float32',
                        units='meters', frame='environment_origin',
                        original_size_wh=frame['size'], shape=[frame['size'][1], frame['size'][0], 3])
        if any(desc.get(key) != value for key, value in expected.items()):
            raise InputError(f'{camera}: dense geometry descriptor identity or geometry mismatch')
        try:
            path = Path(desc['path']).resolve()
        except (KeyError, TypeError, ValueError) as error:
            raise InputError(f'{camera}: dense geometry file path is required') from error
        if path.parent != Path(arrays).resolve().parent:
            raise InputError('Dense geometry must belong to the current observation directory')
        return dict(desc, path=str(path))

    @staticmethod
    def _cached_point(desc, frame, camera, u, v, recomputed):
        try:
            with np.load(desc['path'], allow_pickle=False) as data:
                for key in ('version', 'camera', 'episode_id', 'step_id', 'rig_id'):
                    if key not in data or data[key].shape != () or data[key].item() != desc[key]:
                        raise InputError(f'{camera}: dense archive identity mismatch for {key}')
                if any(key not in data for key in ('xyz_environment', 'valid', 'K', 'T_environment_camera')):
                    raise InputError(f'{camera}: incomplete dense coordinate archive')
                xyz, valid = data['xyz_environment'], data['valid']
                if (xyz.shape != tuple(desc['shape']) or xyz.dtype != np.float32
                        or valid.shape != tuple(desc['shape'][:2]) or valid.dtype != np.bool_):
                    raise InputError(f'{camera}: dense matrix shape or dtype mismatch')
                if (data['K'].shape != (3, 3) or not np.allclose(data['K'], frame['K'], atol=1e-7, rtol=0)
                        or data['T_environment_camera'].shape != (4, 4)
                        or not np.allclose(data['T_environment_camera'], frame['T_environment_camera'], atol=2e-5, rtol=0)):
                    raise InputError(f'{camera}: dense archive calibration disagrees with current geometry')
                cached = xyz[v, u].astype(np.float64)
                if not bool(valid[v, u]) or not np.isfinite(cached).all():
                    raise InputError(f'{camera}: dense surface pixel is invalid')
                if not np.allclose(cached, recomputed, atol=1e-5, rtol=0):
                    raise InputError(f'{camera}: dense surface point disagrees with current raw depth and calibration')
        except InputError:
            raise
        except (OSError, ValueError, KeyError) as error:
            raise InputError(f'{camera}: invalid dense coordinate archive') from error
        return cached

    @classmethod
    def _surface(cls, arrays, camera, pixel, frame, eef, packet):
        try:
            uv = np.asarray(pixel, dtype=float)
        except (ValueError, TypeError) as error:
            raise InputError('pixel must contain finite original-resolution [u,v]') from error
        if uv.shape != (2,) or not np.isfinite(uv).all():
            raise InputError('pixel must contain finite original-resolution [u,v]')
        with np.load(arrays, allow_pickle=False) as data:
            if camera not in data or camera+'_depth_m' not in data or intrinsics_key(camera) not in data:
                raise InputError('Raw aligned RGBD and archived camera intrinsics are required')
            rgb = data[camera]
            depth = data[camera+'_depth_m']
            if not np.allclose(data[intrinsics_key(camera)], frame['K'], atol=1e-7, rtol=0):
                raise InputError('Archived depth/RGB camera intrinsics disagree with observation')
        h, w = rgb.shape[:2]
        if list(frame['size']) != [w, h]:
            raise InputError('RGB geometry differs from calibrated original resolution')
        if depth.shape == (h, w, 1):
            depth = depth[..., 0]
        if depth.shape != (h, w):
            raise InputError('RGB/depth image geometry mismatch')
        if np.any(uv < 1) or uv[0] >= w-1 or uv[1] >= h-1:
            raise InputError('Surface pixel must have a one-pixel image margin')
        u, v = np.rint(uv).astype(int)
        if not (1 <= u < w-1 and 1 <= v < h-1):
            raise InputError('Surface pixel must have a one-pixel image margin')
        patch = depth[v-1:v+2, u-1:u+2]
        if not np.isfinite(patch).all() or np.any(patch < .1) or np.any(patch > 10):
            raise InputError(
                'Invalid/out-of-range depth: current numeric surface queries require the entire '
                '3x3 patch in the configured optical-Z range 0.1..10 m. RGB may still show the '
                'surface and the raw sensor may have closer values; this is not proof of a '
                'physical camera blind zone. Inspect another current view or use spatial_recall '
                'for explicitly historical local evidence. Do not substitute old XYZ as a '
                'current measurement or grasp target.')
        if float(np.ptp(patch)) > .02:
            raise InputError('Depth edge/discontinuity: choose a visible surface interior')
        z = float(depth[v, u])
        camera_point = z * np.linalg.solve(frame['K'], [u, v, 1])
        environment_point = point(frame['T_environment_camera'], camera_point)
        dense = cls._dense_descriptor(packet, arrays, camera, frame)
        if dense is not None:
            environment_point = cls._cached_point(dense, frame, camera, u, v, environment_point)
            source = dict(kind='cached_dense_xyz', path=dense['path'], xyz_key=dense['xyz_key'],
                          valid_key=dense['valid_key'], matrix_index_vu=[int(v), int(u)],
                          version=DENSE_VERSION, sample_checked_against_raw_depth=True)
        else:
            source = dict(kind='raw_depth_backprojection', arrays_path=str(arrays),
                          reason='No dense geometry cache was supplied for this observation')
        return dict(requested_pixel=uv.tolist(), sampled_pixel=[int(u), int(v)],
                    pixel_sampling='nearest original-resolution raw depth pixel; backprojection uses sampled_pixel',
                    optical_depth_m=z, camera_position_m=camera_point.tolist(),
                    environment_position_m=environment_point.tolist(),
                    eef_positions_m={arm: point(inverse(eef[arm][0]), environment_point).tolist() for arm in ARMS},
                    depth_patch_span_m=float(np.ptp(patch)), geometry_source=source,
                    warning=SURFACE_WARNING)

    def summary(self, packet):
        frames, eef = self._geometry(packet)
        cameras = {}
        for camera, frame in frames.items():
            cameras[camera] = dict(T_environment_camera=frame['T_environment_camera'].tolist(),
                T_camera_environment=frame['T_camera_environment'].tolist(),
                K=frame['K'].tolist(), original_size_wh=frame['size'],
                measured_eef_origin_projections={arm: self._projection(frame, eef[arm][0][:3, 3]) for arm in ARMS})
        result = dict(version=VERSION, usable=True, episode_id=packet['episode_id'],
            step_id=packet['step_id'], rig_id=self.state['calibration']['rig_id'],
            state_path=str(self.path), cameras=cameras,
            registered_landmarks=list(self.state['landmarks']),
            note='Camera mounting transforms and intrinsics are fixed for this episode. Wrist poses are recomputed '
                 'from the current measured EEF poses. No online affine calibration is fitted.',
            origin_warning=ORIGIN_WARNING, surface_warning=SURFACE_WARNING)
        if 'dense_geometry' in packet:
            arrays = Path(packet['arrays_path']).resolve()
            if not arrays.is_relative_to(self.output/'observations'):
                raise InputError('Dense geometry observation must belong to this episode')
            result['dense_geometry'] = {
                camera: self._dense_descriptor(packet, arrays, camera, frame)
                for camera, frame in frames.items()}
            result['dense_geometry_note'] = (
                'Dense XYZ matrices are stored as per-camera NPZ files, not expanded into model text. '
                'spatial_query samples the cached surface XYZ and checks its identity, calibration, '
                'validity and agreement with current raw depth. Surface points are not grasp targets.')
        result['near_field_memory'] = self.near_field.summary(packet)
        write_json(self.directory/'latest.json', result)
        return result

    def remember(self, rollout, *, observation_path, camera, label, pixel, radius):
        """Store local scene evidence, independently of rigid robot registration."""
        packet, _ = self._packet(rollout, observation_path)
        self._geometry(packet)
        return self.near_field.remember(packet, camera=camera, label=label,
                                        pixel=pixel, radius=radius)

    def recall(self, rollout, *, observation_path, memory_id):
        """Read dated evidence without promoting it to current sensor geometry."""
        packet, _ = self._packet(rollout, observation_path)
        self._geometry(packet)
        return self.near_field.recall(packet, memory_id=memory_id)

    def record(self, rollout, *, observation_path, camera, arm, landmark, pixel):
        key = self._key(camera, arm, landmark)
        packet, arrays = self._packet(rollout, observation_path)
        frames, eef = self._geometry(packet)
        sample = self._surface(arrays, camera, pixel, frames[camera], eef, packet)
        previous = self.state['landmarks'].get(key)
        if previous and previous['step_id'] == packet['step_id']:
            raise InputError('This landmark is already recorded at this observation step')
        item = dict(camera=camera, arm=arm, landmark=landmark,
            observation_path=str(Path(observation_path).resolve()), arrays_path=str(arrays),
            episode_id=packet['episode_id'], step_id=packet['step_id'],
            local_eef_position_m=sample['eef_positions_m'][arm],
            gripper_opening_command=eef[arm][1], measured_surface=sample,
            K=frames[camera]['K'].tolist(), intrinsics_source=frames[camera]['intrinsics_source'],
            rig_id=self.state['calibration']['rig_id'],
            registration_assumption='User-selected visible point belongs to a rigid surface attached to this EEF. '
                                    'This semantic assumption is not verified by depth; do not register cloth or objects as gripper landmarks.')
        self.state['landmarks'][key] = item
        self._save(dict(kind='rigid_surface_registration', key=key, sample=item))
        return dict(self.summary(packet), recorded_landmark=copy.deepcopy(item))

    def query(self, rollout, *, observation_path, camera, arm, landmark, pixel):
        if not isinstance(pixel, list) or len(pixel) not in (0, 2):
            raise InputError('pixel must be [] for EEF-origin projection or [u,v] for a visible surface')
        key = self._key(camera, arm, landmark)
        packet, arrays = self._packet(rollout, observation_path)
        frames, eef = self._geometry(packet)
        frame = frames[camera]
        identity = dict(version=VERSION, key=key, camera=camera,
            episode_id=packet['episode_id'], step_id=packet['step_id'],
            rig_id=self.state['calibration']['rig_id'], observation_path=str(observation_path))
        if pixel:
            return dict(identity, usable=True, kind='visible_surface_point',
                        **self._surface(arrays, camera, pixel, frame, eef, packet))
        result = dict(identity, kind='measured_eef_origin_projection', arm=arm,
            environment_position_m=eef[arm][0][:3, 3].tolist(),
            **self._projection(frame, eef[arm][0][:3, 3]), warning=ORIGIN_WARNING)
        item = self.state['landmarks'].get(key)
        if item:
            if abs(eef[arm][1]-item['gripper_opening_command']) > 1e-6:
                result['registered_landmark_projection'] = dict(usable=False,
                    reason='gripper opening changed; rigid fingertip registration may no longer apply')
            else:
                position = point(eef[arm][0], np.asarray(item['local_eef_position_m']))
                result['registered_landmark_projection'] = dict(
                    **self._projection(frame, position), environment_position_m=position.tolist(),
                    recorded_at_step=item['step_id'], warning=item['registration_assumption'],
                    evidence_observation_path=item['observation_path'])
        return result


# Preserve the runtime constructor API while changing its implementation/version.
OnlineSpatial = RigidSpatial
