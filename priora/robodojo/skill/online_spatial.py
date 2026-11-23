"""Episode-local, observation-only spatial fitting. No simulator/calibration truth.

The model maps measured link6 positions to (u*Z, v*Z, Z). A tracked rigid
landmark's unknown offset is absorbed in the intercept ONLY within the validated
orientation/opening range. This is not a recovered optical camera extrinsic.
Wrist models use coordinates relative to the measured camera-bearing link6.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from ..io import InputError, write_json

VERSION = 'online_spatial_v2'
CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
ARMS = ('left', 'right')
LIMITS = dict(min_train=12, validation_count=5, pixel_error=5.0,
              depth_error_m=0.02, min_motion_sigma_m=0.005,
              max_condition=100.0, orientation_rad=0.10, opening_change=0.08,
              domain_margin_m=0.01, check_every_steps=100, expire_steps=200,
              max_samples=64)


def rotation(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise InputError('Invalid measured wxyz quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def measured_reference(packet, camera, arm):
    target = packet['current_eef'][arm]
    p, r = np.array(target['position'], dtype=float), rotation(target['quaternion_wxyz'])
    if p.shape != (3,) or not np.isfinite(p).all():
        raise InputError('Invalid measured EEF position')
    if camera != 'cam_high':
        mount = packet['current_eef']['left' if camera == 'cam_left_wrist' else 'right']
        rm = rotation(mount['quaternion_wxyz'])
        p = rm.T @ (p - np.array(mount['position']))
        r = rm.T @ r
    return p, r, float(target['gripper_opening_command'])


def pose_matches(sample, reference):
    dr = np.array(sample['rotation']).T @ np.array(reference['rotation'])
    angle = np.arccos(np.clip((np.trace(dr)-1)/2, -1, 1))
    return (angle <= LIMITS['orientation_rad'] and
            abs(sample['opening']-reference['opening']) <= LIMITS['opening_change'] and
            sample['image_size'] == reference['image_size'])


def in_domain(model, x):
    x = np.asarray(x)
    lo, hi = np.array(model['minimum']), np.array(model['maximum'])
    center = np.array(model['center'])
    # Box AND ellipsoid: do not silently extrapolate into unobserved corners.
    d = x-center
    radius = float(d @ np.array(model['inverse_covariance']) @ d)
    return bool(np.all(x >= lo-LIMITS['domain_margin_m']) and
                np.all(x <= hi+LIMITS['domain_margin_m']) and radius <= model['radius']+1)


def project(model, x):
    lifted = np.array(model['matrix']) @ np.r_[x, 1.0]
    if not np.isfinite(lifted).all() or not 0.1 <= lifted[2] <= 10:
        return None
    return np.r_[lifted[:2]/lifted[2], lifted[2]]


def residual(model, sample):
    pred = project(model, sample['position'])
    if pred is None:
        return dict(pixel_error=1e9, depth_error_m=1e9, passed=False)
    pixel = float(np.linalg.norm(pred[:2]-sample['uvz'][:2]))
    depth = abs(float(pred[2]-sample['uvz'][2]))
    return dict(pixel_error=pixel, depth_error_m=depth,
                passed=pixel <= LIMITS['pixel_error'] and depth <= LIMITS['depth_error_m'])


def fit(samples):
    if len(samples) < LIMITS['min_train']:
        return None, 'insufficient_samples'
    x = np.array([s['position'] for s in samples])
    center = x.mean(axis=0)
    motion = np.linalg.svd(x-center, compute_uv=False)/np.sqrt(len(x))
    if motion[-1] < LIMITS['min_motion_sigma_m'] or motion[0]/motion[-1] > LIMITS['max_condition']:
        return None, 'insufficient_independent_3d_motion'
    uvz = np.array([s['uvz'] for s in samples])
    lifted = np.c_[uvz[:, :2]*uvz[:, 2:3], uvz[:, 2]]
    # Centering avoids ill conditioning from the absolute environment origin.
    coef = np.linalg.lstsq(np.c_[x-center, np.ones(len(x))], lifted, rcond=None)[0].T
    coef[:, 3] -= coef[:, :3] @ center
    if np.linalg.cond(coef[:, :3]) > 1e6:
        return None, 'singular_projection'
    inv_cov = np.linalg.inv((x-center).T @ (x-center)/len(x))
    radius = max(float(d @ inv_cov @ d) for d in x-center)
    model = dict(matrix=coef.tolist(), center=center.tolist(),
                 minimum=x.min(axis=0).tolist(), maximum=x.max(axis=0).tolist(),
                 inverse_covariance=inv_cov.tolist(), radius=radius,
                 motion_sigma_m=motion.tolist(), reference=samples[0],
                 fit_steps=[s['step_id'] for s in samples])
    errors = [residual(model, s) for s in samples]
    if not all(e['passed'] for e in errors):
        return None, 'inconsistent_correspondences'
    model['training_max_pixel_error'] = max(e['pixel_error'] for e in errors)
    model['training_max_depth_error_m'] = max(e['depth_error_m'] for e in errors)
    return model, 'candidate_requires_future_validation'


class OnlineSpatial:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.directory = self.output/'spatial'
        self.directory.mkdir(exist_ok=True)
        self.path = self.directory/'state.json'
        self.state = dict(version=VERSION, episode_id=None, limits=LIMITS,
                          maps={}, event_count=0,
                          scope='landmark-specific link6-equivalent mapping; not optical extrinsics')
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state['version'] != VERSION:
                raise InputError('Spatial state version mismatch')

    def _save(self, event):
        self.state['event_count'] += 1
        write_json(self.directory/f"event_{self.state['event_count']:05d}.json", event)
        write_json(self.path, self.state)

    def _packet(self, rollout, observation_path):
        expected = getattr(rollout, 'observation_path', None)
        if expected is None or Path(observation_path).resolve() != Path(expected).resolve():
            raise InputError('Spatial tools require the latest acknowledged observation_path')
        path = Path(expected).resolve()
        if not path.is_relative_to(self.output/'observations'):
            raise InputError('Observation is outside this episode')
        packet = json.loads(path.read_text())
        if packet['episode_id'] != rollout.episode or packet['step_id'] != rollout.tick:
            raise InputError('Observation episode/step mismatch')
        if self.state['episode_id'] not in (None, rollout.episode):
            raise InputError('Spatial memory cannot cross episodes')
        self.state['episode_id'] = rollout.episode
        arrays = Path(packet['arrays_path']).resolve()
        if not arrays.is_relative_to(self.output/'observations'):
            raise InputError('Arrays must belong to this episode')
        return packet, arrays

    @staticmethod
    def _key(camera, arm, landmark):
        if camera not in CAMERAS or arm not in ARMS:
            raise InputError('Unknown camera or arm')
        if not isinstance(landmark, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', landmark):
            raise InputError('Use a stable 1..64 character landmark name (letters/digits/_/-)')
        return '/'.join((camera, arm, landmark))

    @staticmethod
    def _pixels(arrays, camera, pixel):
        try:
            uv = np.asarray(pixel, dtype=float)
        except (ValueError, TypeError) as error:
            raise InputError('pixel must contain finite original-resolution [u,v]') from error
        if uv.shape != (2,) or not np.isfinite(uv).all():
            raise InputError('pixel must contain finite original-resolution [u,v]')
        with np.load(arrays, allow_pickle=False) as data:
            rgb = data[camera]
            if camera+'_depth_m' not in data:
                raise InputError('Raw aligned RGBD is required; rendered JPEG depth is not numerical input')
            depth = data[camera+'_depth_m']
        h, w = rgb.shape[:2]
        if depth.shape == (h, w, 1):
            depth = depth[..., 0]
        if depth.shape != (h, w):
            raise InputError('RGB/depth image geometry mismatch')
        u, v = np.rint(uv).astype(int)
        if not (1 <= u < w-1 and 1 <= v < h-1):
            raise InputError('Landmark must be inside original image with a one-pixel margin')
        patch = depth[v-1:v+2, u-1:u+2]
        if not np.isfinite(patch).all() or np.any(patch < .1) or np.any(patch > 10):
            raise InputError('Invalid/out-of-range depth; do not submit this landmark')
        if np.ptp(patch) > .02:
            raise InputError('Depth edge/discontinuity: choose a visible rigid surface interior')
        return [float(uv[0]), float(uv[1]), float(depth[v, u])], [w, h]

    def record(self, rollout, *, observation_path, camera, arm, landmark, pixel):
        key = self._key(camera, arm, landmark)
        packet, arrays = self._packet(rollout, observation_path)
        uvz, size = self._pixels(arrays, camera, pixel)
        position, r, opening = measured_reference(packet, camera, arm)
        sample = dict(observation_path=str(observation_path), step_id=packet['step_id'],
                      position=position.tolist(), rotation=r.tolist(), opening=opening,
                      image_size=size, uvz=uvz)
        item = self.state['maps'].setdefault(key, dict(status='collecting', samples=[],
            candidate=None, validation_passes=0, validation_samples=[], last_check_step=None, seen_steps=[],
            camera=camera, arm=arm, landmark=landmark, reason='insufficient_samples'))
        if packet['step_id'] in item['seen_steps']:
            raise InputError('This landmark already has a sample at this step; repeated points do not validate a fit')
        item['seen_steps'].append(packet['step_id'])
        model = item['candidate']
        error = None
        if model:
            error = residual(model, sample)
            compatible = pose_matches(sample, model['reference']) and in_domain(model, position)
            if compatible and error['passed']:
                item['validation_passes'] += 1
                item['validation_samples'].append(sample['position'])
                item['validation_samples'] = item['validation_samples'][-32:]
                item['last_check_step'] = packet['step_id']
                v = np.array(item['validation_samples'])
                spread = (np.linalg.svd(v-v.mean(axis=0), compute_uv=False)/np.sqrt(len(v))
                          if len(v) >= LIMITS['validation_count'] else np.zeros(3))
                if item['status'] == 'stable' or (item['validation_passes'] >= LIMITS['validation_count']
                                                and spread[-1] >= 0.002):
                    item['status'] = 'stable'
                    item['reason'] = 'frozen_parameters_validated_on_future_observations'
                # Fit parameters intentionally unchanged, including check-only samples.
            else:
                item.update(status='collecting', candidate=None, validation_passes=0,
                            samples=[], validation_samples=[], last_check_step=None,
                            reason='domain_or_pose_changed' if not compatible else 'prediction_error')
        if item['candidate'] is None:
            if item['samples'] and not pose_matches(sample, item['samples'][0]):
                item['samples'] = []
                item['reason'] = 'landmark_orientation_or_opening_changed'
            item['samples'].append(sample)
            item['samples'] = item['samples'][-LIMITS['max_samples']:]
            candidate, reason = fit(item['samples'])
            item['reason'] = reason
            if candidate:
                item.update(candidate=candidate, status='validating', validation_passes=0, validation_samples=[])
        self._save(dict(kind='correspondence', key=key, sample=sample, residual=error,
                        status=item['status'], reason=item['reason']))
        return self.summary(packet)

    def _validity(self, item, packet):
        model = item['candidate']
        if not model or item['status'] != 'stable':
            return False, item['reason']
        x, r, opening = measured_reference(packet, item['camera'], item['arm'])
        descriptor = next((i for i in packet.get('images', []) if i['camera'] == item['camera']), None)
        if descriptor is None:
            return False, 'missing_current_image_geometry'
        with Image.open(descriptor['path']) as image:
            size = list(image.size)
        current = dict(rotation=r.tolist(), opening=opening, image_size=size)
        if not pose_matches(current, model['reference']):
            return False, 'orientation_or_opening_outside_validated_range'
        if not in_domain(model, x):
            return False, 'position_outside_validated_region'
        if packet['step_id']-item['last_check_step'] >= LIMITS['expire_steps']:
            return False, 'visual_recheck_overdue'
        return True, 'valid'

    def summary(self, packet):
        if self.state['episode_id'] not in (None, packet['episode_id']):
            raise InputError('Spatial memory cannot cross episodes')
        maps = {}
        for key, item in self.state['maps'].items():
            usable, reason = self._validity(item, packet)
            due = (item['status'] != 'stable' or item['last_check_step'] is None or
                   packet['step_id']-item['last_check_step'] >= LIMITS['check_every_steps'] or not usable)
            entry = dict(status=item['status'], usable=usable, reason=reason, visual_check_due=due,
                         training_samples=len(item['samples']), validation_passes=item['validation_passes'])
            if usable:
                x, _, _ = measured_reference(packet, item['camera'], item['arm'])
                prediction = project(item['candidate'], x)
                if prediction is not None:
                    entry['predicted_landmark_uvz'] = prediction.tolist()
            maps[key] = entry
        result = dict(version=VERSION, state_path=str(self.path), step_id=packet['step_id'],
                      maps=maps, note='Predictions are estimates, never new visual observations. '
                      'Stable maps need sparse visual checks, not refitting every action.')
        write_json(self.directory/'latest.json', result)
        return result

    def query(self, rollout, *, observation_path, camera, arm, landmark, pixel):
        if not isinstance(pixel, list) or len(pixel) not in (0, 2):
            raise InputError('pixel must be [] for projection or [u,v] for inverse mapping')
        key = self._key(camera, arm, landmark)
        packet, arrays = self._packet(rollout, observation_path)
        item = self.state['maps'].get(key)
        if item is None:
            return dict(usable=False, reason='no_correspondences', key=key)
        usable, reason = self._validity(item, packet)
        if not usable:
            return dict(usable=False, reason=reason, key=key)
        model = item['candidate']
        if len(pixel) == 0:
            x, _, _ = measured_reference(packet, camera, arm)
            pred = project(model, x)
            return dict(usable=pred is not None, key=key,
                        predicted_landmark_uvz=None if pred is None else pred.tolist(),
                        parameter_sha256=hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest())
        uvz, size = self._pixels(arrays, camera, pixel)
        if size != model['reference']['image_size']:
            return dict(usable=False, reason='image_geometry_changed')
        matrix = np.array(model['matrix'])
        x = np.linalg.solve(matrix[:, :3], np.r_[np.array(uvz[:2])*uvz[2], uvz[2]]-matrix[:, 3])
        if not in_domain(model, x):
            return dict(usable=False, reason='query_outside_validated_region')
        if camera != 'cam_high':
            mount = packet['current_eef']['left' if camera == 'cam_left_wrist' else 'right']
            x = rotation(mount['quaternion_wxyz']) @ x + mount['position']
        return dict(usable=True, key=key, reference_equivalent_environment_position=x.tolist(),
                    warning='Landmark-specific link6-equivalent coordinate. The unknown landmark offset '
                    'is absorbed in the fit. This is NOT an arbitrary object surface world coordinate '
                    'or a grasp target. Use only within the validated landmark/pose/opening domain.')
