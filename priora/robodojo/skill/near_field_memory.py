"""Bounded references and explicit historical RGBD patches for close-up work.

The 0.1 m cutoff is a configured depth-validity rule, not RGB invisibility.
This module never advances simulation or transports old points with a wrist.
The caller validates the full rigid calibration before invoking these helpers.
"""
import copy
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from ..camera_geometry import validate_transform
from ..camera_intrinsics import CAMERA_ALIASES, key as intrinsics_key, validate_k
from ..io import InputError, sha256, write_json
from .near_field_visual import compare_patch

VERSION = 'near_field_memory_v1'
CAMERAS = tuple(CAMERA_ALIASES)
MAX_SNAPSHOTS = 32
MAX_PATCHES = 64
DEPTH_RANGE_M = (.1, 10.)
WARNING = (
    'Configured valid optical depth is 0.1..10 m. A finite raw depth below 0.1 m '
    'is excluded from current XYZ; RGB may still show the surface. Invalid depth '
    'does not prove empty space, absence of cloth, or a failed sensor. Historical '
    'patches are prior observations only, not current geometry or grasp targets. '
    'Cloth may move or deform after any control step. Reobserve before contact '
    'or grasp confirmation, using another camera or a permitted free-arm view.'
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class NearFieldMemory:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.directory = self.output/'spatial'/'near_field'
        self.patches = self.directory/'patches'
        self.patches.mkdir(parents=True, exist_ok=True)
        self.path = self.directory/'index.json'
        self.state = dict(version=VERSION, episode_id=None, rig_id=None,
                          last_step=None, last_fingerprint=None, snapshots=[], memories=[], visual_checks={})
        if self.path.exists():
            try:
                self.state = json.loads(self.path.read_text())
                if (self.state.get('version') != VERSION
                        or not isinstance(self.state.get('snapshots'), list)
                        or not isinstance(self.state.get('memories'), list)
                        or not isinstance(self.state.get('visual_checks', {}), dict)):
                    raise ValueError('invalid index schema')
            except (OSError, ValueError, TypeError) as error:
                raise InputError('Invalid near-field memory index; use a fresh episode directory') from error
        self.state.setdefault('visual_checks', {})

    def _source_path(self, value, *, parent=None):
        try:
            path = Path(value).resolve()
        except (TypeError, ValueError) as error:
            raise InputError('A source observation file path is required') from error
        if (not path.is_relative_to(self.output/'observations') or not path.is_file()
                or (parent is not None and path.parent != parent)):
            raise InputError('Near-field sources must belong to this episode observation directory')
        return path

    def _packet(self, packet):
        try:
            episode, step = packet['episode_id'], packet['step_id']
            geometry = packet['camera_geometry']
            rig = geometry['rig_id']
            if (not isinstance(episode, str) or not episode or type(step) is not int or step < 0
                    or not isinstance(rig, str) or not rig
                    or geometry.get('version') != 'rigid_camera_geometry_v1'
                    or geometry.get('episode_id') != episode or geometry.get('step_id') != step
                    or set(geometry['cameras']) != set(CAMERAS)):
                raise ValueError('invalid observation identity')
            arrays = self._source_path(packet['arrays_path'])
            observation = self._source_path(packet['observation_path'], parent=arrays.parent)
        except InputError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise InputError('Current acknowledged observation identity and rigid geometry are required') from error
        if self.state['episode_id'] not in (None, episode) or self.state['rig_id'] not in (None, rig):
            raise InputError('Near-field memory cannot cross episodes or camera rigs')
        if self.state['last_step'] is not None and step < self.state['last_step']:
            raise InputError('Stale observation cannot update near-field memory')
        cameras, loaded = {}, {}
        try:
            with np.load(arrays, allow_pickle=False) as data:
                for camera in CAMERAS:
                    rgb = data[camera]
                    depth = np.asarray(data[camera+'_depth_m'], dtype=np.float64)
                    if depth.ndim == 3 and depth.shape[-1] == 1:
                        depth = depth[..., 0]
                    desc = packet['camera_intrinsics'][camera]
                    k = validate_k(desc['K'])
                    t = validate_transform(geometry['cameras'][camera]['T_environment_camera'])
                    if (rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8
                            or depth.shape != rgb.shape[:2]
                            or list(desc['original_size_wh']) != [rgb.shape[1], rgb.shape[0]]
                            or not np.allclose(data[intrinsics_key(camera)], k, atol=1e-7, rtol=0)):
                        raise ValueError('RGBD and calibration mismatch')
                    valid = np.isfinite(depth) & (depth >= .1) & (depth <= 10.)
                    cameras[camera] = dict(K=k.tolist(), T_environment_camera=t.tolist(),
                        original_size_wh=list(desc['original_size_wh']),
                        source=copy.deepcopy(geometry['cameras'][camera].get('source')),
                        intrinsics_source=desc.get('source'),
                        depth_counts=dict(total=int(depth.size), valid=int(valid.sum()),
                            below_min_positive=int((np.isfinite(depth) & (depth > 0) & (depth < .1)).sum()),
                            above_max=int((np.isfinite(depth) & (depth > 10)).sum()),
                            nonpositive_or_nonfinite=int((~np.isfinite(depth) | (depth <= 0)).sum())))
                    loaded[camera] = (rgb, depth, k, t, valid)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise InputError('Near-field memory requires original aligned RGBD and matching archived calibration') from error
        dense = {}
        for camera, desc in packet.get('dense_geometry', {}).items():
            if (camera not in CAMERAS or not isinstance(desc, dict)
                    or any(desc.get(key) != value for key, value in dict(
                        version='dense_rgbd_xyz_v1', camera=camera, episode_id=episode,
                        step_id=step, rig_id=rig).items())):
                raise InputError('Dense source identity differs from the current observation')
            path = self._source_path(desc.get('path'), parent=arrays.parent)
            dense[camera] = dict(path=str(path), xyz_key='xyz_environment', valid_key='valid')
        snapshot = dict(episode_id=episode, step_id=step, rig_id=rig,
            observation_path=str(observation), arrays_path=str(arrays), arrays_sha256=sha256(arrays),
            cameras=cameras, dense_geometry=dense)
        fingerprint = _digest(snapshot)
        if step == self.state['last_step'] and fingerprint != self.state['last_fingerprint']:
            raise InputError('Observation sources changed without a new acknowledged control step')
        return snapshot, loaded, fingerprint

    @staticmethod
    def _history_flags(step, current_step):
        age = current_step-step
        return dict(historical_only=True, current_measurement=False,
            age_control_steps=age, stale_after_controls=age > 0,
            require_reobservation_before_contact=True, object_tracking=False,
            warning=WARNING)

    def _memory_view(self, item, step):
        check = copy.deepcopy(self.state['visual_checks'].get(item['memory_id'],
            dict(status='unverified', invalidated=False, usable_for_planning_hint=False,
                 reason='No RGB reprojection comparison exists for the current observation')))
        return dict(copy.deepcopy(item), **self._history_flags(item['step_id'], step),
                    visual_check=check, visual_status=check['status'],
                    invalidated=check['invalidated'],
                    usable_for_planning_hint=check['usable_for_planning_hint'])

    def _load_memory(self, item):
        """Check immutable evidence before comparing or returning its reference."""
        memory_id = item['memory_id']
        if not isinstance(memory_id, str) or not re.fullmatch(r'nf_[0-9a-f]{32}', memory_id):
            raise InputError('Historical patch has an invalid canonical identity')
        if item['episode_id'] != self.state['episode_id'] or item['rig_id'] != self.state['rig_id']:
            raise InputError('Historical patch episode/rig identity mismatch')
        for field, suffix in (('archive', '.npz'), ('image', '.png')):
            path = Path(item[field+'_path']).resolve()
            if (path != (self.patches/(memory_id+suffix)).resolve() or not path.is_file()
                    or sha256(path) != item[field+'_sha256']):
                raise InputError('Historical patch was modified or moved; refusing altered evidence')
        try:
            with np.load(item['archive_path'], allow_pickle=False) as data:
                metadata = json.loads(data['metadata_json'].item())
                rgb, xyz, valid = data['rgb'], data['xyz_environment'], data['valid']
            if any(item.get(key) != value for key, value in metadata.items()):
                raise ValueError('index metadata differs from immutable archive')
            identity = {key: metadata[key] for key in ('version', 'episode_id', 'step_id',
                'rig_id', 'camera', 'label', 'sampled_pixel', 'radius', 'arrays_sha256')}
            if memory_id != 'nf_'+_digest(identity)[:32]:
                raise ValueError('canonical memory identity changed')
        except (OSError, KeyError, ValueError, TypeError) as error:
            raise InputError('Historical patch identity was altered; refusing changed evidence') from error
        return rgb, xyz, valid

    def _refresh_visual_checks(self, snapshot, loaded):
        """Reproject once per observation; detected motion irreversibly retires a patch."""
        checks = self.state['visual_checks']
        retained = {item['memory_id'] for item in self.state['memories']}
        changed = False
        for memory_id in set(checks)-retained:
            del checks[memory_id]
            changed = True
        for item in self.state['memories']:
            memory_id = item['memory_id']
            previous = checks.get(memory_id, {})
            if previous.get('invalidated') or previous.get('checked_at_step') == snapshot['step_id']:
                continue
            old_rgb, old_xyz, old_valid = self._load_memory(item)
            rgb, depth, k, t, _ = loaded[item['camera']]
            result = compare_patch(old_rgb, old_xyz, old_valid,
                current_rgb=rgb, current_depth_m=depth, K=k, T_environment_camera=t,
                source_step=item['step_id'], current_step=snapshot['step_id'])
            status = result.get('status')
            if status not in ('same_observation', 'consistent', 'invalidated_rgb_change', 'unverified'):
                raise InputError('Unknown RGB memory validation status; refusing unsupported evidence')
            invalidated = status == 'invalidated_rgb_change'
            checks[memory_id] = dict(copy.deepcopy(result), source_step=item['step_id'],
                checked_at_step=snapshot['step_id'], invalidated=invalidated,
                invalidated_at_step=snapshot['step_id'] if invalidated else None,
                usable_for_planning_hint=status in ('same_observation', 'consistent'),
                historical_only=True, current_measurement=False, object_tracking=False,
                require_reobservation_before_contact=True)
            changed = True
        return changed

    def _summary(self, snapshot):
        return dict(version=VERSION, index_path=str(self.path),
            episode_id=snapshot['episode_id'], step_id=snapshot['step_id'], rig_id=snapshot['rig_id'],
            configured_depth_range_m=list(DEPTH_RANGE_M),
            depth_counts={camera: desc['depth_counts'] for camera, desc in snapshot['cameras'].items()},
            snapshot_count=len(self.state['snapshots']), memory_count=len(self.state['memories']),
            snapshot_limit=MAX_SNAPSHOTS, memory_index_limit=MAX_PATCHES,
            snapshots=[dict(episode_id=s['episode_id'], step_id=s['step_id'],
                observation_path=s['observation_path'], arrays_path=s['arrays_path'],
                dense_geometry=s['dense_geometry'], **self._history_flags(s['step_id'], snapshot['step_id']))
                for s in self.state['snapshots'][-3:]],
            memories=[self._memory_view(item, snapshot['step_id']) for item in self.state['memories'][-8:]],
            historical_only=True, current_measurement=False,
            require_reobservation_before_contact=True, object_tracking=False,
            warning=WARNING)

    def _observe(self, packet):
        snapshot, loaded, fingerprint = self._packet(packet)
        changed = False
        if snapshot['step_id'] != self.state['last_step']:
            self.state.update(episode_id=snapshot['episode_id'], rig_id=snapshot['rig_id'],
                last_step=snapshot['step_id'], last_fingerprint=fingerprint)
            self.state['snapshots'] = (self.state['snapshots']+[snapshot])[-MAX_SNAPSHOTS:]
            changed = True
        changed = self._refresh_visual_checks(snapshot, loaded) or changed
        if changed:
            write_json(self.path, self.state)
        return snapshot, loaded

    def summary(self, packet):
        snapshot, _ = self._observe(packet)
        return self._summary(snapshot)

    def remember(self, packet, *, camera, label, pixel, radius=8):
        if camera not in CAMERAS:
            raise InputError('Unknown camera')
        if not isinstance(label, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', label):
            raise InputError('Use a 1..64 character patch label (letters/digits/_/-)')
        if type(radius) is not int or not 1 <= radius <= 32:
            raise InputError('Patch radius must be an integer in 1..32 original pixels')
        try:
            uv = np.asarray(pixel, dtype=float)
        except (TypeError, ValueError) as error:
            raise InputError('pixel must be finite original-resolution [u,v]') from error
        if uv.shape != (2,) or not np.isfinite(uv).all():
            raise InputError('pixel must be finite original-resolution [u,v]')
        snapshot, loaded = self._observe(packet)
        rgb, depth, k, t, valid = loaded[camera]
        h, w = depth.shape
        if np.any(uv < 1) or uv[0] >= w-1 or uv[1] >= h-1:
            raise InputError('Patch center requires a one-pixel image margin')
        u, v = (int(value) for value in np.rint(uv))
        if not (1 <= u < w-1 and 1 <= v < h-1):
            raise InputError('Patch center requires a one-pixel image margin')
        neighborhood = depth[v-1:v+2, u-1:u+2]
        if not valid[v-1:v+2, u-1:u+2].all():
            raise InputError('Patch center has invalid/out-of-range depth (configured 0.1..10 m); preobserve from a valid view')
        if float(np.ptp(neighborhood)) > .02:
            raise InputError('Patch center crosses a depth discontinuity; select a stable surface interior')
        x0, x1, y0, y1 = max(0, u-radius), min(w, u+radius+1), max(0, v-radius), min(h, v+radius+1)
        z = depth[y0:y1, x0:x1].copy()
        mask = valid[y0:y1, x0:x1].copy()
        rows, cols = np.mgrid[y0:y1, x0:x1]
        zvalid = np.where(mask, z, np.nan)
        local = np.stack(((cols-k[0, 2])/k[0, 0]*zvalid,
                          (rows-k[1, 2])/k[1, 1]*zvalid, zvalid), axis=-1)
        xyz = (local @ t[:3, :3].T + t[:3, 3]).astype(np.float32)
        xyz[~mask] = np.nan
        identity = dict(version=VERSION, episode_id=snapshot['episode_id'], step_id=snapshot['step_id'],
            rig_id=snapshot['rig_id'], camera=camera, label=label, sampled_pixel=[int(u), int(v)],
            radius=radius, arrays_sha256=snapshot['arrays_sha256'])
        memory_id = 'nf_'+_digest(identity)[:32]
        for previous in self.state['memories']:
            if previous['memory_id'] == memory_id:
                return self.recall(packet, memory_id=memory_id)
        archive, image = self.patches/(memory_id+'.npz'), self.patches/(memory_id+'.png')
        metadata = dict(identity, memory_id=memory_id, source_observation_path=snapshot['observation_path'],
            source_arrays_path=snapshot['arrays_path'], source_dense_geometry=snapshot['dense_geometry'].get(camera),
            requested_pixel=uv.tolist(), pixel_bounds_xyxy=[x0, y0, x1, y1],
            bounds_convention='exclusive x1,y1; local [v-y0,u-x0] corresponds to original [v,u]',
            original_size_wh=[w, h], patch_shape=[y1-y0, x1-x0, 3],
            frame='environment_origin', units='meters', K=k.tolist(), T_environment_camera=t.tolist(),
            source=copy.deepcopy(snapshot['cameras'][camera]['source']),
            observed_center_environment_position_m=xyz[v-y0, u-x0].tolist(),
            observed_center_optical_depth_m=float(depth[v, u]),
            valid_count=int(mask.sum()), invalid_count=int(mask.size-mask.sum()),
            **self._history_flags(snapshot['step_id'], snapshot['step_id']))
        if archive.exists() or image.exists():
            raise InputError('Immutable patch file already exists without an index entry; refusing overwrite')
        try:
            with archive.open('xb') as stream:
                np.savez_compressed(stream, rgb=rgb[y0:y1, x0:x1].copy(), raw_depth_m=z.astype(np.float32),
                    xyz_environment=xyz, valid=mask, K=k, T_environment_camera=t,
                    metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))
            with image.open('xb') as stream:
                Image.fromarray(rgb[y0:y1, x0:x1]).save(stream, format='PNG')
        except Exception:
            archive.unlink(missing_ok=True)
            image.unlink(missing_ok=True)
            raise
        item = dict(metadata, archive_path=str(archive), image_path=str(image),
                    archive_sha256=sha256(archive), image_sha256=sha256(image))
        self.state['memories'] = (self.state['memories']+[item])[-MAX_PATCHES:]
        self._refresh_visual_checks(snapshot, loaded)
        write_json(self.path, self.state)
        return self._memory_view(item, snapshot['step_id'])

    def recall(self, packet, *, memory_id=''):
        if not isinstance(memory_id, str) or (memory_id and not re.fullmatch(r'nf_[0-9a-f]{32}', memory_id)):
            raise InputError('Use a canonical memory_id returned by near-field remember')
        snapshot, _ = self._observe(packet)
        if not memory_id:
            return self._summary(snapshot)
        item = next((item for item in self.state['memories'] if item['memory_id'] == memory_id), None)
        if item is None:
            raise InputError('Unknown or retired near-field memory_id in this episode index')
        self._load_memory(item)
        return self._memory_view(item, snapshot['step_id'])
