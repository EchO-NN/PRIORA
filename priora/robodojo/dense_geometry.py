"""Dense, original-resolution RGBD surface coordinates for model-side NumPy reads.

Called only when constructing an acknowledged teacher observation, not on every
raw simulator control step. Arrays stay in NPZ files; JSON carries small indices.
"""
import os
from pathlib import Path
import tempfile

import numpy as np

from .camera_geometry import VERSION as GEOMETRY_VERSION, validate_transform
from .camera_intrinsics import CAMERA_ALIASES, validate_k

VERSION = 'dense_rgbd_xyz_v1'
DEPTH_RANGE_M = (.1, 10.)
ARCHIVE_KEYS = ('xyz_environment', 'valid', 'K', 'T_environment_camera',
                'episode_id', 'step_id', 'rig_id', 'camera', 'version')


def build_dense_xyz(depth_m, k, t_environment_camera):
    """Vectorized exact-pixel backprojection, using optical-axis Z in meters.

    Invalid/out-of-range pixels produce NaN triples and False validity. No
    neighborhood smoothing or depth-edge rejection is applied to this map.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2 or any(n <= 0 for n in depth.shape):
        raise ValueError('Dense geometry requires a nonempty original HxW depth array')
    k = validate_k(k)
    t = validate_transform(t_environment_camera)
    valid = np.isfinite(depth) & (depth >= DEPTH_RANGE_M[0]) & (depth <= DEPTH_RANGE_M[1])
    height, width = depth.shape
    v, u = np.indices((height, width), dtype=np.float64)
    # validate_k requires diagonal focal axes and a unit final row. This is
    # inverse(K) [u,v,1] multiplied by optical Z, not normalized ray range.
    clean = np.where(valid, depth, np.nan)
    camera = np.stack(((u-k[0, 2])/k[0, 0]*clean,
                       (v-k[1, 2])/k[1, 1]*clean, clean), axis=-1)
    xyz = (camera @ t[:3, :3].T + t[:3, 3]).astype(np.float32)
    xyz[~valid] = np.nan
    if not np.isfinite(xyz[valid]).all():
        raise ValueError('Dense geometry produced nonfinite coordinates for valid depth')
    return xyz, valid


def _identity(geometry):
    if not isinstance(geometry, dict) or geometry.get('version') != GEOMETRY_VERSION:
        raise ValueError('Dense geometry requires validated rigid camera geometry')
    episode, step, rig = (geometry.get(name) for name in ('episode_id', 'step_id', 'rig_id'))
    if (not isinstance(episode, str) or not episode or type(step) is not int or step < 0
            or not isinstance(rig, str) or not rig):
        raise ValueError('Dense geometry requires an episode, acknowledged step and rig identity')
    if set(geometry.get('cameras', {})) != set(CAMERA_ALIASES):
        raise ValueError('Dense geometry requires all three calibrated cameras')
    return dict(episode_id=episode, step_id=step, rig_id=rig)


def _write_npz(path, **arrays):
    """Publish a complete local archive atomically."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.'+path.stem+'_', suffix='.npz', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_dense_geometry(directory, obs, intrinsics, geometry):
    """Save three XYZ+valid maps and return compact per-camera descriptors.

    Archive indices match the ORIGINAL RGB/depth pixels: [v,u] and C-order flat
    row ``v*width+u``. Preview JPEG pixels must first be mapped to original pixels.
    A valid point describes a visible surface, not an EEF/grasp target.
    """
    identity = _identity(geometry)
    if set(intrinsics) != set(CAMERA_ALIASES):
        raise ValueError('Dense geometry requires original intrinsics for all three cameras')
    prepared = {}
    # Validate all inputs before creating files, so mismatched camera identities
    # cannot leave a seemingly complete three-camera record.
    for camera in CAMERA_ALIASES:
        image = np.asarray(obs[camera])
        depth = np.asarray(obs[camera+'_depth_m'], dtype=np.float64)
        kdesc = intrinsics[camera]
        k = validate_k(kdesc['K'])
        t = validate_transform(geometry['cameras'][camera]['T_environment_camera'])
        if (image.ndim != 3 or image.shape[2] != 3 or any(n <= 0 for n in image.shape[:2])
                or depth.shape != image.shape[:2]
                or list(kdesc.get('original_size_wh', [])) != [image.shape[1], image.shape[0]]):
            raise ValueError(f'{camera}: RGB, depth and original calibration resolutions disagree')
        obs_transform = camera+'_T_environment_camera'
        if obs_transform in obs and not np.allclose(validate_transform(obs[obs_transform]), t, atol=1e-7, rtol=0):
            raise ValueError(f'{camera}: dense map geometry differs from this observation')
        prepared[camera] = (depth, k, t)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    descriptors = {}
    for camera, (depth, k, t) in prepared.items():
        xyz, valid = build_dense_xyz(depth, k, t)
        height, width = depth.shape
        path = directory/(camera+'_xyz_environment.npz')
        _write_npz(path, xyz_environment=xyz, valid=valid, K=k,
                   T_environment_camera=t,
                   episode_id=np.asarray(identity['episode_id']),
                   step_id=np.asarray(identity['step_id'], dtype=np.int64),
                   rig_id=np.asarray(identity['rig_id']),
                   camera=np.asarray(camera), version=np.asarray(VERSION))
        count = int(np.count_nonzero(valid))
        descriptors[camera] = dict(version=VERSION, camera=camera, path=str(path),
            frame='environment_origin', units='meters', dtype='float32',
            shape=[height, width, 3], original_shape=[height, width],
            original_size_wh=[width, height], xyz_key='xyz_environment', valid_key='valid',
            archive_keys=list(ARCHIVE_KEYS), indexing='xyz_environment[v,u,:]',
            flatten_index='row=v*W+u; xyz_environment.reshape(-1,3)[row]',
            invalid_value='NaN in all three coordinates; valid[v,u]=False',
            valid_rule='finite optical-axis depth Z and 0.1 <= Z <= 10.0 meters',
            depth_range_m=list(DEPTH_RANGE_M), valid_count=count, invalid_count=int(valid.size-count),
            source_depth_key=camera+'_depth_m', **identity,
            note='Original-resolution visible-surface coordinates from raw optical depth, intrinsics and current rigid extrinsics. '
                 'Not an EEF/grasp target. No depth-edge filtering in this dense map; spatial_query still checks its raw-depth neighborhood. '
                 'Map preview pixels to original pixels before indexing. Load with np.load(path, allow_pickle=False).')
    return descriptors
