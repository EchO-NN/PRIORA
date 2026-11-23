"""Opt-in, measured aligned camera depth for the teacher; RGB policy is unchanged."""
import os
from pathlib import Path

import numpy as np
from PIL import Image

DEPTH_KEY = 'cam_high_depth_m'
DEPTH_CAMERA = 'cam_high_depth'
DEPTH_RANGE_M = 3.0
CAMERA_ALIASES = dict(cam_high=('cam_high', 'cam_head', 'head_camera', 'top_camera'),
                     cam_left_wrist=('cam_left_wrist', 'left_camera'),
                     cam_right_wrist=('cam_right_wrist', 'right_camera'))
ALL_RGBD_NOTE = (
    'ALL THREE camera inputs are RGBD: cam_high, cam_left_wrist, and cam_right_wrist '
    'are RGB; each immediately following <camera>_depth image is pixel-aligned depth '
    'from that SAME camera at the SAME acknowledged simulation step. Depth is '
    'distance_to_image_plane: optical-axis Z in meters in the individual CAMERA frame, '
    'not world height or Euclidean range. All depth images use the SAME FIXED LINEAR '
    '0.1..10 meter visualization scale: <=0.1 m is white, >=10 m is black; '
    'invalid/nonpositive/nonfinite pixels are magenta. No per-frame normalization. '
    'The decoded grayscale relation is approximately Z = 0.1 + (1-gray/255)*9.9 m; '
    'JPEG quantization means the image is NOT an exact numeric measurement. For precise '
    'depth, read raw_depth_path (.npy float32 meters indexed [v,u]) or observation.npz '
    'key <camera>_depth_m. Raw sensor arrays are not clipped to the visualization range. '
    'Map preview pixels to original_size before sampling; depth and RGB share the '
    'same field of view within each pair. Do not compare different cameras as if they '
    'shared a coordinate frame. Wrist-camera poses move with the arms. The depth is '
    'visible sensor evidence, without object IDs, hidden poses, rewards, or future '
    'states. Pi05 still receives only its original three RGB images and proprioception.'
)
RGBD_NOTE = (
    'Main camera input is RGBD: cam_high is RGB and cam_high_depth is its '
    'pixel-aligned depth from the SAME camera and acknowledged simulation step. '
    'Depth is the simulated sensor distance to the image plane (optical-axis Z), '
    'in meters, not world height or Euclidean range. The depth image uses a FIXED '
    '0..3 meter scale: near is white, far is black, >=3 m is clipped to black, '
    'and invalid/nonpositive/nonfinite pixels are magenta. It is not an RGB '
    'photograph. The JPEG is a visualization, not exact numeric depth; use '
    'raw_depth_path (.npy, float32, meters, indexed [v,u]) or observation.npz '
    'key cam_high_depth_m for exact original-resolution samples. No per-frame '
    'normalization is used. RGB and depth have the same field of view; map '
    'preview pixels to original_size before sampling. Both wrist cameras '
    'remain RGB-only. Use depth as additional visible sensor evidence; it does '
    'not contain object IDs, hidden poses, reward, or future state. Pi05 still '
    'receives the original three RGB images and original proprioception.'
)


def main_camera_input():
    mode = os.environ.get('ROLLOUT_MAIN_CAMERA_INPUT', 'rgb')
    if mode not in ('rgb', 'rgbd', 'rgbd_all'):
        raise ValueError('ROLLOUT_MAIN_CAMERA_INPUT must be rgb, rgbd, or rgbd_all')
    return mode


def depth_cameras():
    mode = main_camera_input()
    return tuple(CAMERA_ALIASES) if mode == 'rgbd_all' else ('cam_high',) if mode == 'rgbd' else ()


def depth_key(camera):
    return camera + '_depth_m'


def depth_range():
    return (0.1, 10.0) if main_camera_input() == 'rgbd_all' else (0.0, DEPTH_RANGE_M)


def sensor_input_note():
    return ALL_RGBD_NOTE if main_camera_input() == 'rgbd_all' else RGBD_NOTE


def configure_depth(cfg):
    """Add requested depth annotators without changing physics/task settings."""
    if not depth_cameras():
        return
    from omegaconf import OmegaConf
    for camera in depth_cameras():
        native = next((k for k in CAMERA_ALIASES[camera] if k in cfg.camera.annotator), None)
        if native is None:
            raise ValueError(f'No explicit {camera} annotator configuration')
        OmegaConf.update(cfg, f'camera.annotator.{native}.distance_to_image_plane_capture',
                         dict(type='distance_to_image_plane', device='cpu'), force_add=True)
    cfg.eval_cfg.observation.vision.depth = True


def measured_depth(camera, rgb_shape):
    if 'depth' not in camera:
        raise ValueError('RGBD requested but native camera depth is missing')
    depth = np.asarray(camera['depth'], dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.shape != tuple(rgb_shape[:2]):
        raise ValueError('RGB and depth resolutions are not aligned')
    if not (np.isfinite(depth) & (depth > 0)).any():
        raise ValueError('Main depth has no valid positive sensor measurements')
    return depth.copy()


def depth_artifact(depth, directory, camera='cam_high'):
    """Save lossless sensor values and a labeled-descriptor visual attachment."""
    directory = Path(directory)
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError('Depth must be an HxW array')
    if camera not in CAMERA_ALIASES:
        raise ValueError('Unknown depth camera: '+camera)
    raw = directory/(depth_key(camera)+'.npy')
    np.save(raw, depth, allow_pickle=False)
    valid = np.isfinite(depth) & (depth > 0)
    near, far = depth_range()
    clean = np.where(valid, depth, far)
    gray = np.rint(255 * (1 - np.clip((clean-near)/(far-near), 0, 1))).astype(np.uint8)
    visual = np.repeat(gray[..., None], 3, axis=2)
    visual[~valid] = [255, 0, 255]
    path = directory/(camera+'_depth.png')
    Image.fromarray(visual).save(path)
    return dict(camera=camera+'_depth', modality='depth', path=str(path),
        paired_rgb_camera=camera, raw_depth_path=str(raw),
        raw_dtype='float32', units='meters', depth_type='distance_to_image_plane',
        original_size=[depth.shape[1], depth.shape[0]], raw_indexing='[v,u]',
        valid_fraction=float(valid.mean()), visualization_range_m=[near, far],
        description=sensor_input_note())
