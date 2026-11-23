"""Explicit teacher camera calibration; never reads camera/world transforms."""
import numpy as np

CAMERA_ALIASES = {
    'cam_high': ('cam_high', 'cam_head', 'head_camera', 'top_camera'),
    'cam_left_wrist': ('cam_left_wrist', 'left_camera'),
    'cam_right_wrist': ('cam_right_wrist', 'right_camera'),
}
VERSION = 'known_camera_intrinsics_v1'
INTRINSICS_NOTE = (
    'Known camera intrinsics are supplied for cam_high and BOTH wrist cameras. '
    'Each image descriptor camera_intrinsics.K is calibrated for its original_size_wh, '
    'not the resized JPEG. K = [[fx,0,cx],[0,fy,cy],[0,0,1]] in pixel units; '
    'u increases right and v down, using continuous zero-based image coordinates. '
    'The optical CAMERA frame has +X right, +Y down and +Z forward. With the same '
    'camera raw optical-axis depth Z in meters, p_camera = Z * inverse(K) * [u,v,1]. '
    'Read exact raw depth from raw_depth_path/observation.npz, not JPEG grayscale. '
    'A resized attachment has its own codex_preview.camera_intrinsics.K and an '
    'explicit preview_to_original_pixel mapping; convert preview pixels before '
    'sampling original depth or calling spatial_record/spatial_query. RGB and '
    'its paired depth use the same original K. Intrinsics do NOT give a camera '
    'position/orientation or a camera-to-EEF/world transform by themselves. When '
    'camera_geometry is supplied, its explicit fixed URDF/mount calibration and '
    'current measured EEF composition provide the camera-to-environment matrices. '
    'Use only matrices matching this observation episode/step; wrist cameras move '
    'with their arms. If camera_geometry is absent, extrinsics remain unknown. '
    'A visible surface point is not automatically a link6/pinch target: fingertip '
    'offsets, grasp orientation, contact and clearance remain separate.'
)


def key(camera):
    return camera + '_intrinsics_K'


def validate_k(matrix):
    k = np.asarray(matrix, dtype=np.float64)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1]) or abs(k[0, 1]) > 1e-9 or abs(k[1, 0]) > 1e-9):
        raise ValueError('Invalid pinhole camera intrinsics; refusing invented calibration')
    return k


def descriptor(matrix, size):
    k = validate_k(matrix)
    width, height = map(int, size)
    if width <= 0 or height <= 0:
        raise ValueError('Invalid calibrated image resolution')
    return dict(schema=VERSION, K=k.tolist(), original_size_wh=[width, height],
        fx=float(k[0, 0]), fy=float(k[1, 1]), cx=float(k[0, 2]), cy=float(k[1, 2]),
        units='pixels', model='pinhole', distortion_coefficients=[0., 0., 0., 0., 0.],
        pixel_convention='continuous zero-based u right, v down',
        optical_frame='+X right, +Y down, +Z forward',
        source='runtime Isaac camera focal/aperture/resolution; checked against camera.get_intrinsics_matrix()',
        extrinsics_provided=False)


def capture(env, images):
    """Read the live sensor after reset; no rendering, stepping or pose reads."""
    manager = env.camera_manager
    names = list(manager.camera_names[0])
    arrays, descriptions = {}, {}
    for name, aliases in CAMERA_ALIASES.items():
        matches = [i for i, native in enumerate(names) if native in aliases]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one native camera for {name}')
        camera = manager.cameras[0][matches[0]]
        if camera.get_lens_distortion_model() != 'pinhole' or camera.get_projection_mode() != 'perspective':
            raise ValueError(f'{name}: only an undistorted perspective sensor is supported')
        # The SDK K assumes centered apertures. Reject nonzero offsets rather
        # than silently publish a centered principal point for an off-axis lens.
        offsets = [camera.prim.GetAttribute(field).Get() for field in
                   ('horizontalApertureOffset', 'verticalApertureOffset')]
        if any(value is None or not np.isfinite(value) or abs(value) > 1e-9 for value in offsets):
            raise ValueError(f'{name}: nonzero/unknown aperture offset is unsupported')
        width, height = map(int, camera.get_resolution())
        if np.asarray(images[name]).shape[:2] != (height, width):
            raise ValueError(f'{name}: RGB shape differs from live calibrated resolution')
        focal, horizontal, vertical = [float(value) for value in (
            camera.get_focal_length(), camera.get_horizontal_aperture(), camera.get_vertical_aperture())]
        if not all(np.isfinite(v) and v > 0 for v in (focal, horizontal, vertical)):
            raise ValueError(f'{name}: missing or invalid runtime optical parameters')
        k = validate_k([[width*focal/horizontal, 0, width/2],
                        [0, height*focal/vertical, height/2], [0, 0, 1]])
        sdk = camera.get_intrinsics_matrix(device='cpu')
        if hasattr(sdk, 'detach'):
            sdk = sdk.detach().cpu().numpy()
        if not np.allclose(k, sdk, rtol=1e-6, atol=1e-5):
            raise ValueError(f'{name}: runtime optics and SDK intrinsics disagree')
        arrays[key(name)] = k
        descriptions[name] = descriptor(k, [width, height])
    return arrays, descriptions


def from_observation(obs, *, required=False):
    present = [name for name in CAMERA_ALIASES if key(name) in obs]
    if not present and not required:
        return {}
    if set(present) != set(CAMERA_ALIASES):
        raise ValueError('All three calibrated camera matrices are required')
    return {name: descriptor(obs[key(name)], [obs[name].shape[1], obs[name].shape[0]]) for name in present}


def preview_calibration(original, original_size, preview_size):
    """Pillow resize maps pixel centers with the half-pixel convention."""
    if list(original['original_size_wh']) != list(original_size):
        raise ValueError('Camera calibration resolution differs from image attachment')
    sx, sy = np.asarray(preview_size, dtype=float) / np.asarray(original_size, dtype=float)
    transform = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]])
    k = transform @ validate_k(original['K'])
    return dict(K=k.tolist(), size_wh=list(preview_size), units='pixels',
        fx=float(k[0, 0]), fy=float(k[1, 1]), cx=float(k[0, 2]), cy=float(k[1, 2]),
        pixel_convention='continuous zero-based u right, v down',
        preview_to_original_pixel=dict(scale_xy=[float(1/sx), float(1/sy)],
            offset_xy=[float((1/sx-1)/2), float((1/sy-1)/2)],
            formula='u_original=(u_preview+0.5)/sx-0.5; v_original=(v_preview+0.5)/sy-0.5'),
        interpolation_note='depth JPEG uses nearest resampling; sample original raw depth for metric values')
