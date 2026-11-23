"""Deduplicate camera metadata only at the GPT presentation boundary."""
from pathlib import Path

CAMERAS = frozenset(('cam_high', 'cam_left_wrist', 'cam_right_wrist'))
VERSION = 'teacher_camera_summary_v1'
GUIDE = """# Compact camera input

Camera parameters are supplied once per observation, shared by its RGB/depth
pair. images contain camera names, source paths and any codex_preview path,
size and original_size. Paths remain absolute. Images and chronology are not
removed. History frames keep their own episode, step and simulation-time labels;
never apply current wrist extrinsics to a different historical observation.

Displayed K and transform matrices have at most four decimal places. Exact
calibration remains in the archived observation/NPZ; spatial tools use that full
precision. Use these rounded matrices for interpretation, and archived values
for precise geometric calculations.
camera_intrinsics[camera].K is the original-image pinhole K in pixels, calibrated
for original_size_wh. Pixel u is right, v down; optical axes are +X right, +Y down,
+Z forward. For a resized preview with sizes (w,h) and (W,H) original, map pixel
centers by u_original=(u_preview+0.5)*W/w-0.5 and likewise for v. Use original K
and original pixels for raw depth/XYZ queries; do not use K directly on resized
pixels. RGB/depth pairs share original calibration. Camera-to-environment
T_environment_camera is in camera_geometry.cameras[camera]; it maps homogeneous
optical coordinates into environment-origin meters. Wrist transforms update
with measured EEF poses. The reverse transform is its matrix inverse. Missing
calibration remains unknown. Any per-image calibration override applies only
to that image. Fixed mounts, URDF/USD provenance, hashes and full calibration
are in full_camera_metadata_path (or the recorded observation/request JSON).

When depth is supplied, it is optical-axis Z, not world height or Euclidean
range. Depth images use fixed linear 0.1..10 m: near is white, far is black,
invalid is magenta; visualization_range_m records the actual limits. Raw depth
arrays are not clipped. JPEG grayscale is not an exact measurement. Read the
image's raw_depth_path or observation NPZ for exact values. dense_geometry[camera]
gives the original-resolution NPZ path, shape, validity counts and depth_range_m.
Use np.load(path, allow_pickle=False): xyz_environment[v,u,:] is a visible surface
point in environment meters; valid[v,u] is its mask, invalid XYZ is NaN. These
are not grasp/link6 targets. Spatial tools still enforce their validity and
depth-edge checks. Full archive fields/indexing documentation stay in the
recorded metadata; no object identities or future object states are supplied.
"""


def _select(value, fields):
    return {key: value[key] for key in fields if key in value}


def _matrix_preview(value):
    if isinstance(value, list):
        return [_matrix_preview(item) for item in value]
    if isinstance(value, float):
        rounded = round(value, 4)
        return 0.0 if rounded == 0 else rounded
    return value


def _intrinsics(value):
    result = _select(value, ('K', 'original_size_wh', 'distortion_coefficients'))
    if 'K' in result:
        result['K'] = _matrix_preview(result['K'])
    return result


def _geometry(value):
    return {key: _matrix_preview(item) for key, item in value.items()
            if key == 'T_environment_camera'}


def _dense(value):
    return _select(value, ('path', 'shape', 'valid_count', 'invalid_count', 'depth_range_m'))


def compact_camera_packet(value):
    """Copy each observation independently; never edit arrays or raw archives.

    Keep image bytes/paths, temporal identity and runtime validity. Round display
    matrices only; exact calibration stays in the archives. Compact only known
    camera schemas, not arbitrary tool results.
    """
    if isinstance(value, list):
        return [compact_camera_packet(item) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get('memory_tool') == 'get_images':
        return value  # Memory retrieval has its own per-image provenance contract.
    result = {key: item if key == 'images' else compact_camera_packet(item)
              for key, item in value.items()}
    if 'images' not in value:
        return result
    result['images'] = value['images']
    current = [item for item in value['images'] if isinstance(item, dict)
               and not item.get('teacher_history')
               and item.get('paired_rgb_camera', item.get('camera', '').removesuffix('_depth')) in CAMERAS]
    if not current or not any('camera_intrinsics' in item for item in current):
        return result

    # Promote embedded calibration when older replies have no top-level copy.
    intrinsics = dict(value.get('camera_intrinsics', {}))
    geometry = dict(value.get('camera_geometry', {}))
    cameras = dict(geometry.get('cameras', {}))
    dense = dict(value.get('dense_geometry', {}))
    for item in current:
        camera = item.get('paired_rgb_camera', item['camera'].removesuffix('_depth'))
        for field, target in (('camera_intrinsics', intrinsics),
                              ('camera_geometry', cameras), ('dense_geometry', dense)):
            if isinstance(item.get(field), dict):
                target.setdefault(camera, item[field])

    result['camera_intrinsics'] = {c: _intrinsics(v) for c, v in intrinsics.items()}
    if cameras:
        result['camera_geometry'] = dict(
            _select(geometry, ('episode_id', 'step_id', 'frame')),
            cameras={c: _geometry(v) for c, v in cameras.items()})
    if dense:
        result['dense_geometry'] = {c: _dense(v) for c, v in dense.items()}
    result['camera_payload_format'] = VERSION
    metadata_path = (value.get('observation_path')
                     or value.get('environment_state', {}).get('observation_path')
                     or value.get('request_path'))
    if not metadata_path and current[0].get('path'):
        candidate = Path(current[0]['path']).parent / 'observation.json'
        if candidate.is_file():
            metadata_path = str(candidate)
    if metadata_path:
        result['full_camera_metadata_path'] = metadata_path

    images = []
    for item in value['images']:
        if not isinstance(item, dict) or item.get('teacher_history') or not any(item is c for c in current):
            images.append(item)
            continue
        camera = item.get('paired_rgb_camera', item['camera'].removesuffix('_depth'))
        short = _select(item, ('camera', 'path', 'modality', 'paired_rgb_camera',
            'episode_id', 'step_id', 'simulation_time_s', 'original_size',
            'raw_depth_path', 'visualization_range_m', 'valid_fraction'))
        if isinstance(item.get('codex_preview'), dict):
            short['codex_preview'] = _select(item['codex_preview'], ('path', 'size', 'original_size'))
        # Distinct per-image calibration must not silently inherit another view.
        for field, canonical, compact in (('camera_intrinsics', intrinsics, _intrinsics),
                ('camera_geometry', cameras, _geometry), ('dense_geometry', dense, _dense)):
            if isinstance(item.get(field), dict) and compact(item[field]) != compact(canonical[camera]):
                short[field] = compact(item[field])
        images.append(short)
    result['images'] = images
    for field in ('camera_intrinsics_note', 'sensor_input_note', 'active_view_instruction'):
        result.pop(field, None)  # Fixed rules are in the persistent startup prompt.
    return result
