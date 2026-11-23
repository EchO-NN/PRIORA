# Episode spatial memory

Status: awaiting first observation. Episode ID: unknown.
Numeric authority: workspace.json spatial_numeric_state_path / spatial_status_path.
Program version: rigid_spatial_v3. Use supplied camera_intrinsics and
camera_geometry; configured rigid transforms need no episode-specific affine fit.
Dense numeric observations: the current dense_geometry map points to per-camera
NPZ files with xyz_environment [H,W,3] float32 and valid [H,W] bool.
Historical local patches: workspace.json near_field_memory_path, version
near_field_memory_v1; current range diagnostics and recent entries appear in
spatial_status.near_field_memory. This is separate from rigid gripper landmarks.

## Near-field warning and saved local observations

Valid metric depth is optical-axis Z in [0.1, 10] m. Below 0.1 m, current numeric
queries/XYZ are invalid even if RGB remains visible or recorded raw depth has a
value; this is not proof of a physical camera blind zone. Save useful visible
patches with spatial_remember BEFORE approaching or occluding them. Recall is
always historical: check its source step/age and intervening contact, reacquire
the feature in current views before contact, and never use old cloth/object XYZ
as current geometry or carry that scene memory with a moving wrist transform.

| Memory ID | Target / physical identity | Source step | Camera | Last visually confirmed | RGB visual status | State changed since capture? | Required recheck |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | none recorded | unknown | unknown | unknown | unverified | unknown | Capture a useful valid patch before close approach |

The host compares RGB after reprojecting old XYZ into the current view.
same_observation/consistent permit only historical planning hints, not current
3D or stable-grasp proof. invalidated_rgb_change permanently invalidates a
memory for planning; keep its audit record and save a new valid patch/new ID.
unverified (including occlusion/out-of-frame/low coverage) is also unusable for
planning even if its coordinate file exists. Occlusion is not proof of object
motion. Uniform cloth, lighting and orientation limit RGB-change evidence.

spatial_recall(observation_path, memory_id="") lists retained evidence. Failed
depth lookup does not authorize guessed depth, invented rigid offsets or a new
motion; inspect another view and preserve the existing control gate.

## Coordinate references and landmark identities

EEF reference: link6, environment-origin meters, quaternion wxyz.
Camera optical axes: +X right, +Y down, +Z forward.
Wrist geometry: T_env_camera = T_env_eef @ T_eef_camera; current measured EEF
updates are program-managed. Head geometry is fixed for its configured mount.
Pinch offset and visible-surface offsets: do not assume zero. A projected link6
origin is a reference, not a claim of a visible robot surface at that pixel.

| Camera | Target arm / reference | Landmark name and physical identity | Visibility | Observed offset / uncertainty |
| --- | --- | --- | --- | --- |
| cam_high | unknown | unknown | unknown | unknown |
| cam_left_wrist | unknown | unknown | unknown | unknown |
| cam_right_wrist | unknown | unknown | unknown | unknown |

## Geometry and evidence limits

Use the current observation's supplied K and rigid matrices; distinguish original
pixels from resized JPEG pixels. Raw-depth queries reconstruct the visible
surface in the environment and both EEF frames. Precomputed arrays use original
RGB [v,u] indexing, or flattened row v*W+u, in environment-origin meters.
Check valid and NaN, convert preview pixels before indexing and use current files.
They do not identify a hidden
surface, object center or valid grasp target. Local robot landmark offsets need
correct identity and appropriate rigidity; finger opening can change an offset.
Projection alone does not establish visibility, contact or collision clearance.

## Evidence changes and superseded hypotheses

None. Record substantive evidence changes without duplicating numeric matrices.
Automatic measured-pose updates do not require rewriting these notes. Report any
geometry discrepancy; do not fit an affine correction or edit host state.

## Current operational guidance

None confirmed. Keep SPATIAL_PROMPT.md concise and consistent with observed evidence.
