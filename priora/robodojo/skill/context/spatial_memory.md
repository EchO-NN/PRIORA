# Rigid EEF / three-camera spatial memory (v3)

Use the RGBD, measured EEF poses, camera_intrinsics, camera_geometry and
dense_geometry explicitly supplied to this episode. The host computes camera
geometry from the configured URDF fixed joints, camera mounting offsets and
optical-axis convention. These
robot/sensor calibration inputs are authorized. Do not query private scene state,
object poses, reward internals or additional simulator transforms yourself.
No dedicated calibration moves or per-episode affine fitting are required.
Use the existing camera views, without moving an arm solely to improve a view.
In demonstration tasks, remain still until the demonstrator returns.

## Near-field depth validity and historical local observations

**Configured valid depth is optical-axis Z from 0.1 to 10 m, not Euclidean
distance to the camera or distance to the EEF.** A surface closer than 0.1 m
cannot produce a valid numeric point through spatial_query or dense XYZ under
this configuration. RGB can still show that surface, and its recorded raw depth
may contain a number. The configured filter is not proof of an optical clipping
plane or a physical camera blind zone. Do not infer an absent object, empty
gripper or grasp failure solely from invalid depth. Missing values, far-range
values, occlusion and depth discontinuities can also make a lookup unusable.
No failed lookup authorizes inventing metric depth, a fingertip offset or a
rigid landmark correspondence.

Before moving close enough to invalidate depth or occlude a task-relevant
feature, retain a local observation while it is visible and numerically valid.
This is observation memory, not a dedicated calibration move or a new reason
to take over. The host automatically indexes the last 32 acknowledged archived
three-camera observations with their RGB/depth/XYZ pointers and source step.
It does not infer or track object identity for you. On each start/execute,
spatial_status.near_field_memory contains current per-camera depth-range
statistics, recent observation pointers and remembered local-patch entries.
The compact summary shows the latest three observations and eight patches;
the full index retains up to 64 patch entries. workspace.json supplies
near_field_memory_version=near_field_memory_v1 and near_field_memory_path,
which points to controller/spatial/near_field/index.json. Patch files live
under the same near_field directory. Preserve these host-owned records.

Use the two observation-only memory tools with the latest acknowledged
observation_path:

- spatial_remember(observation_path, camera, label, pixel=[u,v], radius=N)
  saves a named local patch centered on original RGB pixel (u,v), with integer
  radius from 1 to 32 pixels. Give a concrete physical label, such as
  right_lower_corner_before_approach: labels contain 1–64 ASCII letters,
  digits, underscores or hyphens, with no spaces. The label is your
  identification, not host-verified segmentation. Convert preview pixels first.
  The center and
  its raw-depth 3x3 neighborhood must be valid and sufficiently continuous;
  invalid pixels elsewhere in the patch remain masked. A successful save
  records observed surface geometry, not a hidden pinch point or grasp proof.
  Keep the returned memory_id in SPATIAL_MEMORY.md. If this target is already
  too close or occluded, inspect the archived pointers; do not fabricate a
  valid current memory from an unrelated earlier pixel.
- spatial_recall(observation_path, memory_id) retrieves the saved patch and
  its source metadata. Use memory_id="" to inspect the compact current index
  summary; read near_field_memory_path for all retained index entries.
  A recall is ALWAYS historical, including immediately after capture; check
  source step, age_control_steps and stale_after_controls along with any
  intervening motion/contact. Numerical age is not a guarantee that the scene
  stayed unchanged. This tool does not propagate a surface through physics,
  certify present visibility or promote old XYZ to a current control target.

The host additionally checks each retained patch against current RGB: it
reprojects the old environment-frame points into the current camera using the
known K and rigid camera transform, then compares aligned local RGB samples.
This compensates for camera motion under a stationary-surface hypothesis; it
does not track a moving or deforming object. Inspect the returned visual status:

- same_observation: the patch belongs to this very observation. It still
  represents an observed surface, not proof of grasp, pinch geometry or contact.
- consistent: enough aligned samples show only small RGB changes. The patch
  may serve as a historical planning hint, never a verified current 3D target
  or proof that a grasp is stable.
- invalidated_rgb_change: sufficiently supported large RGB changes permanently
  invalidate this memory for planning. Its original archive remains available
  for audit; to use a new scene observation, capture a valid new patch with
  spatial_remember and use its new memory_id. Later similar colors do not
  reactivate the old memory.
- unverified: occlusion, out-of-frame projection, insufficient aligned coverage
  or another unresolved comparison prevents verification. Do not use this
  memory as a planning hint. Occlusion alone is not evidence that the object
  moved, and this status does not establish grasp failure.

Do not use invalidated or unverified patches for planning even when their XYZ
files still exist. Uniform white cloth can move without a distinctive RGB
change, while lighting, surface orientation and rendering changes can alter RGB
without matching physical deformation. RGB consistency is therefore limited
evidence, not a replacement for current visual reacquisition and depth checks.
Record the current visual status and any invalidation in SPATIAL_MEMORY.md.

Maintain the local-observation table in SPATIAL_MEMORY.md: target identity,
memory_id, source step/camera, last visual confirmation, intervening changes
and the next required check. A visually consistent prior patch may help
recognize a corner, its local
shape and the approach context when current metric depth is missing. Before
contact or resuming manipulation based on that memory, visually reacquire the
feature in current observations and reassess whether its geometry changed.
After cloth folding, dragging, grasping or release, old cloth coordinates may
already be wrong. Wrist motion updates camera geometry; it does NOT carry an
old cloth/object surface along with the wrist, even if a grasp was intended.
Keep scene patches distinct from spatial_record's confirmed rigid gripper
landmarks, whose reuse is guarded by opening state. A cloth corner or held
object must never be registered as a rigid robot surface.

If the current lookup fails, first inspect the other current views and original
RGB, checking depth validity and identity. An existing valid alternative view
may resolve the geometry. This memory adds no movement permission; corrective
motion requires the outcome/intent gate. Do not lower the depth threshold, edit recordings, extrapolate
deformation or bypass numerical reachability to manufacture an answer.

## Coordinate contract

Let T_A_B transform a homogeneous point from frame B into frame A; matrices act
on column vectors. Every EEF pose refers to link6 in environment-origin meters,
with quaternion wxyz. The camera optical axes are +X right, +Y down, +Z forward.
For either wrist camera, the host composes each observation's measured pose:

    T_env_camera(t) = T_env_eef(t) @ T_eef_camera

T_eef_camera is the fixed rigid mounting transform, including the URDF chain,
configured camera offset and conversion to the optical frame. T_env_eef changes
with the current measured robot pose; the URDF and rigid mounting matrix do not.
The head camera has a fixed T_env_camera for its configured episode mount.
In the full archived camera_geometry, version is rigid_camera_geometry_v1, with episode_id, step_id,
rig_id and a cameras map keyed by the three camera names. Each entry supplies
T_environment_camera and may supply its inverse T_camera_environment. The fixed
T_eef_camera is present for wrist cameras and null for the head. parent_frame is
environment, left_eef or right_eef; source records the calibration provenance.
GPT tool replies summarize this as episode_id, step_id, frame and each camera's
T_environment_camera, with at most four decimal places. Inverse/fixed-mount
matrices, full precision and provenance are available through
full_camera_metadata_path, not repeated on every image. Use archived values or
spatial tools for precise calculations.
The host locks the episode's rig_id, K and fixed mounts and recomposes wrist
transforms from each measured EEF pose.
Use this observation's camera_geometry, not a wrist transform from an earlier
observation, a planned target pose, or an assumed identity transform.

Each original RGB/depth pair shares K. GPT replies supply original K once per
camera. Convert resized JPEG pixel centers to original coordinates using its
codex_preview size/original_size and the startup camera guide; do not apply
original K directly to resized pixels. With raw optical-axis depth Z:

    p_camera = Z * inverse(K) @ [u, v, 1]
    [p_env, 1] = T_env_camera @ [p_camera, 1]
    [p_eef, 1] = inverse(T_env_eef) @ [p_env, 1]

A reconstructed point is the visible surface at that pixel, not necessarily the
object center, cloth corner behind an occluder, link6 origin or pinch point.
Colorized depth previews and JPEG intensities are not metric depth. Use original
RGB pixel coordinates and the raw depth array through the spatial tools.
The host handles transforms and depth validation; do not estimate metric depth
by reading preview colors or substitute a guessed focal length.

## Precomputed dense coordinates for every original RGB pixel

At each GPT observation boundary, the host precomputes a point array for each
RGBD camera from that observation's raw depth, K and current rigid transform.
The full archived dense_geometry supplies each camera's NPZ descriptor with version
dense_rgbd_xyz_v1, path, shape=[H,W,3], original_size_wh=[W,H], units="meters",
frame="environment_origin", xyz_key="xyz_environment", valid_key="valid" and
pixel indexing. Its episode_id, step_id and rig_id identify the source observation
and calibration. GPT replies retain only path, shape, validity counts and
depth_range_m; the rest of the fixed file-format documentation is supplied here
and the source metadata is available on demand. These are recorded numeric observations
available to your normal Python tools; the model does not consume a native
point-cloud embedding or a huge list of floats in the conversation.

Each file contains xyz_environment, float32 with shape [H, W, 3], and valid, bool
with shape [H, W]. xyz_environment[v, u] is the [X, Y, Z] point in environment-origin
meters for the original RGB pixel (u, v). Invalid entries are NaN and have
valid[v, u] == false. Keep the validity mask; do not replace NaN with zero or
interpret an invalid pixel as a point at the origin. A valid depth sample still
represents only the surface visible from this camera, not a hidden grasp site.

Use np.load(path, allow_pickle=False) to inspect these files. In this example,
observation is the current observation packet and camera is the selected view:

```python
import numpy as np

path = observation['dense_geometry'][camera]['path']
with np.load(path, allow_pickle=False) as data:
    xyz = data['xyz_environment']
    valid = data['valid']
    point = xyz[v, u] if valid[v, u] else None
    roi_xyz = xyz[v0:v1, u0:u1]
    roi_valid = valid[v0:v1, u0:u1]
    flat_xyz = xyz.reshape(-1, 3)
    flat_valid = valid.reshape(-1)
```

The flattened row index is v * W + u; array indexing is [v, u], never [u, v].
Coordinates correspond to the original RGB resolution, not the resized JPEG or
an RGB/depth side-by-side panel. If selecting a pixel in a JPEG preview, first
apply that attachment's preview_to_original_pixel mapping; for a resize factor
sx, u_original = (u_preview + 0.5) / sx - 0.5, and likewise for v with sy. Use the
nearest valid integer original pixel when indexing, check bounds and retain the
actual sampled pixel. Do not silently use the preview K with original pixels.

Reading points or regions with normal Python is allowed. Reuse the precomputed
array instead of rebuilding a whole image's XYZ matrix on every query, and print
only the relevant points or summaries. For a selected control-relevant point,
spatial_query uses the cached coordinates and additionally checks the raw-depth
3x3 neighborhood for invalid values and discontinuities. The per-pixel valid
mask alone does not certify this neighborhood, visibility of another feature,
scene clearance, reachability or grasp stability. Use the latest observation's
file; a previous image's pixel index or wrist viewpoint is not current evidence.

## Observation-only spatial tools

After robodojo_start, inspect the three views, measured link6 poses and supplied
calibration. Initialize SPATIAL_MEMORY.md with this episode's known geometry,
visible landmark identities and remaining uncertainties. Rigid geometry is
available immediately when the provided calibration is valid. Pixel queries
need no landmark registration, training correspondences or calibration motion.

spatial_query(observation_path, camera, arm, landmark, pixel=[u,v]) retrieves the
precomputed surface point at an original RGB pixel and returns it in camera,
environment and both EEF frames. It validates the aligned raw depth and identifies
the nearest sampled raw-depth pixel, checking its 3x3 neighborhood for depth
discontinuities. This replaces the old landmark-specific "link6-equivalent"
affine estimate. Use the latest acknowledged observation.
Camera names are cam_high, cam_left_wrist and cam_right_wrist; arm is left/right.
Do not treat a depth discontinuity, invalid depth or out-of-image query as a
numerical grasp target. Reinspect the source image when the sampled surface is
ambiguous.

spatial_query(..., pixel=[]) always projects the selected arm's measured link6
origin as a kinematic reference. It does NOT claim that the physical origin is
visible, lies on the rendered gripper surface or has matching surface depth.
If the named landmark was previously registered, its separate rigid-surface
projection may also be returned, but only at the same gripper opening. Read the
reference labels: the link6 origin and registered surface point are different
quantities. Projection alone does not prove visibility or absence of occlusion.

Call spatial_record(observation_path, camera, arm, landmark, pixel=[u,v]) only
when the named surface point is confidently identifiable. Registration is
optional and is not required for pixel/depth queries. It records raw-depth
geometry and can bind an observed rigid robot landmark to the selected arm's
local link6 frame; it does not learn or refit camera extrinsics. Use a stable
short landmark name and state which physical point it names. Do not bind a cloth
point, held object, shadow or silhouette as a rigid robot landmark. A finger
surface can move relative to link6 as the opening changes; do not assume its
old offset remains valid across such changes. Never invent correspondence or
visibility to agree with a projected reference.

All four spatial tools do not move, reset, infer pi05, consume simulation steps or advance
the start -> infer -> execute contract. They do not change the takeover gate or
authorize a grasp, correction or calibration motion.

## Program-managed geometry and evidence

Program version is rigid_spatial_v3. The host stores numeric state and events
under controller/spatial/; workspace.json supplies the read-only state/status
paths. Every start/execute result includes spatial_status; the automatic summary
provides all camera transforms and projections of both EEF origins in each view.
Those origin projections are references, not labeled visible surface pixels.
Consult supplied geometry and status when using a query, and preserve any
diagnostic uncertainty.

Current measured EEF poses update the wrist-camera transforms automatically.
Fixed mounting transforms are reused without recurring GPT matrix fitting or
the old affine sampling-domain and expiry rules. Missing or invalid supplied
calibration is an error, not permission to invent a transform or fall back to an
old fit.
If visual evidence contradicts geometry, inspect coordinate/frame selection,
pixel resizing, depth validity, landmark identity and mount assumptions. Report
the discrepancy instead of editing recorded geometry or averaging it away.

A geometry lookup does not establish the pinch offset, feasible approach angle,
scene clearance, cloth capture or successful grasp. Preserve the distinction
between a visible point, a registered robot surface and the commanded link6
pose. Existing target-distance, numerical reachability, measured-arrival and
native episode limits still apply. No hidden object truth or physics prediction
is added by this calibrated coordinate conversion.

## Concise working notes

SPATIAL_MEMORY.md records landmark identity, current references, observed
correspondences and unresolved offsets or occlusions. SPATIAL_PROMPT.md is concise
operational guidance, not an editable system prompt. Keep task progress in
NOTES.md. Update notes when evidence or guidance changes, not on every automatic
pose update. Start every episode fresh; rigid robot/sensor configuration may be
reused by the host, while prior-episode task evidence must not be imported.
Do not edit host source, prompts, numeric state or recorded observations.
Mark superseded hypotheses as obsolete.
