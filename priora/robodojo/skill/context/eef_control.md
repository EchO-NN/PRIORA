# RoboDojo / dual ARX X5 embodiment contract

You are the autonomous mixed-policy agent; the implementation assistant does
not choose online corrections. Keep normal shell, calculation, image viewing,
notes and analysis tools. The three blocking rollout tools are additive.
Use the colleague outcome/intent correction gate and original task instruction.
Do not move an arm solely for a new camera viewpoint. Use the existing views;
uncertainty alone does not authorize an EEF correction.
There is no rollback, alternate controller connection, object truth, reward
query or future simulation for planning. Native success/timeout is returned.

## Differences from RoboLab (do not use Panda geometry here)

- `robot_profile=robodojo`, simulator RoboDojo, robot dual ARX X5.
- Policy is the official RoboDojo-finetuned OpenPI/JAX checkpoint, configuration
  `pi05_base_aloha_full_sim_arx-x5_seed_0`, normalizer `arx_x5_sim`.
- Three image attachments: `cam_high`, `cam_left_wrist`, `cam_right_wrist`.
  Compare all relevant views; native analysis tools can crop and reopen them.
- Proprio/action order is `[left joints 1..6, left opening, right joints 1..6,
  right opening]`. Native opening is continuous **0=closed, 1=open**. The
  student proposal retains continuous opening (only clipped to native [0,1]).
  Native gripper proprio is the last controller opening command, not a
  measured finger gap or proof of grasp. Arm joints and EEF poses are measured.
- Every inference returns all **50** target steps and both-arm FK poses.
  `student` chooses 1–30 steps before observing/inferencing again; the remaining
  proposal suffix is discarded. `eef` chooses 1–30 total controls, either toward
  one absolute pose or an ordered schedule of absolute pose nodes, with
  measured-state IK after every control. `edit` remains 1–5.
  Thirty controls are 1.2 simulated seconds, without intermediate GPT review;
  select fewer near contact or uncertain geometry. Native termination can
  stop either mode early. This does not allow cumulative 30-times-5-cm targets.
- `current_eef` has `left` and `right`; each trajectory row has both arms.
  Each pose refers to that arm's **link6 in environment-origin coordinates**,
  meters and `rpy_deg` (roll/pitch/yaw degrees, extrinsic xyz). GPT tool replies
  show at most three decimal places and omit duplicate `quaternion_wxyz`.
  Recorded JSON/NPZ keep full-precision quaternions in **wxyz** order. Both arms
  share these axes, not two rotated root frames. The robot-only FK is checked
  against actual measured link poses.
- Supplied `camera_intrinsics` and `camera_geometry` authorize calibrated
  pixel/depth conversion using configured robot and sensor geometry. Let T_A_B
  map frame B into frame A: wrist T_env_camera = T_env_eef @ T_eef_camera, with
  the fixed mount derived from the URDF, configured offsets and optical axes.
  The host updates T_env_eef from the current measured pose; the head-camera
  transform is fixed for its mount. Use the latest observation's matrices.
  A raw-depth surface point in environment coordinates is not necessarily the
  desired link6 target: account for the contact/pinch offset and orientation.
  `spatial_query` can project a link6 reference without claiming its origin is
  visible, and can also project a separately registered rigid landmark at the
  same gripper opening. Do not substitute
  surface depth at an occluded reference pixel for that reference's true depth.
- Each current RGBD view also has a precomputed `dense_geometry` NPZ file:
  `xyz_environment` is float32 [H,W,3] in environment-origin meters, indexed
  by original RGB [v,u]; `valid` is bool [H,W], with NaN at invalid XYZ pixels.
  Flattened row v*W+u matches original pixel (u,v). Convert JPEG-preview pixels
  before indexing, preserve validity, and use the latest observation's files.
  Native Python point/ROI reads are permitted; spatial queries use these cached
  points while retaining raw-depth neighborhood checks. A valid surface point
  still needs a separate contact/pinch offset and feasible link6 approach pose.
- Configured valid depth is optical-axis Z in [0.1, 10] m. A closer surface has
  invalid numeric query/XYZ geometry even if RGB remains visible or raw depth
  contains a value; this is not proof of a physical camera blind zone. Before
  approach/occlusion, spatial_remember can retain a labeled local RGB/depth/XYZ
  patch. spatial_recall always returns historical evidence with source step and
  age, never a current grasp target. Reacquire the feature in current views
  before contact; deformation/contact can invalidate old scene coordinates.
  Check aligned-RGB visual status: invalidated_rgb_change and unverified
  memories must not guide planning even when their XYZ files remain. Large
  supported changes permanently invalidate the old memory; capture a new valid
  patch. consistent is only a historical hint, not current geometry or grasp
  proof. Occlusion/low coverage is unverified, not proof of object motion.
  Do not carry cloth/object patches with wrist transforms or register them as
  rigid gripper landmarks. Check another existing view when depth fails;
  corrective motion still requires the outcome/intent gate and limits. See
  context/spatial_memory.md and spatial_status.near_field_memory.
- `target` for `eef` has `left` and `right`, each containing `position`,
  `quaternion_wxyz`, `gripper_closed` (true means closed, converted to opening=0).
  This output schema is unchanged. Convert Euler angles with
  `Rotation.from_euler('xyz', rpy_deg, degrees=True).as_quat()` (SciPy returns
  xyzw; reorder to wxyz), or use the recorded full-precision pose to hold an
  arm stationary. Never put Euler angles into the quaternion field.
  Both must be explicit. To leave an arm stationary, give its current measured
  pose and intended gripper command. With nonempty `waypoints`, top-level
  `target` exactly mirrors the last node's target; execution follows the nodes,
  not that final target throughout. Every node requires both arms explicitly.
  Translation has no fixed distance cap, either per node or across the sequence.
  Every target must still pass numerical reachability within its allocated steps.
  Per arm, every node's rotation stays <=0.35 rad from the initial measured pose
  AND from the preceding node.
- `edit` has `left` and `right`, each the existing `delta_position`,
  `delta_rotation_vector`, `gripper=keep/open/closed`. Offsets are in the common
  environment frame. Finite position offsets have no fixed distance cap;
  rotation-vector magnitude stays <=0.35 rad. Actual motion uses bounded local IK.
  A zero/keep edit leaves that arm's student target unchanged;
  it does NOT freeze that arm. `keep` preserves the continuous proposal opening.
- EEF corrections use local damped least-squares IK, recomputed after every
  real control ACK. Task increments <=2 cm/0.1 rad and each joint increment
  <=0.05 rad. No cuRobo planning or teleportation. A requested target is not
  evidence of arrival/contact; inspect after-execution RGB and measured pose.
- Do not reuse DROID's +0.1311 m pad offset. X5's native config reports
  `gripper_bias=0.145` but that scalar alone does not establish an offset axis;
  derive grasp geometry from robot geometry and observations if needed.
- Native control observations are at 25 Hz (0.04 s); physics steps internally
  use the upstream interpolation/controller. Codex observes at chunk boundaries,
  not each physics substep. All executed control observations are recorded.

## EEF nodes with numerical preflight and measured arrival checks

For a multi-phase correction, provide `waypoints` as an ordered array of
`{"label": "approach", "steps": 8, "target": {"left": ..., "right": ...}}`.
Labels are nonempty English descriptions, at most 80 characters; they do not
trigger implicit operations. Each `steps` is a positive integer. Their sum is
the top-level `steps`, must be at most 30 and must fit in the remaining native
episode budget. For a single fixed target use
`waypoints=[]`; non-EEF modes also use `waypoints=[]`. Old recordings without
this field retain the same fixed-target interpretation. All hybrid EEF decisions
now undergo the preflight and arrival checks below: `waypoints=[]` or an omitted
list is checked as a single node, so the older format cannot bypass these checks.
This preserves input format compatibility, not unconditional admission of an
older step allocation. No new model-authored response fields are required.
Student/edit execution and the separate GPT-only baseline are unchanged.

Before the first action, the controller numerically checks the ENTIRE sequence
starting from the current measured joint positions. It uses the same robot-only
bounded damped least-squares IK, advancing predicted joint positions through each
node's allocated number of controls. Every node must end within 0.002 m position
and 0.02 rad rotation of its requested target for BOTH arms, with joint limits
respected. If any node fails, or the total allocated controls exceed the remaining
native budget, zero actions are executed and the decision is returned for GPT to
adjust targets, orientations or step allocation. This preflight does not advance
physics or call a model. It is a finite-budget numerical feasibility check under
the current robot model, not an additional simulation, collision planner or proof
that a failed target is impossible to reach by any other method.

At each control, the controller selects the current node, recomputes bounded
IK from measured robot state, sends that node's gripper command and awaits the
actual control ACK. The node's `steps` is its maximum allocation: if it reaches
the target early, it keeps holding the same target through the end of that
allocation. At the node boundary, BOTH actual measured EEF poses must be within
0.002 m / 0.02 rad of their targets before the next node can start. Otherwise
later nodes are immediately cancelled, and a fresh observation plus the partially
executed history returns to GPT for replanning. No automatic extension of the
allocation, unrequested retry, later closure or later lift is allowed. Native
termination always stops execution immediately.

Gripper commands remain active from the FIRST control of their node and run
concurrently with any arm movement in that same node. To guarantee the ordering
"approach open, then close", use a separate open-gripper approach node and a
same-pose closed-gripper grasp node. A mixed movement/closure node does NOT wait
for pose arrival before starting closure. A wait node repeats the preceding
target AND gripper state: it holds that command for real simulation controls
rather than sleeping in wall-clock time. All of these checks establish robot
pose tracking within the specified tolerances, not collision-free movement,
cloth contact, finger capture or grasp stability; there is no contact-force
feedback or new GPT visual judgment inside a sequence.

For example, suppose the measured left pose is position `[0.20, 0.10, 0.90]`,
quaternion `[1, 0, 0, 0]`; the right pose is `[0.20, -0.10, 0.90]` with the same
quaternion. The following **illustrative schedule only** uses 24 total controls
(0.96 simulated seconds). It is not a calibrated grasp point or a guaranteed
grasp, and these numbers must never replace this episode's measured poses.
All targets stay within 3 cm of the illustrative initial poses. These geometric
bounds alone do not establish reachability: the real measured joint positions
and numerical preflight must validate any actual submission, including its
step allocation.
The [complete JSON example](eef_waypoints_example.json) demonstrates the required
fields, including the final-target mirror; it is documentation, not a command
to execute. Replace its request ID, coordinates and assessment with live evidence.

| Node label | Controls | Left target position (m) | Left gripper | Right target / gripper |
| --- | ---: | --- | --- | --- |
| `approach` | 8 | `[0.20, 0.10, 0.87]` | open | fixed initial pose / open |
| `wait` | 3 | `[0.20, 0.10, 0.87]` | open | fixed initial pose / open |
| `grasp` | 5 | `[0.20, 0.10, 0.87]` | closed | fixed initial pose / open |
| `lift` | 8 | `[0.20, 0.10, 0.89]` | closed | fixed initial pose / open |

All four nodes explicitly include both arms' full position, quaternion and
boolean `gripper_closed`. The top-level `target` equals the final `lift` target,
not the first `approach` target. A transition confirms that both measured robot
poses met the tolerances, not that cloth entered the fingers or that the grasp
is stable. For example, if `approach` misses its measured pose tolerance after
8 controls, neither `wait`, `grasp` nor `lift` executes; GPT receives the partial
result and must replan. If grasp facts need visual confirmation, stop this
decision after `approach`, `grasp`
or a small `lift`, inspect the returned observation, then obtain a fresh pi05
proposal and make the next gate-conforming decision. A longer sequence is
appropriate only when proceeding without an intermediate observation is justified
by current evidence; adding waypoints does not change the takeover gate.

Follow `next_call` paths exactly. Use English for public explanations, notes and
reports; preserve keys, enums, paths and the original instruction. Prior episodes
are not evidence of this episode's state or of student-only improvement.
