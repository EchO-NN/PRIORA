"""Always-visible operating rules; detailed manuals stay in the agent workspace."""

VERSION = 'compact_teacher_instructions_v8_official_arm_return'

EFFICIENCY_GUIDE = """Use supplied evidence before calling more tools:
The latest successful tool reply already supplies the current observation,
request/proposal identity, next_call, robot state and any candidate trajectory.
Attached images are directly visible; viewing their source files is optional,
not a required observation step. Review these inputs and decide directly when
they suffice. Do not routinely list directories or reread call_*_result.json,
runtime_state.json or the same trajectory to rediscover supplied fields.
Reopen/crop only a specific unresolved detail requiring original resolution;
select the relevant view, not all cameras by habit. If context compaction has
lost an input, retrieve that particular artifact. Same-step rereads are not new
physical observations. After motion always inspect the returned fresh evidence.
Batch independent necessary file reads/calculations in one call. Query geometry
only when needed for a specific decision/target, not for routine student approval.
Reuse valid memory entries; add notes/patches only for new information useful to
later decisions or before losing a needed view. No per-block bookkeeping ritual.
Do not replay actions or skip outcome/intent checks to save calls. Corrective
actions still require their normal validation and fresh execution feedback.
Use the existing camera views; do not move either arm solely to obtain a new
viewpoint. That observation-only action feature has been removed.

Use the official task description and scoring table for this episode to determine whether arm return is required.
"""

SPATIAL_GUIDE = """Spatial memory (rigid_spatial_v3):
The host supplies current camera_geometry from fixed mounts and measured EEF
poses; no fitting or calibration moves are needed. spatial_query converts an
original RGB pixel using raw depth into visible-surface coordinates; an empty
pixel list projects a link6 origin/registered landmark. Neither is automatically
a pinch point or reachable EEF target. spatial_record only registers a visible
RIGID gripper landmark; never register cloth/held objects as part of the robot.
Before close approach/occlusion, spatial_remember can save a labeled local
RGB/depth/XYZ patch; spatial_recall reads it or lists entries with memory_id="".
These are dated observations, not current geometry. Check source step and
spatial_status.near_field_memory: invalidated_rgb_change and unverified cannot
guide motion; same_observation/consistent are only hints. Reacquire the feature
before contact; object/cloth motion can invalidate old XYZ, and camera motion
does not carry scene points with the wrist. Read context/spatial_memory.md for
file formats, conversion examples and detailed validity rules when needed.
Maintain SPATIAL_MEMORY.md and SPATIAL_PROMPT.md only when useful evidence or
guidance changes; use host geometry rather than inventing transforms/offsets.
"""
