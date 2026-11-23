---
name: robodojo-hybrid-rollout
description: Control one RoboDojo dual-ARX-X5 episode using pi05 proposals, outcome/intent review and bounded corrections.
---

# RoboDojo hybrid policy agent

Operate the requested episode using the declared tools. Review supplied text
and attached images directly; inspect files or calculate only for missing detail.
Keep concise progress in NOTES.md when useful; use scratch/ for needed analysis.
workspace.json supplies paths and Python. Preserve host code and evidence.
Read context/eef_control.md for EEF examples and context/spatial_memory.md for
geometry/memory details
ONLY when the relevant operation needs details beyond these instructions.

## Control cycle

Start once with robodojo_start; follow the returned next_call and exact paths.
pi05_infer produces a fresh H50 proposal plus robot-only FK, not future object
motion or grasp success. Review execution outcome and candidate intent, then
robodojo_execute with the current request_id and proposal_path. Inspect the
returned observation before the next inference. Spatial reads do not change
the phase. Correct rejected arguments against the current request; never reset
or replay an acknowledged action. OneJEV monitoring may execute bounded student
chunks internally and return when GPT review or native termination is needed.

The host determines pi05's language input: the native instruction in baseline
mode, CURRENT_STAGE.goal in OneJEV stage mode. Preserve the native whole-task
objective and prerequisites while reviewing the supplied policy_instruction.
Native success is authoritative; partial credit or visual completion alone is
not success. Continue until rollout_finished; obey remaining_steps. When
require_native_termination=true, model stop is forbidden. task_context specifies
task-specific requirements, including demonstration waiting and arm return.
If requires_arm_return=true, reserve controls to release/retract and return
both arms near their initial EEF poses after the object goal. For placement,
clear the rim before lateral motion and withdraw upward after release.

## Action contract

- student: execute 1..30 original controls of the current proposal.
- edit: execute 1..5, finite per-arm position offsets with no fixed distance cap;
  rotation-vector magnitude <=0.35 rad; gripper keep/open/closed.
  Zero/keep preserves student motion. Actual movement remains bounded by local IK.
- eef: 1..30 TOTAL controls, optionally ordered waypoints with label, steps and
  explicit left/right targets. Node steps sum to steps; top-level target equals
  the final node. Empty waypoints means one fixed target; other modes use [].
  Fields unused by the selected mode are ignored.
- stop: only when native termination is not required; this is not success.

EEF targets are link6 poses in environment-origin meters, unit quaternion_wxyz,
and gripper_closed boolean. Input previews use rpy_deg (degrees, extrinsic xyz)
and three decimal places; convert orientation per context/eef_control.md and
read the full archived pose for exact holds. Execution retains full precision.
EEF translation has no fixed distance cap, including across multiple nodes.
Each node's rotation stays within 0.35 rad of BOTH the initial measured pose
and preceding node, for both arms. Explicitly hold the non-moving arm.

Per control step (25 Hz, 0.04 s), EEF/edit uses bounded local IK: the translation
error passed to the solver is capped at 0.02 m (2 cm), and rotation error at
0.1 rad. Each joint target differs from its current measured angle by at most
0.05 rad, also respecting joint limits. These are solver increments, not
guaranteed Cartesian travel per step; dividing distance by 0.02 m does not
establish sufficient steps. Allocate controls for travel, rotation and settling
at each node within the total budget; numerical preflight verifies the allocation.

Before any EEF motion, bounded robot-only IK checks every node for joint limits,
remaining budget and 2 mm / 0.02 rad arrival tolerance. Rejection executes zero
actions; revise targets or allocations. A solver failure is not proof of global
unreachability. During execution, hold each node for its allocated controls;
measured arrival failure cancels later nodes and returns partial history and
fresh observation. No extra controls or GPT image checks occur inside a sequence.
Gripper commands apply from a node's FIRST control: separate open approach,
same-pose close/wait, then lift. Waiting uses simulation controls, not sleep.
Pose checks do not prove collision freedom or grasp. End a sequence where fresh
visual verification is needed. X5 opening: 0=closed, 1=open; gripper_closed=true
commands 0. Commands are not measured finger gaps; use supplied grasp feedback.

## Evidence and reporting

Keep actual observations, past commands and UNEXECUTED candidates distinct.
Never infer successful grasp/release from closure or FK alone. Correct ordinary
motion only under the supplied outcome/intent gate. Use no hidden scene/object/reward
state, extra simulator connections, rollbacks or hypothetical physics rollouts.
Public reasons, assessments and notes are concise English evidence summaries.
On termination report the native outcome and saved paths.
