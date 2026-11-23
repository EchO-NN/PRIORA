# How PRIORA decides who acts

A control cycle has two separate questions. Did the last executed action advance the current subtask? Does the proposed action fit what needs to happen next? A failed grasp can be followed by a useful WALL recovery. A successful grasp can be followed by a proposal that carries the object to the wrong destination. GPT evaluates both questions before choosing an action. [Review criteria](../priora/robodojo/robodojo_server/gate_assessment.py)

## Inputs and memory

| Component | Inputs | Output |
| --- | --- | --- |
| WALL | Current RGB cameras, robot state, language instruction and episode-specific keyframe memory | 32 absolute dual-arm EEF targets. Each arm has XYZ, a quaternion and a gripper command |
| GPT | Official instruction and scoring card, current RGB-D views, candidate motion and fingertip overlays, measured grasp feedback, same-episode history | A reviewed action, or a new instruction for WALL before requesting the next proposal |
| RoboDojo | Validated controls | Acknowledged control steps, fresh observations and native termination and scoring |

An episode opens one isolated teacher workspace and conversation. The controller archives the main camera at 2 Hz and retains GPT's public dialogue for that episode. WALL owns its own keyframe memory. A new episode starts a new WALL session and a new GPT workspace. [WALL session handling](../priora/robodojo/wall_oss/policy_server.py) · [Teacher history](../priora/robodojo/teacher_history.py) · [Workspace isolation](../priora/robodojo/skill/episode_isolation.py)

## Reviewing candidate motion

Camera intrinsics, robot poses and rigid camera mounts relate pixels from all three cameras to the robot's environment frame. The renderer projects the two fingertips of each gripper onto the current images, using the proposed prefix. The main overlay is always included when projection is available. Wrist overlays are selected using visibility, spatial motion and gripper state. The paths describe commanded motion. Measured gripper state and contact after execution provide separate evidence about whether a grasp held. [Camera geometry](../priora/robodojo/camera_geometry.py) · [View selection](../priora/robodojo/realtime_monitor/current_view_selection.py) · [Grasp feedback](../priora/robodojo/grasp_feedback.py)

## Correcting the action or the instruction

For single-object tasks, WALL first receives the complete instruction. GPT can correct an execution error while keeping the same goal. If the proposal pursues an incompatible goal, GPT can split the instruction. For multi-object tasks it can assign a coherent object or category from the outset. A routine retry preserves the active subtask unless the observed state warrants changing it. The tool called `pi05_infer` in the recorded protocol requests WALL's proposal. [Teacher instructions](../priora/robodojo/wall_oss/teacher.md)

For a GPT-authored EEF correction, the shared cuRobo implementation checks the timed waypoints and the joint trajectory before motion. The solver applies the same frame conventions and joint limits at execution. Its collision checks cover the configured robot self-collision, table and elbow geometry. A rejected preflight leaves the simulation step unchanged so GPT can revise the plan. On acceptance, the controller records the prefix the simulator actually executed. [Shared cuRobo implementation](../priora/robodojo/wall_oss/curobo_kinematics.py) · [Motion smoothing](../priora/robodojo/wall_oss/motion_smoothing.py)
