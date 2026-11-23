# PRIORA: Integrating Learned Action Priors into Generalist Robot Agents

PRIORA couples a demonstration-trained action policy with a generalist multimodal agent. In each RoboDojo control cycle, WALL proposes a bimanual trajectory. The GPT agent sees the proposal before execution and chooses how much of it to execute, whether to edit it, or whether to supply its own end-effector waypoints. The next decision uses fresh observations after the simulator acknowledges the executed controls.

## How an episode works

1. The controller opens an isolated workspace and starts one teacher conversation for one layout. GPT receives the task's official detailed instruction and scoring table. The native evaluator retains responsibility for scoring.
2. WALL-OSS 1.0 Lite takes RGB views, robot state, its keyframe memory and a language instruction. It proposes 32 steps of absolute dual-arm end-effector poses and gripper commands, with 16 values per step. For a single-object task the teacher initially tries the full instruction. It can assign a smaller goal when the proposed intent is wrong. For multi-object tasks it can select an object or category. A subtask persists across ordinary corrections.
3. GPT reviews the main and wrist RGB-D observations, the candidate trajectory drawn in camera space, measured finger/contact feedback, and the current goal. Camera intrinsics and rigid camera-to-robot transforms support pixel-to-world queries, projected fingertip paths and episode-local spatial memory. The teacher has access to the conversation and a 2 Hz main-camera history from this episode.
4. GPT can release 1–16 WALL steps, submit a bounded 1–5 step edit, or plan 1–30 end-effector steps across timed waypoints. Shared cuRobo preflight and execution use consistent frames and tracking limits. Smoothing reduces abrupt starts and approaches. A rejected preflight consumes zero physical steps. The controller records the actual executed prefix and observes again.
5. The simulator supplies termination, step limit and native outcome. The run directory keeps observations, proposals, controls, tool decisions and a replay for inspection.

The tool name `pi05_infer` is retained for compatibility with the recorded tool protocol. In this project it calls WALL.

## Install

Use Linux with an NVIDIA GPU. The controller, simulator and policy can use separate Python environments. The commands below assume a source checkout. Run them from its root unless a different checkout is named.

**Controller.** Install Python 3.10 or newer, Git, FFmpeg, bubblewrap and a CJK font for the replay labels. The reference host used Python 3.11 and Codex CLI `0.169.0-alpha.11.1`. The CLI must support the app-server protocol, dynamic tools and the model selected for the teacher.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
codex login
```

**Simulator.** Clone the benchmark and install its assets, Isaac Sim 5.1, Isaac Lab 2.3 and cuRobo following the benchmark's installation guide. Use a dedicated checkout for the evaluation patch. It records the instruction clarification for the demonstration-arm sorting task. The reference assets revision is `269457a3b42d8d0ae69898a601f2c1487c2f826e`. With the X5 assets installed, set `ROBODOJO_X5_URDF="$ROBODOJO_ROOT/Assets/Robots/x5/X5A.urdf"` in the environment to include the native-arm tests. The run verifies source and layout hashes before it opens the teacher conversation. [RoboDojo installation](https://robodojo-benchmark.com/doc/usage/install-and-download/)

```bash
export PRIORA_ROOT="$PWD"
git clone https://github.com/RoboDojo-Benchmark/RoboDojo.git ../RoboDojo
export ROBODOJO_ROOT="$(cd ../RoboDojo && pwd)"
git -C "$ROBODOJO_ROOT" checkout f57e1633ffed59761bc8e568cfe7530c4b33381c
git -C "$ROBODOJO_ROOT" submodule update --init --recursive
git -C "$ROBODOJO_ROOT" apply "$PRIORA_ROOT/priora/benchmarks/robodojo.patch"
```

**Action policy.** Use a separate CUDA environment for Wall-X. Check out the revision below, follow its installation instructions for PyTorch and the compiled operators, then install PRIORA into that same environment. Download the RoboDojo model snapshot with its sibling `checkpoint/` and `assets/processor/` directories. The service verifies the checkpoint SHA-256 on startup. [Wall-X installation](https://github.com/StarrickLiu/wall-x/tree/39fa65b9c9e95b0316db0fc9dbe41211e3a0d4cc) · [WALL-OSS 1.0 Lite RoboDojo weights](https://huggingface.co/x-square-robot/wall-oss-1.0-lite-robodojo)

```bash
git clone https://github.com/StarrickLiu/wall-x.git ../wall-x
git -C ../wall-x checkout 39fa65b9c9e95b0316db0fc9dbe41211e3a0d4cc
# After installing Wall-X in its CUDA environment
python -m pip install -e "$PRIORA_ROOT"
hf download x-square-robot/wall-oss-1.0-lite-robodojo --revision 9b20e7051e60d9a9cf9da085a93874ea5f55f9e0 --local-dir ./weights/wall
export PRIORA_WALL_CHECKPOINT="$PWD/weights/wall/checkpoint"
CUDA_VISIBLE_DEVICES=0 priora-wall --port 19660
```

## Run a task

Keep the policy server running. In a second terminal activate the controller environment. Supply the Python interpreter that can import RoboDojo's Isaac packages. Use the local simulator's usual Vulkan and Isaac environment settings.

```bash
export SIM_PYTHON=/path/to/robodojo-environment/bin/python
CASE=build_tower__standard__g0__l28
priora cases
priora check --robodojo "$ROBODOJO_ROOT" --case "$CASE"
priora run --robodojo "$ROBODOJO_ROOT" --sim-python "$SIM_PYTHON" \
  --case "$CASE" --output "runs/$CASE" --policy-port 19660 --sim-port 19113
```

The default teacher is `gpt-6.2-astra` with `xhigh` reasoning through the signed-in Codex CLI. For a compatible Responses endpoint, set `PRIORA_BASE_URL` and provide `PRIORA_API_KEY` through the environment. The teacher sends episode observations to the chosen service. The runtime checks the reported teacher identity and effort. Each new episode requires a fresh output directory. Multiple simulations on one machine can share a policy port while using distinct simulation ports and output directories.

By default the runner retrieves the official task card and verifies its text digest. Use `--card /path/to/card.json` for an offline copy with the same digest. The trusted registry is in [cards.json](priora/robodojo/wall_oss/cards.json).

## Reproducibility and outputs

| Panel | Layouts per task | Episodes |
| --- | ---: | ---: |
| [Paired evaluation](priora/benchmarks/paired252.json) | 6 | 252 |
| [Additional layouts](priora/benchmarks/unseen210.json) (default) | 5 | 210 |

Both panels cover 42 tasks. Pass `--panel priora/benchmarks/paired252.json` to `cases`, `check` or `run` to select the paired layouts. The manifests pin layout groups, reset indices, seeds, layout hashes and support trajectories. They describe evaluation inputs. This release contains the current agent configuration; historical experiments used several earlier versions.

| File under the output directory | Contents |
| --- | --- |
| `layout.json` | Selected case identity and frozen fixture |
| `controller/codex_workspace/PROMPT.md` | Exact teacher prompt and scoring card for this run |
| `controller/history.json` | Decisions with acknowledged execution intervals |
| `controller/codex_workspace/episode_history_2hz/` | Main-camera history for this episode |
| `controller/debug_video/` | Replay artifacts |
| `controller/result.json` | Terminal controller outcome |
| `sim/` | Native simulator artifacts |

The main implementation entry points are the [episode launcher](priora/cli.py), [WALL service](priora/robodojo/wall_oss/policy_server.py), [teacher tools](priora/robodojo/wall_oss/client.py), [teacher prompt](priora/robodojo/wall_oss/teacher.md) and [shared cuRobo implementation](priora/robodojo/wall_oss/curobo_kinematics.py). See [third-party notices](THIRD_PARTY_NOTICES.md) for component attribution.
