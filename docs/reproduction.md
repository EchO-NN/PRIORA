# Reproduce a RoboDojo episode

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

By default the runner retrieves the official task card and verifies its text digest. Use `--card /path/to/card.json` for an offline copy with the same digest. The trusted registry is in [cards.json](../priora/robodojo/wall_oss/cards.json).

See the [evaluation panels and output files](evaluation.md) to interpret a finished run.
