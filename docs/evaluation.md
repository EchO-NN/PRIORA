# Evaluation panels and run artifacts

| Panel | Layouts per task | Episodes |
| --- | ---: | ---: |
| [Paired evaluation](../priora/benchmarks/paired252.json) | 6 | 252 |
| [Additional layouts](../priora/benchmarks/unseen210.json) (default) | 5 | 210 |

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

To compare controllers on the same layout, match the panel digest and case identity. The layout group (`eval_seed`) selects a group of published layouts. The reset seed selects an entry within that group. After these match, use the native evaluation record to determine whether the episode finished, whether it counts toward success rate, and what score the evaluator returned.

The [tower example](../examples/build_tower/README.md) follows one acknowledged step. The [case verifier](../priora/robodojo/evaluation.py) implements the layout and source checks.
