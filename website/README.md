# PRIORA project page

The [project page](https://echo-nn.github.io/PRIORA/) compares methods on RoboDojo and presents up to six independent full episodes per task variant. Standard and random variants have separate galleries.

The static site source lives here. The published `gh-pages` branch also contains the media. To preview it locally:

```sh
git fetch origin gh-pages
git worktree add --detach ../PRIORA-pages origin/gh-pages
python -m http.server 8000 --directory ../PRIORA-pages
```

Replace the figure with ID `pipeline-figure` in `index.html` when the pipeline illustration is ready.

## Results and media

`data/catalog.json` contains aggregate percentages and source information. PRIORA pools each evaluated layout's final outcome. The dimension rates average their task-family rates, and the overall mean weights the five dimensions equally. Video selection is independent of these calculations.

A method’s `adaptations` field records documented use of task-specific skills or recursive self-improvement in the displayed configuration. Either flag displays an asterisk in the comparison and heatmap. RoboDojo publishes its protocol disclosures separately.

Examples use public names of the form `task--variant--1`. The matching JSON trace contains the control period, subtask strings, and half-open segments `[start, end, mode, subtaskIndex]`. Each example runs from its initial observation through its terminal frame. Each source episode appears once. The trace includes the exact video SHA-256 and frame count. The example badges reflect the final outcome of the source rollout. Task cards and the player show the success rate over all evaluated layouts for that variant.

Each video stores three synchronized 640 × 360 views side by side at 25 frames per second, ordered main, left wrist, right wrist. Playback speed ranges from 0.25× to 4×. The camera controls, control timeline, and subtask display share the video clock at every speed.

Triangles mark the first control step of each new recorded subtask within the clip. Select a triangle to seek to that boundary and see the previous and next instructions. The vertical line tracks the current playback time.
