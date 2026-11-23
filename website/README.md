# PRIORA project page

The [live site](https://echo-nn.github.io/PRIORA/) is a static GitHub Pages project. The published tree on `gh-pages` contains the videos and posters. This directory on `main` contains the page, numeric results and compact control traces.

To preview the complete published site locally, run these commands from the repository.

```sh
git fetch origin gh-pages
git worktree add --detach ../PRIORA-pages origin/gh-pages
python -m http.server 8000 --directory ../PRIORA-pages
```

Replace the figure with ID `pipeline-figure` in `index.html` when the pipeline illustration is ready. Use a relative asset path to support the GitHub project subdirectory.

## Data

`data/catalog.json` contains the 42 task families, their 54 task/variant combinations, the two evaluation panels, the source protocols for the published baselines, and references to episode media and traces. Percentages in method records use a 0–100 scale. PRIORA rates are calculated from the Boolean final outcome of each layout. A null outcome is pending. Additional attempts marked `displayOnly` are available in the player and excluded from the metrics. For each dimension, the page averages its task-family rates. The five-dimension mean weights the dimensions equally.

Each replay trace contains the number of executed control steps, the control period in seconds, a dictionary of subtask strings, and segments of the form `[start, end, mode, subtaskIndex]`. The control interval is zero-based and half-open. Frame zero is the reset observation. WALL, GPT edit, and GPT takeover refer to executed controls. Archival main-camera reconstructions are identified in the player.

The header uses the official [Shanghai Jiao Tong University seal artwork](https://vi.sjtu.edu.cn/index.php/articles/bulletin/13), displayed in the blue from its [visual identity palette](https://vi.sjtu.edu.cn/index.php/articles/base/3).
