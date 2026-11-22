"""Graphs of a finished (or running) simulation."""
import math
import os

import numpy as np

from ..core.calendar import real_year
from ..record.reader import RunReader

# Nicer titles for known statistics; anything else is shown under its own name.
TITLES = {
    "global_temperature_c": "Global temperature change (C)",
    "sea_level_m": "Sea level (m)",
    "exposed_shelf_share": "Ice-age land still above sea (share of today's land)",
    "drought_share": "Land in drought (share)",
    "volcanic_cooling_c": "Volcanic cooling (C)",
}


def _format_axis(ax):
    def fmt(x, _):  # axis labels: round years (off by one around year 0 doesn't matter here)
        x = int(round(x))
        if x < 0:
            return f"{-x:,} BCE"
        return f"{x:,} CE" if x > 0 else "0"
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(fmt))


def plot_run(run_dir, out=None, keys=None, smooth_years=0):
    """A grid of time-series graphs for a run; saved to `out` (default: <run>/graphs.png).

    keys          which statistics to plot (default: all)
    smooth_years  optional running average to make noisy series readable
    """
    from matplotlib.figure import Figure  # draws to a file without opening a window

    run = RunReader(run_dir)
    stats = run.stats()
    years = stats.pop("year")
    names = [k for k in (keys or stats.keys()) if k in stats and np.isfinite(stats[k]).any()]
    if not names:
        raise ValueError("no statistics to plot")
    cols = 2 if len(names) > 1 else 1
    rows = math.ceil(len(names) / cols)
    fig = Figure(figsize=(7 * cols, 2.6 * rows))
    axes = fig.subplots(rows, cols, squeeze=False)
    x = real_year(years)
    for ax, name in zip(axes.ravel(), names):
        y = stats[name]
        if smooth_years and len(y) > smooth_years:
            kernel = np.ones(smooth_years) / smooth_years
            y = np.convolve(np.nan_to_num(y), kernel, mode="same")
        ax.plot(x, y, linewidth=1)
        ax.set_title(TITLES.get(name, name.replace("_", " ")), fontsize=10)
        ax.grid(alpha=0.3)
        _format_axis(ax)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    meta = run.meta
    fig.suptitle(f"World {meta['world_seed']}, history seed {meta['sim_seed']}", fontsize=12)
    fig.tight_layout()
    out = out or os.path.join(run_dir, "graphs.png")
    fig.savefig(out, dpi=100)
    return out
