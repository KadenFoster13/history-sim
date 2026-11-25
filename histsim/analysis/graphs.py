"""Graphs of a finished (or running) simulation."""
import math
import os

import numpy as np

from ..core.calendar import real_year
from ..record.reader import RunReader

# Nicer titles for known statistics; anything else is shown under its own name.
LOG_SCALE = {"world_population", "foragers", "herders", "farmers", "largest_settlement"}
TITLES = {
    "world_population": "World population (Earth's history shown as grey bars)",
    "foragers": "Foragers (hunter-gatherers)",
    "herders": "Herders",
    "farmers": "Farmers",
    "inhabited_land_share": "Share of land with people",
    "famine_deaths_per_year": "Famine deaths per year",
    "drowned_per_year": "People displaced by rising seas per year",
    "domestications": "Species tamed (all events)",
    "farming_origins": "Independent beginnings of farming",
    "species_tamed": "Different species tamed",
    "farmed_land_share": "Share of land farmed",
    "herded_or_farmed_land_share": "Share of land farmed or herded",
    "settled_farmland_share": "Share of farmland settled (not shifting) farming",
    "intensive_farmland_share": "Share of farmland farmed intensively (half way or more)",
    "farm_density_low": "Farmers per km2: thinnest tenth of farmland",
    "farm_density_high": "Farmers per km2: densest tenth of farmland",
    "farm_density_top": "Farmers per km2: densest hundredth of farmland",
    "global_temperature_c": "Global temperature change (C)",
    "sea_level_m": "Sea level (m)",
    "exposed_shelf_share": "Ice-age land still above sea (share of today's land)",
    "drought_share": "Land in drought (share)",
    "volcanic_cooling_c": "Volcanic cooling (C)",
    "cultures": "Cultures (peoples with their own language)",
    "language_families": "Language families",
    "largest_culture_share": "Largest people's share of world population",
    "towns": "Towns (2,000+ people, named and smaller)",
    "named_towns": "Named towns (the main place of each area)",
    "cities": "Cities (10,000+ people)",
    "cities_100k": "Cities of 100,000+",
    "largest_settlement": "Largest town or city (Earth's shown as grey bars)",
    "urban_share": "Share of people in towns and cities",
    "states": "States",
    "kingdoms": "Kingdoms and empires",
    "empires": "Empires",
    "people_in_states": "Share of people living in states",
    "largest_state_share": "Largest state's share of world population",
    "wars": "Wars being fought",
    "wars_started": "Wars so far",
    "war_deaths_per_year": "Deaths in conquest per year",
    "trade_links": "Trade links between towns",
    "trade_volume": "Trade volume (relative)",
    "longest_trade_km": "Longest distance goods travel (km)",
    "inventions": "Inventions (first finds and independent ones)",
    "technologies_most": "Most technologies known in one place",
    "technologies_mean": "Technologies known by the average person",
}
TECH_GROUP_TITLES = {"farming": "farming technology", "metals": "metals", "transport": "transport",
                     "rule": "technology of rule", "building": "building and war"}

# Earth, for the largest place: (simulation year, low, typical, high).
LARGEST_PLACE_TARGETS = [(6_500, 5e3, 1e4, 4e4), (8_000, 2e4, 6e4, 1.5e5), (10_000, 2e5, 1e6, 2e6)]


def _format_axis(ax):
    def fmt(x, _):  # axis labels: round years (off by one around year 0 doesn't matter here)
        x = int(round(x))
        if x < 0:
            return f"{-x:,} BCE"
        return f"{x:,} CE" if x > 0 else "0"
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(fmt))


def _plot_run_time(ax, years, timing, colours):
    """Seconds of computing per century of history, by system (stacked)."""
    step = 100
    rows = np.flatnonzero((years - years[0]) % step == 0)
    if len(rows) < 2:
        return
    x = real_year(years[rows[1:]] - step / 2)
    names = sorted(timing, key=lambda k: -np.nan_to_num(timing[k])[-1])
    rates = [np.diff(np.nan_to_num(timing[k])[rows]) for k in names]
    ax.stackplot(x, rates, labels=names, alpha=0.85)
    ax.legend(fontsize=7, loc="upper left", ncol=2, frameon=False, labelcolor=colours["fg"])
    ax.set_title("Run time: seconds of computing per century, by system", fontsize=10)
    ax.grid(alpha=0.3, color=colours["grid"])
    _format_axis(ax)


def _plot_knowing(ax, years, knowing, group, colours):
    """Share of the world's people who know each technology of a group."""
    from ..tech import TECHS
    x = real_year(years)
    for t in TECHS:
        if t.group == group and t.key in knowing:
            ax.plot(x, np.nan_to_num(knowing[t.key]), linewidth=1.2, label=t.name)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper left", ncol=2, frameon=False, labelcolor=colours["fg"])
    ax.set_title(f"Share of people who know: {TECH_GROUP_TITLES.get(group, group)}", fontsize=10)
    ax.grid(alpha=0.3, color=colours["grid"])
    _format_axis(ax)


def plot_run(run_dir, out=None, keys=None, smooth_years=0, dark=None):
    """A grid of time-series graphs for a run; saved to `out` (default: <run>/graphs.png).

    keys          which statistics to plot (default: all)
    smooth_years  optional running average to make noisy series readable
    dark          dark background (default: the viewers' remembered setting)
    """
    from matplotlib.figure import Figure  # draws to a file without opening a window

    from .style import palette, style_figure
    colours = palette(dark)

    run = RunReader(run_dir)
    stats = run.stats()
    years = stats.pop("year")
    # Run time per system (cumulative seconds): drawn together in one graph.
    timing = {k[len("seconds_"):]: stats.pop(k) for k in list(stats) if k.startswith("seconds_")}
    # Share of people knowing each technology: one graph per group of technologies.
    knowing = {k[len("know_"):]: stats.pop(k) for k in list(stats) if k.startswith("know_")}
    names = [k for k in (keys or stats.keys()) if k in stats and np.isfinite(stats[k]).any()]
    if timing and (keys is None or "run_time" in keys):
        names.append("run_time")
    from ..tech import GROUPS, TECHS
    for group in GROUPS:
        if any(t.group == group and t.key in knowing for t in TECHS) and (keys is None or f"tech_{group}" in keys):
            names.append(f"tech_{group}")
    if not names:
        raise ValueError("no statistics to plot")
    cols = 2 if len(names) > 1 else 1
    rows = math.ceil(len(names) / cols)
    fig = Figure(figsize=(7 * cols, 2.6 * rows))
    axes = fig.subplots(rows, cols, squeeze=False)
    x = real_year(years)
    for ax, name in zip(axes.ravel(), names):
        if name == "run_time":
            _plot_run_time(ax, years, timing, colours)
            continue
        if name.startswith("tech_"):
            _plot_knowing(ax, years, knowing, name[len("tech_"):], colours)
            continue
        y = stats[name]
        if smooth_years and len(y) > smooth_years:
            kernel = np.ones(smooth_years) / smooth_years
            y = np.convolve(np.nan_to_num(y), kernel, mode="same")
        ax.plot(x, y, linewidth=1, color=colours["line"])
        if name in LOG_SCALE and np.nanmax(y) > 0:
            ax.set_yscale("log")
        targets = None
        if name == "world_population":
            from ..config import POPULATION_TARGETS
            targets = POPULATION_TARGETS
        elif name == "largest_settlement":
            targets = LARGEST_PLACE_TARGETS
        for year, low, typical, high in targets or ():
            rx = real_year(year)
            if x.min() <= rx <= x.max():
                ax.plot([rx, rx], [low, high], color=colours["target"], linewidth=4, alpha=0.6)
                ax.plot(rx, typical, "o", color=colours["target"], markersize=4)
        ax.set_title(TITLES.get(name, name.replace("_", " ")), fontsize=10)
        ax.grid(alpha=0.3, color=colours["grid"])
        _format_axis(ax)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    meta = run.meta
    fig.suptitle(f"World {meta['world_seed']}, history seed {meta['sim_seed']}", fontsize=12)
    style_figure(fig, axes.ravel(), colours)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / (2.6 * rows)))   # leave room for the title
    out = out or os.path.join(run_dir, "graphs.png")
    fig.savefig(out, dpi=100, facecolor=fig.get_facecolor())
    return out
