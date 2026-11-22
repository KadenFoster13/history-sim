"""Map pictures of a run: where people live and how, at chosen years."""
import os

import numpy as np

from ..core.calendar import label
from ..record.reader import RunReader

MODE_COLOURS = {1: (0.95, 0.80, 0.35), 2: (0.80, 0.45, 0.20), 3: (0.20, 0.65, 0.20)}  # forage, herd, farm
MAP_KINDS = ("people", "density", "arrival")


def base_map(world):
    """A muted terrain picture to draw on."""
    from worldgen.render import render
    return render(world, "Elevation") * 0.45 + 0.5


def people_map(world, layers):
    """How people live: forager (gold), herder (orange), farmer (green); brighter = denser."""
    img = base_map(world)
    pop = layers["population"]
    mode = layers.get("subsistence", np.where(pop > 0, 1, 0))
    area = (world.grid.km_per_px ** 2) * np.maximum(np.cos(np.radians(world.grid.lat)), 0.01)
    density = pop / area
    strength = np.clip((np.log10(np.maximum(density, 1e-4)) + 2.5) / 3.0, 0.25, 1)[..., None]  # 0.003..3+/km2
    for value, colour in MODE_COLOURS.items():
        here = (mode == value) & (pop > 0)
        img[here] = img[here] * (1 - strength[here]) + np.array(colour) * strength[here]
    return np.clip(img, 0, 1)


def density_map(world, layers):
    """People per km2 on a log scale."""
    from matplotlib import colormaps
    img = base_map(world)
    pop = layers["population"]
    area = (world.grid.km_per_px ** 2) * np.maximum(np.cos(np.radians(world.grid.lat)), 0.01)
    d = np.log10(np.maximum(pop / area, 1e-4))
    here = pop > 0
    img[here] = colormaps["inferno"](np.clip((d[here] + 3) / 5, 0, 1))[:, :3]
    return img


def arrival_map(world, layers):
    """When people first arrived (before year 0), from early (dark) to late (light); grey = nobody yet."""
    from matplotlib import colormaps
    img = base_map(world)
    arrival = layers["arrival_year"]
    reached = np.isfinite(arrival)
    if reached.any():
        lo, hi = np.nanmin(arrival[reached]), min(np.nanmax(arrival[reached]), 0)
        t = np.clip((arrival[reached] - lo) / max(hi - lo, 1), 0, 1)
        img[reached] = colormaps["viridis"](t)[:, :3]
    land = ~world.ocean & ~world.lake
    img[land & ~reached] = (0.6, 0.6, 0.6)
    return img


def plot_maps(run_dir, years=None, kind="people", out=None, world=None):
    """A grid of maps of `kind` (people, density or arrival) at several years."""
    from matplotlib.figure import Figure
    from ..world import load_world

    run = RunReader(run_dir)
    world = world or load_world(run.meta["world_seed"], run.meta["world_width"])
    if years is None:
        span = run.last_year - run.first_year
        years = [run.first_year + round(span * f / 5 / 100) * 100 for f in range(6)]
    draw = {"people": people_map, "density": density_map, "arrival": arrival_map}[kind]
    cols = 2
    rows = (len(years) + 1) // cols
    fig = Figure(figsize=(8 * cols, 4.2 * rows))
    axes = fig.subplots(rows, cols, squeeze=False)
    for ax, year in zip(axes.ravel(), years):
        layers = run.layers_at(year)
        ax.imshow(draw(world, layers), interpolation="nearest")
        pop = layers.get("population")
        total = f", {pop.sum() / 1e6:,.1f} million people" if pop is not None else ""
        ax.set_title(f"{label(run.snapshot_year(year))}{total}", fontsize=10)
        ax.set_axis_off()
    for ax in axes.ravel()[len(years):]:
        ax.set_visible(False)
    legend = {"people": "gold = foragers, orange = herders, green = farmers (brighter = more people)",
              "density": "people per km2 (log scale, dark to bright)",
              "arrival": "when people arrived: dark = earliest, light = latest, grey = not yet reached"}[kind]
    fig.suptitle(f"World {run.meta['world_seed']}, history seed {run.meta['sim_seed']}: {legend}", fontsize=11)
    fig.tight_layout()
    out = out or os.path.join(run_dir, f"maps_{kind}.png")
    fig.savefig(out, dpi=80)
    return out
