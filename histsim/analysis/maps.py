"""Map pictures of a run: where people live and how, at chosen years."""
import os

import numpy as np

from ..core.calendar import label
from ..record.reader import RunReader

# forage (gold), herd (orange), farm (crimson: stands out on green land)
MODE_COLOURS = {1: (0.95, 0.80, 0.35), 2: (0.85, 0.45, 0.15), 3: (0.78, 0.12, 0.40)}
MAP_KINDS = ("people", "density", "cultures", "families", "states", "arrival")
WAR_RED = np.array([0.85, 0.08, 0.08])
BACKGROUNDS = ("terrain", "biomes", "white")    # what maps are drawn over
TOWN_MIN = 2000
WHITE_LAND, WHITE_SEA = (0.97, 0.97, 0.96), (0.80, 0.87, 0.93)

_BASE = {}


def base_map(world, background="terrain"):
    """The picture maps are drawn on (a fresh copy each time):
    terrain  muted relief (elevation colours)
    biomes   muted biome colours (forest, grassland, desert...)
    white    plain white land and pale blue sea
    """
    key = (id(world), background)
    if key not in _BASE:
        from worldgen.render import render
        if len(_BASE) > 6:
            _BASE.clear()
        if background == "biomes":
            img = render(world, "Biomes") * 0.55 + 0.42
        elif background == "white":
            water = (world.ocean | world.lake)[..., None]
            img = np.where(water, np.array(WHITE_SEA), np.array(WHITE_LAND))
        else:
            img = render(world, "Elevation") * 0.45 + 0.5
        _BASE[key] = np.clip(img, 0, 1).astype(np.float32)
    return _BASE[key].copy()


def people_map(world, layers, tables=None, background="terrain", year=None):
    """How people live: forager (gold), herder (orange), farmer (crimson); stronger = denser."""
    img = base_map(world, background)
    pop = layers["population"]
    mode = layers.get("subsistence", np.where(pop > 0, 1, 0))
    area = (world.grid.km_per_px ** 2) * np.maximum(np.cos(np.radians(world.grid.lat)), 0.01)
    density = pop / area
    strength = np.clip((np.log10(np.maximum(density, 1e-4)) + 2.5) / 3.0, 0.25, 1)[..., None]  # 0.003..3+/km2
    for value, colour in MODE_COLOURS.items():
        here = (mode == value) & (pop > 0)
        img[here] = img[here] * (1 - strength[here]) + np.array(colour) * strength[here]
    return np.clip(img, 0, 1)


def density_map(world, layers, tables=None, background="terrain", year=None):
    """People per km2 on a log scale."""
    from matplotlib import colormaps
    img = base_map(world, background)
    pop = layers["population"]
    area = (world.grid.km_per_px ** 2) * np.maximum(np.cos(np.radians(world.grid.lat)), 0.01)
    d = np.log10(np.maximum(pop / area, 1e-4))
    here = pop > 0
    img[here] = colormaps["inferno"](np.clip((d[here] + 3) / 5, 0, 1))[:, :3]
    return img


def culture_map(world, layers, tables=None, background="terrain", year=None):
    """Each culture in its own colour (related cultures in similar colours)."""
    img = base_map(world, background)
    cult = layers.get("culture")
    if cult is None or not tables or not tables.get("cultures"):
        return img
    colours = culture_colours(tables)
    here = cult > 0
    img[here] = colours[cult[here]] * 0.85 + img[here] * 0.15
    return img


def family_map(world, layers, tables=None, background="terrain", year=None):
    """Language families: every culture coloured by the family its language belongs to."""
    img = base_map(world, background)
    cult = layers.get("culture")
    if cult is None or not tables or not tables.get("cultures"):
        return img
    colours = culture_colours(tables, by_family=True)
    here = cult > 0
    img[here] = colours[cult[here]] * 0.85 + img[here] * 0.15
    return img


def state_map(world, layers, tables=None, background="terrain", year=None):
    """States in their colours with darker borders; stateless land left plain; fronts of the
    wars being fought in `year` in red."""
    img = base_map(world, background)
    own = layers.get("state")
    if own is None or not tables or not tables.get("states"):
        return img
    colours = state_colours(tables)
    here = own > 0
    img[here] = colours[own[here]] * 0.8 + img[here] * 0.2
    from worldgen.generator import shift
    border = np.zeros(own.shape, bool)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        border |= (shift(own, dy, dx) != own) & here
    img[border] *= 0.55
    for a, b in active_war_pairs(tables, year):
        front = np.zeros(own.shape, bool)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nb = shift(own, dy, dx)
            front |= ((own == a) & (nb == b)) | ((own == b) & (nb == a))
        img[front] = WAR_RED
    return img


def state_colours(tables):
    """An (n, 3) colour lookup by state id."""
    states = tables["states"]
    out = np.full((len(states), 3), 0.6)
    for st in states:
        if st is not None:
            out[st["id"]] = st["colour"]
    return out


def active_war_pairs(tables, year):
    """(attacker, defender) pairs at war in `year`."""
    if year is None:
        return []
    pairs = []
    for war in tables.get("wars") or []:
        if war["started"] <= year and (war["ended"] is None or war["ended"] > year):
            pairs += [(a, d) for a in war["attackers"] for d in war["defenders"]]
    return pairs


def culture_colours(tables, by_family=False):
    """An (n, 3) colour lookup by culture id (id 0 and unknown = grey)."""
    import colorsys
    cultures = tables["cultures"]
    out = np.full((len(cultures), 3), 0.6)
    family_colour = {}
    for c in cultures:
        if c is None:
            continue
        if by_family:
            if c["family"] not in family_colour:
                h, _, _ = colorsys.rgb_to_hsv(*c["colour"])
                family_colour[c["family"]] = colorsys.hsv_to_rgb(h, 0.75, 0.85)
            out[c["id"]] = family_colour[c["family"]]
        else:
            out[c["id"]] = c["colour"]
    return out


def draw_towns(ax, layers, tables, labels=6, colour="black"):
    """Towns as dots sized by population; the largest few named."""
    size = layers.get("settlement")
    if size is None:
        return
    rows, cols = np.nonzero(size >= TOWN_MIN)
    if not len(rows):
        return
    s = size[rows, cols]
    ax.scatter(cols, rows, s=4 + 10 * np.log10(s / TOWN_MIN + 1) ** 2, c="white", edgecolors=colour,
               linewidths=0.6, zorder=3)
    ids = layers.get("settlement_id")
    places = (tables or {}).get("settlements") or []
    if ids is None or not places:
        return
    h, w = size.shape
    placed = []
    for k in np.argsort(-s)[:labels * 4]:
        r, c = rows[k], cols[k]
        if any(abs(r - pr) < h * 0.05 and abs(c - pc) < w * 0.09 for pr, pc in placed):
            continue          # too close to a label already placed
        sid = int(ids[r, c])
        if 0 < sid < len(places) and places[sid]:
            placed.append((r, c))
            ax.annotate(f"{places[sid]['name']} ({_short_number(s[k])})", (c, r),
                        xytext=(4, 3), textcoords="offset points", fontsize=7, color=colour, zorder=4,
                        bbox={"boxstyle": "round,pad=0.12", "fc": (1, 1, 1, 0.6), "ec": "none"})
            if len(placed) >= labels:
                break


def _short_number(n):
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k" if n >= 1e3 else f"{n:.0f}"


def arrival_map(world, layers, tables=None, background="terrain", year=None):
    """When people first arrived (before year 0), from early (dark) to late (light); grey = nobody yet."""
    from matplotlib import colormaps
    img = base_map(world, background)
    arrival = layers["arrival_year"]
    reached = np.isfinite(arrival)
    if reached.any():
        lo, hi = np.nanmin(arrival[reached]), min(np.nanmax(arrival[reached]), 0)
        t = np.clip((arrival[reached] - lo) / max(hi - lo, 1), 0, 1)
        img[reached] = colormaps["viridis"](t)[:, :3]
    land = ~world.ocean & ~world.lake
    img[land & ~reached] = (0.6, 0.6, 0.6)
    return img


def plot_maps(run_dir, years=None, kind="people", out=None, world=None, dark=None, background=None):
    """A grid of maps of `kind` (see MAP_KINDS) at several years (dark: see style.palette)."""
    from matplotlib.figure import Figure
    from ..world import load_world
    from .style import palette, preferred_background, style_figure
    colours = palette(dark)
    background = background or preferred_background()

    run = RunReader(run_dir)
    world = world or load_world(run.meta["world_seed"], run.meta["world_width"])
    if years is None:
        span = run.last_year - run.first_year
        years = [run.first_year + round(span * f / 5 / 100) * 100 for f in range(6)]
    draw = {"people": people_map, "density": density_map, "cultures": culture_map,
            "families": family_map, "states": state_map, "arrival": arrival_map}[kind]
    tables = run.tables()
    cols = 2
    rows = (len(years) + 1) // cols
    fig = Figure(figsize=(8 * cols, 4.2 * rows))
    axes = fig.subplots(rows, cols, squeeze=False)
    for ax, year in zip(axes.ravel(), years):
        layers = run.layers_at(year)
        ax.imshow(draw(world, layers, tables, background=background, year=year), interpolation="nearest")
        if kind != "arrival":
            draw_towns(ax, layers, tables)
        pop = layers.get("population")
        total = f", {pop.sum() / 1e6:,.1f} million people" if pop is not None else ""
        own = layers.get("state")
        if own is not None and kind == "states":
            total += f", {len(np.unique(own[own > 0]))} states"
        cult = layers.get("culture")
        if cult is not None and kind in ("cultures", "families"):
            alive = np.unique(cult[cult > 0])
            if kind == "cultures":
                total += f", {len(alive)} cultures"
            else:
                fams = {tables["cultures"][c]["family"] for c in alive} if tables.get("cultures") else set()
                total += f", {len(fams)} language families"
        ax.set_title(f"{label(run.snapshot_year(year))}{total}", fontsize=10)
        ax.set_axis_off()
    for ax in axes.ravel()[len(years):]:
        ax.set_visible(False)
    legend = {"people": "gold = foragers, orange = herders, crimson = farmers (stronger = more people)",
              "density": "people per km2 (log scale, dark to bright)",
              "cultures": "cultures (related ones in similar colours); dots = towns",
              "families": "language families; dots = towns",
              "states": "states (darker edges = borders, red = war fronts); dots = towns",
              "arrival": "when people arrived: dark = earliest, light = latest, grey = not yet reached"}[kind]
    fig.suptitle(f"World {run.meta['world_seed']}, history seed {run.meta['sim_seed']}: {legend}", fontsize=11)
    style_figure(fig, axes.ravel(), colours)
    fig.tight_layout()
    out = out or os.path.join(run_dir, f"maps_{kind}.png")
    fig.savefig(out, dpi=80, facecolor=fig.get_facecolor())
    return out
