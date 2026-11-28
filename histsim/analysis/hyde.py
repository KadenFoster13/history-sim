"""Compare Earth runs with HYDE 3.3 and the blended regional targets (`histsim score`).

HYDE 3.3 (History Database of the Global Environment; Klein Goldewijk, Utrecht University,
CC-BY) maps population on a 5-arc-minute grid back to 10,000 BCE. Its files are not shipped
here: download the baseline `<year>_pop.zip` files (e.g. 1300AD_pop.zip) from
https://doi.org/10.24416/UU01-94FNH0 (folder hyde33_c7_base_mrt2023/zip) into one folder,
unzipped or not. The site blocks scripted downloads, so fetch them in a browser.

HYDE is itself partly modelled (it spreads the same regional totals over land by a
suitability map), so it says more about where people lived inside a region than about
regional totals. `score` therefore compares totals with the blended targets
(analysis/earth_targets.py) and the pattern with HYDE rescaled to those targets.
"""
import os
import zipfile

import numpy as np

from .earth_targets import REGIONS, region_owner, targets_1300, world_1300


def _hyde_name(year):
    """HYDE's file label for a real year: 1300 -> 1300AD, 0 -> 0AD, -1000 -> 1000BC."""
    return f"{year}AD" if year >= 0 else f"{-year}BC"


def _regrid(data, cellsize, width):
    """Sum a global lat/lon grid (rows from 90 N, columns from 180 W) into a width x width/2 map."""
    nr, nc = data.shape
    d = 360.0 / width
    rows = np.clip(((np.arange(nr) + 0.5) * cellsize / d).astype(int), 0, width // 2 - 1)
    cols = np.clip(((np.arange(nc) + 0.5) * cellsize / d).astype(int), 0, width - 1)
    rs = np.flatnonzero(np.r_[True, np.diff(rows) > 0])
    cs = np.flatnonzero(np.r_[True, np.diff(cols) > 0])
    out = np.zeros((width // 2, width), np.float64)
    out[np.ix_(rows[rs], cols[cs])] = np.add.reduceat(np.add.reduceat(data.astype(np.float64), cs, axis=1), rs, axis=0)
    return out


def _read_asc(text):
    lines = text.split("\n", 6)
    hdr = {k.lower(): float(v) for k, v in (ln.split() for ln in lines[:6])}
    data = np.array(lines[6].split(), np.float32)
    data = data.reshape(int(hdr["nrows"]), int(hdr["ncols"]))
    data[data < 0] = 0
    return data, hdr["cellsize"]


def load_hyde(folder, year, width=800, cache_dir=None):
    """People per map cell from HYDE's `popc` grid for a real year (e.g. 1300), on a map
    `width` pixels wide. Reads popc_<year>.asc, or the same inside <year>_pop.zip."""
    name = _hyde_name(year)
    cache = os.path.join(cache_dir, f"hyde_{name}_{width}.npy") if cache_dir else None
    if cache and os.path.exists(cache):
        return np.load(cache)
    asc = os.path.join(folder, f"popc_{name}.asc")
    if os.path.exists(asc):
        with open(asc) as f:
            text = f.read()
    else:
        with zipfile.ZipFile(os.path.join(folder, f"{name}_pop.zip")) as z:
            text = z.read(f"popc_{name}.asc").decode("ascii")
    data, cellsize = _read_asc(text)
    grid = _regrid(data, cellsize, width).astype(np.float32)
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache, grid)
    return grid


def blended_map(hyde, targets=None):
    """HYDE rescaled region by region so each region sums to its blended target."""
    targets = targets or targets_1300()
    owner = region_owner(hyde.shape)
    out = hyde.astype(np.float64).copy()
    for name in REGIONS:
        m = owner == name
        total = out[m].sum()
        if total > 0:
            out[m] *= targets[name]["target"] * 1e6 / total
    return out


def score(sim_population, hyde, ocean=None):
    """Score a mean population map against the 1300 targets and HYDE's pattern.
    Returns (lines of text, dict of numbers)."""
    t = targets_1300()
    owner = region_owner(sim_population.shape)
    blend = blended_map(hyde, t)
    lines = [f"{'region':<26}{'sim':>7}{'target':>8}{'band':>14}{'all sources':>15}{'sim/target':>11}  conf    in band"]
    errors, in_range = [], 0
    for name in REGIONS:
        m = owner == name
        s, r = sim_population[m].sum() / 1e6, t[name]
        ok = r["lo"] * 0.9 <= s <= r["hi"] * 1.1
        in_range += ok
        errors.append(abs(np.log2(max(s, 0.01) / r["target"])))
        lines.append(f"{name:<26}{s:7.1f}{r['target']:8.1f}{r['lo']:7.1f}-{r['hi']:<6.1f}{r['min']:8.1f}-{r['max']:<6.1f}"
                     f"{s / r['target']:11.2f}  "
                     f"{r['confidence']:<7} {'yes' if ok else 'no'}")
    wt = world_1300()
    total = sim_population.sum() / 1e6
    land = ~ocean if ocean is not None else np.ones(sim_population.shape, bool)
    from scipy.ndimage import uniform_filter
    s2, b2 = uniform_filter(sim_population.astype(np.float64), 5), uniform_filter(blend, 5)
    overlap = float(np.minimum(s2 / s2.sum(), b2 / b2.sum())[land].sum())
    nums = {"world": total, "regions_in_range": in_range, "mean_log2_error": float(np.mean(errors)),
            "pattern_overlap": overlap}
    lines.append(f"world: sim {total:.0f}M, target {wt['target']:.0f}M (band {wt['lo']:.0f}-{wt['hi']:.0f}, sources {wt['min']:.0f}-{wt['max']:.0f})")
    lines.append(f"regions in band: {in_range}/{len(REGIONS)}; mean error {2 ** nums['mean_log2_error']:.2f}x "
                 f"(|log2| {nums['mean_log2_error']:.2f}); pattern overlap at ~2 degrees {overlap:.2f} (1 = same)")
    return lines, nums


def ratio_map(sim_population, hyde, ocean, out, title):
    """Map of sim / HYDE (rescaled to the targets), red where the sim is denser."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter
    h, w = sim_population.shape
    d = 360.0 / w
    area = (d * 111.2) ** 2 * np.cos(np.radians(90 - (np.arange(h) + 0.5) * d))[:, None]
    blend = blended_map(hyde)
    s2, b2 = uniform_filter(sim_population.astype(np.float64), 3), uniform_filter(blend, 3)
    ratio = np.where(~ocean, np.log2((s2 + area * 0.2) / (b2 + area * 0.2)), np.nan)
    top, bottom = int(h * 0.05), int(h * 0.82)
    fig, ax = plt.subplots(figsize=(16, 7.5))
    im = ax.imshow(ratio[top:bottom], cmap="RdBu_r", vmin=-3, vmax=3, extent=[-180, 180, 90 - bottom * d, 90 - top * d])
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_ticks([-3, -2, -1, 0, 1, 2, 3])
    cb.set_ticklabels(["1/8", "1/4", "1/2", "same", "2x", "4x", "8x"])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)
