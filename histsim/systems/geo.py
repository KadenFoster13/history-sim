"""Map helpers shared by the systems: cell areas, neighbours, the coastline over time."""
import numpy as np
from worldgen.generator import distance_to, shift

DIRECTIONS4 = ((0, 1), (0, -1), (1, 0), (-1, 0))


def cell_area_km2(world):
    kpp = world.grid.km_per_px
    return (kpp * kpp * np.maximum(np.cos(np.radians(world.grid.lat)), 0.01)).astype(np.float32)


def cell_scale(world, cfg):
    """How big this world's cells are compared with the 800-px world the rules were tuned
    on (1.0 there, 0.5 at 1600 px). Rules that work cell by cell use it to stay the same in
    km and km2 at any map size:
      speeds (a front moving one cell at a time)   rate / scale
      diffusion (migration between neighbours)     rate / scale**2
      chances per cell (per km2 of land)           rate * scale**2
      chances per frontier cell (per km of front)  rate * scale
      sizes counted in cells                       cells / scale**2
    """
    return float(cfg.reference_width / world.ocean.shape[1])


def rounds_for(scale):
    """Rounds per step for rules that move a front one cell at a time: on finer maps a
    front must cross more cells in the same years (one round can't cross two cells)."""
    return max(1, int(round(1 / scale)))


def band(values, scale):
    """A river or coast mask (or values) as a band of the same width in km as on the 800-px
    world, where it is one ~50 km cell wide: widened on finer maps; on coarser maps a cell
    is only partly in the band, so the values (as floats) are scaled down by cell width."""
    if scale > 1:
        return np.asarray(values).astype(np.float32) / scale
    r = max(0, int(round(1 / scale)) - 1)
    if r == 0:
        return values
    from scipy.ndimage import maximum_filter
    out = maximum_filter(np.asarray(values).astype(np.float32), size=2 * r + 1, mode=("nearest", "wrap"))
    return out.astype(np.asarray(values).dtype)


def cells_for(cells, scale):
    """A size in cells tuned on the 800-px world, at this world's cell size."""
    return max(1, int(round(cells / scale ** 2)))


def neighbour_count(mask):
    """How many of each cell's 4 neighbours are in `mask` (wraps east-west)."""
    m = mask.astype(np.int8)
    return sum(shift(m, dy, dx) for dy, dx in DIRECTIONS4)


def touches(mask):
    return neighbour_count(mask) > 0


def land_now(world, sea_level_m):
    """Dry land at a given sea level: today's land plus shallow sea exposed when the sea is lower."""
    land = ~world.ocean & ~world.lake
    if sea_level_m < 0:
        land = land | (world.ocean & (world.elevation_m > sea_level_m))
    return land


class ShelfSource:
    """For each sea cell, the nearest land cell: exposed sea floor (ice-age coastal plains)
    borrows its climate and vegetation from it."""

    def __init__(self, world):
        _, iy, ix = distance_to(~world.ocean)
        self.index = iy * world.ocean.shape[1] + ix

    def fill(self, values, world):
        """`values` with every sea cell replaced by its nearest land cell's value."""
        out = values.copy()
        flat = out.reshape(-1)
        sea = world.ocean.ravel()
        flat[sea] = values.reshape(-1)[self.index.ravel()[sea]]
        return out


# ------------------------------------------------------------ travel graph
EARTH_RADIUS_KM = 6371.0


def cell_graph(world, cost, mask):
    """A graph of the map for shortest-path work: an edge between each pair of 4-neighbour
    cells in `mask` (wrapping east-west), weighted km x the average `cost` of the two.

    Returns (graph, edges): a scipy sparse matrix over all cells (row-major index) and
    the (a, b) cell-index arrays of its edges, each pair listed once.
    """
    from scipy.sparse import coo_matrix

    h, w = mask.shape
    idx = np.arange(h * w).reshape(h, w)
    kpp = world.grid.km_per_px
    cos_lat = np.maximum(np.cos(np.radians(world.grid.lat)), 0.05)
    east = mask & np.roll(mask, -1, axis=1)
    south = mask[:-1] & mask[1:]
    a = np.concatenate([idx[east], idx[:-1][south]])
    b = np.concatenate([np.roll(idx, -1, axis=1)[east], idx[1:][south]])
    wt = np.concatenate([(kpp * cos_lat * 0.5 * (cost + np.roll(cost, -1, axis=1)))[east],
                         (kpp * 0.5 * (cost[:-1] + cost[1:]))[south]])
    wt = np.maximum(wt, 1e-3)
    n = h * w
    graph = coo_matrix((np.concatenate([wt, wt]), (np.concatenate([a, b]), np.concatenate([b, a]))),
                       shape=(n, n)).tocsr()
    return graph, (a, b)


def nearest_seed(graph, seeds, n):
    """For every cell, which seed (index into `seeds`) is closest along the graph (-1 = none)."""
    from scipy.sparse.csgraph import dijkstra

    seeds = np.asarray(seeds, int)
    if not len(seeds):
        return np.full(n, -1)
    _, _, sources = dijkstra(graph, directed=False, indices=seeds, min_only=True, return_predecessors=True)
    label = np.full(n, -1)
    found = sources >= 0
    lookup = {int(s): i for i, s in enumerate(seeds)}
    label[found] = np.vectorize(lookup.get)(sources[found])
    return label


def subgraph(graph, edges, keep):
    """`graph` with only the edges whose two ends satisfy keep(a, b) (arrays of cell indices)."""
    from scipy.sparse import coo_matrix

    a, b = edges
    ok = keep(a, b)
    data = np.asarray(graph[a[ok], b[ok]]).ravel()
    n = graph.shape[0]
    return coo_matrix((np.concatenate([data, data]), (np.concatenate([a[ok], b[ok]]),
                                                      np.concatenate([b[ok], a[ok]]))), shape=(n, n)).tocsr()


def great_circle_km(xyz_a, xyz_b):
    """Distance over the globe between unit vectors (arrays with a last axis of 3)."""
    chord = np.linalg.norm(np.asarray(xyz_a) - np.asarray(xyz_b), axis=-1)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2, 0, 1))


def spaced_sample(rng, weight, n, xyz, min_km, tries=20):
    """Up to `n` cells drawn by `weight` (flat), no two closer than `min_km`."""
    flat = np.asarray(weight, float).ravel()
    total = flat.sum()
    if total <= 0 or n <= 0:
        return []
    candidates = rng.choice(flat.size, size=min(flat.size, n * tries), p=flat / total)
    chosen, points = [], []
    for c in candidates:
        p = xyz[c]
        if points and great_circle_km(np.array(points), p).min() < min_km:
            continue
        chosen.append(int(c))
        points.append(p)
        if len(chosen) >= n:
            break
    return chosen
