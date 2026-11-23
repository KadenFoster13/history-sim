"""Enhance: show part of the history map in finer detail.

The world generator regenerates the chosen area 2-8x finer (coastlines, rivers, relief:
see worldgen.region). History is simulated on the world map's cells, so each fine pixel
takes the state, culture, way of life... of a nearby world cell. Which cell is chosen
wobbles with smooth noise, so borders come out as natural wandering lines instead of
blocky steps. The wobble is small (about a quarter of a cell, never more than 0.6), so
each pixel shows its own cell unless it lies near the cell's edge: borders move by at most
part of a cell (~30 km on an 800-px world), never whole cells. History itself is always
simulated on the world map's cells; enhancing only changes the picture.
Coastlines come from the finer map: new bits of shore take the nearest land's values.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

PAD = 3                      # world cells around the area, so wobbles can reach past its edge
WOBBLE = ((2.5, 0.2), (0.9, 0.13), (0.35, 0.06))   # (noise spacing, amplitude), in world cells
MAX_WOBBLE = 0.6             # world cells: a pixel never shows a cell further than this from its own
COUNTED = ("population",)    # people per cell: shared out among the fine pixels


class EnhancedArea:
    """Maps a region (worldgen.generate_region) back to the world cells it shows."""

    def __init__(self, world, region):
        self.region = region
        box = region.box
        f = box.factor
        h, w = world.ocean.shape
        self.shape = (h, w)
        self.factor = f
        rows = np.clip(np.arange(box.row0 - PAD, box.row0 + box.rows + PAD), 0, h - 1)
        cols = (box.col0 + np.arange(-PAD, box.cols + PAD)) % w
        index = rows[:, None] * w + cols[None, :]
        water = (world.ocean | world.lake)[np.ix_(rows, cols)]
        land_fill = _nearest(index, water)          # water cells -> the nearest land cell
        water_fill = _nearest(index, ~water)        # land cells -> the nearest water cell

        # Where each fine pixel's centre sits, in (padded) world cells, plus the wobble.
        fh, fw = box.rows * f, box.cols * f
        y = (np.arange(fh) + 0.5) / f - 0.5 + PAD
        x = (np.arange(fw) + 0.5) / f - 0.5 + PAD
        yy, xx = np.meshgrid(y, x, indexing="ij")
        rng = np.random.default_rng([box.row0, box.col0 % w, f, 7])
        dy, dx = _wobble(rng, yy, xx), _wobble(rng, yy, xx)
        np.clip(dy, -MAX_WOBBLE, MAX_WOBBLE, out=dy)
        np.clip(dx, -MAX_WOBBLE, MAX_WOBBLE, out=dx)
        r = np.clip(np.rint(yy + dy), 0, len(rows) - 1).astype(np.intp)
        c = np.clip(np.rint(xx + dx), 0, len(cols) - 1).astype(np.intp)
        fine_water = region.ocean | region.lake
        self.index = np.where(fine_water, water_fill[r, c], land_fill[r, c])

    def layers(self, layers):
        """The run's layers (world-sized arrays) as seen in the enhanced area."""
        out = {}
        flat = self.index.ravel()
        for name, layer in layers.items():
            if not isinstance(layer, np.ndarray) or layer.shape != self.shape:
                continue
            fine = layer.ravel()[flat].reshape(self.index.shape)
            if name in COUNTED:
                fine = fine / self.factor ** 2
            out[name] = fine
        return out


def _nearest(index, empty):
    """`index` with every `empty` cell replaced by the index of the nearest non-empty cell."""
    if empty.all() or not empty.any():
        return index
    _, (iy, ix) = distance_transform_edt(empty, return_indices=True)
    return index[iy, ix]


def _wobble(rng, yy, xx):
    """Smooth noise (in world cells) at the given positions."""
    total = np.zeros(yy.shape, np.float32)
    for spacing, amplitude in WOBBLE:
        gh = int(yy.max() / spacing) + 4
        gw = int(xx.max() / spacing) + 4
        grid = rng.normal(0, 1, (gh, gw))
        total += amplitude * map_coordinates(grid, [yy / spacing, xx / spacing], order=3, mode="nearest")
    return total
