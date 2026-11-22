"""Map helpers shared by the systems: cell areas, neighbours, the coastline over time."""
import numpy as np
from worldgen.generator import distance_to, shift

DIRECTIONS4 = ((0, 1), (0, -1), (1, 0), (-1, 0))


def cell_area_km2(world):
    kpp = world.grid.km_per_px
    return (kpp * kpp * np.maximum(np.cos(np.radians(world.grid.lat)), 0.01)).astype(np.float32)


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
