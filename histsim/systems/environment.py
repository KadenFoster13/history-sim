"""Environment: the world's changing climate and coastline over history.

Each step it asks worldgen's ClimateHistory for the current year's climate and records:
- `climate`: temperature change and rainfall factor per cell (state.climate)
- `exposed_shelf` layer: today's shallow sea that is still dry land because the sea is
  lower (ice-age land bridges, flooded as the ice melts)
- `drought` layer: land with rainfall at least 25% below normal this year
"""
import numpy as np
from worldgen import ClimateHistory

from ..core.simulation import System


class Environment(System):
    name = "environment"

    def __init__(self, every=1):
        self.every = every
        self.history = None

    def setup(self, state):
        self.history = ClimateHistory(state.world)
        state.add_layer("exposed_shelf", np.bool_)
        state.add_layer("drought", np.bool_)

    def step(self, state):
        world = state.world
        climate = self.history.at(state.year)
        state.climate = climate
        land = ~world.ocean & ~world.lake
        shelf = world.ocean & (world.elevation_m > climate.sea_level_m)
        was = state.cells["exposed_shelf"]
        if was.any() and not shelf.any():
            state.event("environment", "The last land bridges sink beneath the rising sea.")
        state.cells["exposed_shelf"] = shelf
        state.cells["drought"] = land & (climate.precip_factor < 0.75)
        for start, end, lat, lon, radius, severity in climate.droughts:
            if start <= state.year < start + self.every:
                state.event("drought", f"A great drought begins near {abs(lat):.0f}{'N' if lat >= 0 else 'S'} "
                                       f"{abs(lon):.0f}{'E' if lon >= 0 else 'W'}.",
                            lat=lat, lon=lon, years=round(end - start))

        cos_lat = np.cos(np.radians(world.grid.lat))
        land_area = float((land * cos_lat).sum())
        state.stats.update({
            "global_temperature_c": climate.global_temperature_c,
            "sea_level_m": climate.sea_level_m,
            "exposed_shelf_share": float((shelf * cos_lat).sum()) / land_area,
            "drought_share": float((state.cells["drought"] * cos_lat).sum()) / land_area,
            "volcanic_cooling_c": climate.volcanic,
        })
