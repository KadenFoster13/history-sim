"""Environment: the world's changing climate and coastline over history.

Each step it asks worldgen's ClimateHistory for the current year's climate and records:
- `climate`: temperature change and rainfall factor per cell (state.climate)
- `exposed_shelf` layer: today's shallow sea that is still dry land because the sea is
  lower (ice-age land bridges, flooded as the ice melts)
- `drought` layer: land with rainfall at least 25% below normal this year
The maps (and these layers) are rebuilt every `step_years`, when the systems that read
them run; global temperature, sea level and drought events are kept every year.
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
        world = state.world
        self.history = ClimateHistory(world)
        state.add_layer("exposed_shelf", np.bool_)
        state.add_layer("drought", np.bool_)
        # The climate's maps are only read by the systems that run every `step_years`, so
        # they are rebuilt on those years only; the global numbers are kept every year.
        self.detail_every = state.config.step_years
        self._first = None
        land = ~world.ocean & ~world.lake
        self.cos_lat = np.cos(np.radians(world.grid.lat))
        self.land_area = float((land * self.cos_lat).sum())
        # Ice-age land still above sea, for any sea level: ocean cells by height.
        heights = world.elevation_m[world.ocean]
        order = np.argsort(heights)
        self._shelf_heights = heights[order]
        weights = self.cos_lat[world.ocean][order]
        self._shelf_area = np.concatenate([np.cumsum(weights[::-1])[::-1], [0.0]])   # area above each height

    def _shelf_share(self, sea_level_m):
        above = np.searchsorted(self._shelf_heights, sea_level_m, side="right")
        return float(self._shelf_area[above]) / self.land_area

    def step(self, state):
        world = state.world
        year = state.year
        if self._first is None:
            self._first = year
        history = self.history
        if (year - self._first) % self.detail_every == 0 or state.climate is None:
            climate = history.at(year)
            state.climate = climate
            land = ~world.ocean & ~world.lake
            state.cells["exposed_shelf"] = world.ocean & (world.elevation_m > climate.sea_level_m)
            state.cells["drought"] = land & (climate.precip_factor < 0.75)
            state.stats["drought_share"] = float((state.cells["drought"] * self.cos_lat).sum()) / self.land_area
            g, sea, volcanic = climate.global_temperature_c, climate.sea_level_m, climate.volcanic
        else:
            from worldgen.climate_history import sea_level_m
            g, sea, volcanic = history.global_temperature(year), sea_level_m(year), history.volcanic_cooling(year)
        shelf_share = self._shelf_share(sea)
        if state.stats.get("exposed_shelf_share", 0) > 0 and shelf_share == 0:
            state.event("environment", "The last land bridges sink beneath the rising sea.")
        for start, end, lat, lon, radius, severity in history.droughts:
            if start <= year < start + self.every:
                state.event("drought", f"A great drought begins near {abs(lat):.0f}{'N' if lat >= 0 else 'S'} "
                                       f"{abs(lon):.0f}{'E' if lon >= 0 else 'W'}.",
                            lat=lat, lon=lon, years=round(end - start))
        state.stats.update({
            "global_temperature_c": g,
            "sea_level_m": sea,
            "exposed_shelf_share": shelf_share,
            "volcanic_cooling_c": volcanic,
        })
