"""Farming and herding: where wild species are tamed, and how farming spreads.

- Domestication: where a wild species lives and its foragers are crowded, people may
  start taming it. Easy species are tamed sooner, hard ones (the maize-like grain) much
  later; farmers tame new species more readily than foragers. Nothing is tamed in the
  unstable climate before the Holocene settles.
- Improvement: a newly tamed crop yields little at first and improves over centuries.
- Spread: each tamed species spreads to neighbouring cells where it can grow, at
  roughly the real pace (about 1 km a year). Climate changes more north-south than
  east-west, so farming spreads along latitudes more easily. Tsetse zones keep cattle,
  horses and camels out. The sea stops it (until seafaring, in a later phase).
- Food: farming feeds many more people per km2 than foraging; herding suits open,
  dry land. Each cell lives by whichever feeds the most people.
"""
import numpy as np
from worldgen import SPECIES
from worldgen.generator import distance_to

from ..core.calendar import label
from ..core.simulation import System
from .geo import DIRECTIONS4, cell_area_km2, neighbour_count
from .population import FARMING, FORAGING, HERDING, OPENNESS, _by_biome

TSETSE_SENSITIVE = {"cattle", "horse", "camel"}
INDEPENDENT_KM = 1000      # a domestication this far from any farming counts as a new origin


class Farming(System):
    name = "farming"

    def __init__(self):
        self.every = None

    def setup(self, state):
        cfg, world = state.config, state.world
        self.every = cfg.step_years
        self.n = len(SPECIES)
        self.is_crop = np.array([s.kind == "crop" for s in SPECIES])
        self.food = np.array([s.food for s in SPECIES], np.float32)
        self.difficulty = np.array([s.difficulty for s in SPECIES], np.float32)
        self.area = cell_area_km2(world)
        land = ~world.ocean & ~world.lake
        # Where each species can be kept, once tamed.
        self.suit = world.species_suitability.astype(np.float32).copy()
        for i, sp in enumerate(SPECIES):
            if sp.key in TSETSE_SENSITIVE:
                self.suit[i][world.tsetse > cfg.tsetse_block] = 0
        self.suit[:, ~land] = 0
        self.grazing = _by_biome(world.biome, OPENNESS, 0.5)
        # How easily farming spreads into each cell: faster along rivers and coasts (the
        # Danube, the Nile), slower over rough ground, and patchy (local choices, chance).
        from worldgen.noise import Perlin3
        from worldgen.hydrology import cell_river_flow
        from worldgen.noise import fbm
        from .geo import touches
        corridor = (cell_river_flow(world) > 300) | touches(world.ocean)
        patchy = np.exp(0.8 * fbm(Perlin3(world.cfg.seed * 7919 + state.seed * 31 + 271828), world.grid.xyz, 8, 3))
        self.spread_ease = ((1 + 1.5 * corridor) * patchy / (1 + world.slope_m_per_km / 15)).astype(np.float32)
        self.arable = (0.25 + 0.75 * world.arable).astype(np.float32)
        state.add_layer("tamed", np.uint32, record="changes")   # bit i set = uses SPECIES[i]
        state.cells["maturity"] = np.zeros((self.n,) + world.ocean.shape, np.float32)
        state.record_modes["maturity"] = "none"
        state.tables["farming"] = self
        state.tables["domestications"] = []
        self._first_reach = set()
        self._stats(state)

    def has(self, state, i):
        return (state.cells["tamed"] & np.uint32(1 << i)) != 0

    # ------------------------------------------------------------- capacity
    def capacity(self, state, forage, land):
        """People each cell can feed (whole cell), and how they live (subsistence)."""
        cfg = state.config
        climate = state.climate
        tamed = state.cells["tamed"]
        maturity = state.cells["maturity"]
        crop_best = np.zeros(land.shape, np.float32)
        crop_sum = np.zeros(land.shape, np.float32)
        herd = np.zeros(land.shape, np.float32)
        for i in np.flatnonzero(self._present(tamed)):
            here = (tamed & np.uint32(1 << i)) != 0
            value = np.where(here, self.suit[i] * maturity[i] * self.food[i], 0)
            if self.is_crop[i]:
                crop_best = np.maximum(crop_best, value)
                crop_sum += value
            else:
                herd += value
        crops = crop_best + 0.25 * (crop_sum - crop_best)          # a varied package helps
        weather = 1.0
        if climate is not None:
            weather = np.clip(climate.precip_factor, 0.3, 1.2) * (1 - 0.08 * np.maximum(0, -climate.temperature_offset_c))
        farm = cfg.farmer_density * crops * self.arable * weather
        herding = cfg.herder_density * np.minimum(herd, 2.0) * self.grazing * weather
        per_km2 = np.stack([forage / np.maximum(self.area, 1e-6), herding, farm + 0.3 * herding])
        mode = np.argmax(per_km2, axis=0) + 1                     # FORAGING, HERDING, FARMING
        cap = np.where(land, per_km2.max(axis=0) * self.area, 0).astype(np.float32)
        return cap, np.where(cap > 0, mode, 0).astype(np.uint8)

    def _present(self, tamed):
        bits = np.bitwise_or.reduce(tamed.ravel()) if tamed.size else 0
        return np.array([(int(bits) >> i) & 1 for i in range(self.n)], bool)

    # ----------------------------------------------------------------- step
    def step(self, state):
        cfg, world = state.config, state.world
        dt = self.every
        rng = state.rng.get("farming")
        tamed = state.cells["tamed"]
        maturity = state.cells["maturity"]
        pop = state.cells["population"]
        people = pop > 0
        farms_anything = tamed != 0

        # Crops improve over the centuries after taming.
        for i in np.flatnonzero(self._present(tamed)):
            here = (tamed & np.uint32(1 << i)) != 0
            gain = 1 - np.exp(-dt / (cfg.crop_maturity_years * self.difficulty[i]))
            maturity[i] = np.where(here, maturity[i] + (1 - maturity[i]) * gain, maturity[i])

        # Domestication.
        holocene = np.clip((state.year - cfg.holocene_start_year) / 500, 0, 1)
        if state.climate is not None and state.climate.global_temperature_c < -1.5:
            holocene *= 0.1
        if holocene > 0:
            crowding = np.clip(pop / np.maximum(state.cells["forager_capacity"], 1e-6), 0, 1.5)
            for i, sp in enumerate(SPECIES):
                candidates = world.species_native[i] & people & ~self.has(state, i) & (self.suit[i] > 0)
                if not candidates.any():
                    continue
                # Needs real crowding (pressure to find more food), and a species worth
                # the effort: food plants and animals before fibre.
                hazard = (cfg.domestication_rate * holocene * self.suit[i] * crowding ** 3 / sp.difficulty ** 2
                          * (0.15 + sp.food) * np.where(farms_anything, cfg.farmer_domestication_bonus, 1.0))
                chance = 1 - np.exp(-hazard * dt)
                hit = candidates & (rng.random(chance.shape) < chance)
                if hit.any():
                    cell = int(rng.choice(np.flatnonzero(hit)))   # one origin per species per step
                    self._domesticate(state, i, cell)

        # Spread to neighbours.
        for i in np.flatnonzero(self._present(tamed)):
            here = (tamed & np.uint32(1 << i)) != 0
            near = neighbour_count(here)
            targets = ~here & (near > 0) & people & (self.suit[i] > 0.3)
            if not targets.any():
                continue
            chance = 1 - np.exp(-cfg.spread_rate * dt * (near / 4) * (0.5 + self.suit[i]) * self.spread_ease)
            take = targets & (rng.random(chance.shape) < chance)
            if take.any():
                tamed[take] |= np.uint32(1 << i)
                # The newcomers get the variety their neighbours grow (slightly less adapted).
                best = np.zeros_like(maturity[i])
                from worldgen.generator import shift
                for dy, dx in DIRECTIONS4:
                    best = np.maximum(best, shift(np.where(here, maturity[i], 0), dy, dx))
                maturity[i][take] = best[take] * 0.97
                self._first_landmass_events(state, i, take)
        self._stats(state)

    def _domesticate(self, state, i, cell):
        world = state.world
        tamed = state.cells["tamed"]
        r, c = divmod(cell, world.ocean.shape[1])
        farming_now = tamed != 0
        independent = True
        if farming_now.any():
            dist_km = distance_to(farming_now)[0][r, c] * world.grid.km_per_px
            independent = dist_km > INDEPENDENT_KM
        tamed[r, c] |= np.uint32(1 << i)
        state.cells["maturity"][i][r, c] = 0.05
        lat, lon = float(world.grid.lat[r, c]), float(world.grid.lon[r, c])
        sp = SPECIES[i]
        record = {"year": state.year, "species": sp.key, "row": r, "col": c, "lat": lat, "lon": lon,
                  "independent": independent}
        state.tables["domestications"].append(record)
        where = f"{abs(lat):.0f}{'N' if lat >= 0 else 'S'} {abs(lon):.0f}{'E' if lon >= 0 else 'W'}"
        verb = "First farming in the world" if not state.stats.get("domestications") else (
            "A new, independent beginning of farming" if independent else "Farmers tame another species")
        state.event("domestication", f"{verb}: people near {where} tame the {sp.label}.",
                    species=sp.key, lat=lat, lon=lon, independent=independent)

    def _first_landmass_events(self, state, i, take):
        pass  # landmass-level spread events come with named regions in phase 2

    def _stats(self, state):
        tamed = state.cells["tamed"]
        records = state.tables["domestications"]
        land = ~state.world.ocean & ~state.world.lake
        crop_bits = sum(1 << i for i in range(self.n) if self.is_crop[i])
        state.stats.update({
            "domestications": len(records),
            "farming_origins": sum(1 for r in records if r["independent"]),
            "species_tamed": len({r["species"] for r in records}),
            "farmed_land_share": float(((tamed & np.uint32(crop_bits)) != 0)[land].mean()),
            "herded_or_farmed_land_share": float((tamed != 0)[land].mean()),
        })
