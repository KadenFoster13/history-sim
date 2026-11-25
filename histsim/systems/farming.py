"""Farming and herding: where wild species are tamed, and how farming spreads.

- Domestication: where a wild species lives and its foragers are crowded, people may
  start taming it. Easy species are tamed sooner, hard ones (the maize-like grain) much
  later; farmers tame new species more readily than foragers. Nothing is tamed in the
  unstable climate before the Holocene settles.
- Improvement: a newly tamed crop yields little at first and improves over centuries.
- River water: on floodplains crops grow on the receding flood whatever the rain (the
  Nile, Mesopotamia, the Indus), and the yearly silt makes them yield up to twice as
  much. Where people know canals (histsim/tech.py) this extends to dry irrigable land.
- Farming technology raises what land can be intensified to: the plough and manuring on
  rain-fed fields, canals on dry irrigable land, terraces on steep land.
- Spread: each tamed species spreads to neighbouring cells where it can grow, at
  roughly the real pace (about 1 km a year on average) but unevenly, as on Earth:
  * fast along rivers and coasts and over good soil (the first farmers crossed central
    Europe's loess valleys in two or three centuries);
  * by pioneers who settle good river or coast land 50-300 km ahead of the front;
  * slowly where foragers live well (rich coasts, lakes, rivers): farming stalls at
    such frontiers, sometimes for a thousand years, until its crops clearly beat
    foraging (as on the edge of northern Europe).
  Climate changes more north-south than east-west, so farming spreads along latitudes
  more easily. Tsetse zones keep cattle, horses and camels out. The sea stops it (until
  seafaring, in a later phase).
- Food: farming feeds many more people per km2 than foraging; herding suits open,
  dry land. Each cell lives by whichever feeds the most people.
- Shifting and settled farming: farming begins as shifting (slash-and-burn) fields with
  long fallows, feeding few people per km2. Where farmers grow dense and their land
  fills up they work it harder over the centuries (Boserup: shorter fallow, ploughs,
  manure, terraces, irrigation), so `intensity` (0..1) rises and the land feeds more: up
  to 6x on rain-fed fields, 30x on floodplains and on dry irrigable land a state holds
  (canals), and more again for rice-like paddies. The denser the farmers, the faster it
  goes, so dense places run ahead while thin land stays shifting; methods spread from
  neighbours, and states speed it up. Intensity fades where land empties or is
  abandoned, and a little when it is fought over, so collapses lower what the land can
  feed for centuries.
"""
from types import SimpleNamespace

import numpy as np
from worldgen import SPECIES
from worldgen.generator import distance_to

from ..core.calendar import label
from ..core.simulation import System
from .geo import DIRECTIONS4, band, cell_area_km2, cell_scale, neighbour_count, rounds_for
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
        self.scale = cell_scale(world, cfg)
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
        # River water: crops on floodplains (and, with canals, irrigable land) don't need rain.
        self.flood = getattr(world, "floodplain", None)
        self.flood = np.zeros(land.shape, np.float32) if self.flood is None else self.flood.astype(np.float32)
        self.flood = np.where(land, band(self.flood, self.scale), 0).astype(np.float32)   # same width in km
        self.irrigable = getattr(world, "irrigable", None)
        self.irrigable = self.flood if self.irrigable is None else self.irrigable.astype(np.float32)
        self.suit_dry = self.suit.copy()
        self.suit_watered = _watered_suitability(world, self.is_crop, land)
        watered_arable = (world.soil_fertility * np.clip(1 - world.slope_m_per_km / 60, 0, 1)
                          * np.clip(world.warm_months / 6, 0, 1) * (1 - 0.6 * world.wetland))
        self.arable_dry = world.arable.astype(np.float32)
        self.arable_watered = np.where(land, watered_arable, 0).astype(np.float32)
        self._canals = None
        self.world_native = world.species_native
        self._apply_water(cfg)
        # How easily farming spreads into each cell: faster along rivers and coasts (the
        # Danube, the Nile) and over good soil, slower over rough ground, and patchy
        # (local choices, chance).
        from worldgen.noise import Perlin3
        from worldgen.hydrology import cell_river_flow
        from worldgen.noise import fbm
        from .geo import touches
        flow = cell_river_flow(world)
        river = band(np.clip(np.log10(np.maximum(flow, 1) / 100) / 1.5, 0, 1), self.scale)  # 100 m3/s -> 0, ~3,000 -> 1
        coast = np.where(land, band(touches(world.ocean), self.scale), 0)
        self.corridor = (river > 0.2 / max(self.scale, 1)) | (coast > 0)
        patchy = np.exp(0.8 * fbm(Perlin3(world.cfg.seed * 7919 + state.seed * 31 + 271828), world.grid.xyz, 8, 3))
        self.spread_ease = ((1 + cfg.river_corridor * river + cfg.coast_corridor * coast)
                            * (0.15 + self.arable_eff) ** 1.5 * patchy
                            / (1 + world.slope_m_per_km / 15)).astype(np.float32)
        self.landmass = world.landmass_id
        self.km_per_px = world.grid.km_per_px
        self.cos_lat = np.maximum(np.cos(np.radians(world.grid.lat[:, 0])), 0.05)
        self.slope = world.slope_m_per_km.astype(np.float32)
        # Canals transform dry land; where rain already waters the fields they add little.
        self.dryness = np.clip((1000 - world.precip_mm) / 600, 0, 1).astype(np.float32)
        self.wet_grain = next((i for i, sp in enumerate(SPECIES) if sp.key == "wet_grain"), None)
        state.add_layer("tamed", np.uint32, record="changes")   # bit i set = uses SPECIES[i]
        state.add_layer("intensity", np.float32, record="snapshot")
        state.cells["maturity"] = np.zeros((self.n,) + world.ocean.shape, np.float32)
        state.record_modes["maturity"] = "none"
        state.tables["farming"] = self
        state.tables["domestications"] = []
        self._first_reach = set()
        self._stats(state)

    def _apply_water(self, cfg, canal_cells=None):
        """Crop suitability and farmland with river water: floodplains always, irrigable
        land where people dig canals (`canal_cells`: the canals technology, per cell) or as
        far as `canal_irrigation` (0..1, everywhere) says."""
        canals = float(getattr(cfg, "canal_irrigation", 0.0))
        key = (canals, None if canal_cells is None else int(canal_cells.sum()))
        if key == self._canals:
            return
        self._canals = key
        dug = canals if canal_cells is None else np.maximum(canals, canal_cells.astype(np.float32))
        # A real floodplain (0.4+) waters its fields fully; canals as far as they reach.
        water = np.maximum(np.clip(self.flood / 0.4, 0, 1), dug * self.irrigable)
        watered = self.suit_watered * water[None]
        self.suit = np.where(self.is_crop[:, None, None], np.maximum(self.suit_dry, watered), self.suit_dry)
        self.can_tame = self.world_native & (self.suit > 0)    # where each species could be tamed
        arable = np.maximum(self.arable_dry, self.arable_watered * water)
        self.arable_eff = arable.astype(np.float32)
        self.arable = (0.25 + 0.75 * arable).astype(np.float32)

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
        # Floodplains renew their soil every year (silt): up to (1 + flood_yield) times the harvest.
        farm = cfg.farmer_density * crops * self.arable * weather * (1 + cfg.flood_yield * self.flood)
        intensity = state.cells.get("intensity")
        if intensity is not None:
            farm = farm * (1 + (self.ceiling(state) - 1) * intensity)
        herding = cfg.herder_density * np.minimum(herd, 2.0) * self.grazing * weather
        per_km2 = np.stack([forage / np.maximum(self.area, 1e-6), herding, farm + 0.3 * herding])
        mode = np.argmax(per_km2, axis=0) + 1                     # FORAGING, HERDING, FARMING
        cap = np.where(land, per_km2.max(axis=0) * self.area, 0).astype(np.float32)
        return cap, np.where(cap > 0, mode, 0).astype(np.uint8)

    def knows(self, state, key):
        """Cells whose people know a technology (False everywhere without the technology system)."""
        tech = state.cells.get("tech")
        if tech is None:
            return np.zeros(state.shape, bool)
        from ..tech import bit
        return (tech & np.uint64(bit(key))) != 0

    def ceiling(self, state):
        """The most each cell's farmland can be intensified (x what it feeds unintensified):
        rain-fed fields up to `rainfed_ceiling`, more with the plough and with manuring;
        floodplains, and dry irrigable land where canals are dug, up to `irrigated_ceiling`;
        more for rice-like paddies; less on poor or steep land (terraces halve that)."""
        cfg = state.config
        canals = self.irrigable * self.dryness * self.knows(state, "canals")
        water = np.maximum(np.clip(self.flood / 0.4, 0, 1), canals)
        steep = self.slope / np.where(self.knows(state, "terraces"), 2 * cfg.steep_slope, cfg.steep_slope)
        quality = np.clip(self.arable_eff / 0.6, 0.3, 1) * np.clip(1 - steep, 0.3, 1)
        rain = (cfg.rainfed_ceiling - 1) * (1 + cfg.plough_ceiling * self.knows(state, "plough")
                                            + cfg.manuring_ceiling * self.knows(state, "manuring"))
        top = (rain + (cfg.irrigated_ceiling - 1 - rain) * water) * quality
        if self.wet_grain is not None:
            paddy = np.where(self.has(state, self.wet_grain), np.clip(self.suit[self.wet_grain], 0, 1), 0)
            top = top * (1 + (cfg.paddy_ceiling - 1) * paddy)
        return (1 + top).astype(np.float32)

    def _intensify(self, state):
        """Crowded farmland is worked harder; emptied or abandoned land lets it go."""
        from worldgen.generator import shift
        cfg, dt = state.config, self.every
        intensity = state.cells["intensity"]
        cap = state.cells.get("capacity")
        mode = state.cells.get("subsistence")
        if cap is None or mode is None:
            return
        pop = state.cells["population"]
        farming = (mode == FARMING) & (pop > 0)
        pressure = np.where(cap > 0, pop / np.maximum(cap, 1e-6), 0)
        # Farmers intensify once their land is filling up, and the more of them there are per
        # km2 the faster (Boserup: density drives it, and intensive land holds more people,
        # so dense places run ahead while thin shifting-farming land stays as it is).
        filling = np.clip((pressure - cfg.intensify_pressure) / 0.1 + 0.5, 0, 1)
        density = pop / np.maximum(self.area, 1e-6)
        push = filling * np.clip(density / cfg.intensify_density, 0, 3)
        lead = np.zeros_like(intensity)
        for dy, dx in DIRECTIONS4:
            lead = np.maximum(lead, shift(intensity, dy, dx) - intensity)
        speed = 1 + cfg.intensity_neighbour * np.clip(lead, 0, 1)
        own = state.cells.get("state")
        if own is not None:
            speed = speed * np.where(own > 0, 1 + cfg.state_intensify, 1)
        # Slow to start, then faster as methods, works and markets build on each other.
        gain = dt / cfg.intensify_years * push * speed * (cfg.intensity_start + intensity)
        grown = intensity + np.minimum(gain, 1) * (1 - intensity)
        # Emptied land (people well below what it feeds) or land no longer farmed decays.
        slack = np.where(farming, np.clip((0.5 - pressure) / 0.5, 0, 1), 1.0)
        kept = 0.5 ** (dt / cfg.intensity_decay_years * slack)
        intensity[:] = np.where(farming, grown, intensity) * kept

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

        self._intensify(state)

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
            tamed_flat = tamed.ravel()
            for i, sp in enumerate(SPECIES):
                # Only cells where it can happen (worked out cell by cell; the random draw
                # still covers the whole map, so runs stay the same as before).
                cells = np.flatnonzero(self.can_tame[i] & people)
                cells = cells[(tamed_flat[cells] & np.uint32(1 << i)) == 0]
                if not len(cells):
                    continue
                # Needs real crowding (pressure to find more food), and a species worth
                # the effort: food plants and animals before fibre.
                hazard = (cfg.domestication_rate * self.scale ** 2 * holocene * self.suit[i].ravel()[cells]
                          * crowding.ravel()[cells] ** 3 / sp.difficulty ** 2 * (0.15 + sp.food)
                          * np.where(farms_anything.ravel()[cells], cfg.farmer_domestication_bonus, 1.0))
                chance = 1 - np.exp(-hazard * dt)
                hit = cells[rng.random(tamed.shape).ravel()[cells] < chance]
                if len(hit):
                    cell = int(rng.choice(hit))   # one origin per species per step
                    self._domesticate(state, i, cell)

        # Canals water dry irrigable land where they are known (rechecked every 50 years).
        if state.year - getattr(self, "_water_year", -10 ** 9) >= 50:
            self._water_year = state.year
            self._apply_water(cfg, self.knows(state, "canals") if "tech" in state.cells else None)
        # Spread to neighbours (in several rounds on fine maps: see geo.rounds_for).
        from worldgen.generator import shift
        forage = state.cells["forager_capacity"] / np.maximum(self.area, 1e-6)
        rounds = rounds_for(self.scale)
        pace = 1 / (self.scale * rounds)
        for i in np.flatnonzero(self._present(tamed)):
            here = (tamed & np.uint32(1 << i)) != 0          # where it was before this step
            for _ in range(rounds):
                self._spread(state, i, pace, forage, people, farms_anything, rng)
            self._pioneers(state, i, here, people, forage, farms_anything, rng)
        self._stats(state)

    def _spread(self, state, i, pace, forage, people, farms_anything, rng):
        """One round of species i spreading to neighbouring cells."""
        from worldgen.generator import shift
        cfg, dt = state.config, self.every
        tamed = state.cells["tamed"]
        maturity = state.cells["maturity"]
        here = (tamed & np.uint32(1 << i)) != 0
        near = neighbour_count(here)
        targets = ~here & (near > 0) & people & (self.suit[i] > 0.3)
        if targets.any():
            # The variety the neighbours grow (newcomers get it slightly less adapted).
            best = np.zeros_like(maturity[i])
            for dy, dx in DIRECTIONS4:
                best = np.maximum(best, shift(np.where(here, maturity[i], 0), dy, dx))
            welcome = self._welcome(cfg, i, best, forage, farms_anything)
            chance = 1 - np.exp(-cfg.spread_rate * pace * dt * (near / 4) * (0.5 + self.suit[i])
                                * self.spread_ease * welcome)
            take = targets & (rng.random(chance.shape) < chance)
            if take.any():
                tamed[take] |= np.uint32(1 << i)
                maturity[i][take] = best[take] * 0.97
                self._first_landmass_events(state, i, take)

    def _potential(self, cfg, i, maturity):
        """People per km2 species i could feed where grown at this maturity (alone)."""
        value = self.suit[i] * maturity * self.food[i]
        if self.is_crop[i]:
            return cfg.farmer_density * value * self.arable * (1 + cfg.flood_yield * self.flood)
        return cfg.herder_density * np.minimum(value, 2.0) * self.grazing

    def _welcome(self, cfg, i, maturity, forage, farms_anything):
        """How readily people take up species i (0..1): farmers and herders add new species
        gladly; foragers switch only once it would clearly feed more people than foraging
        (rich foragers on coasts, lakes and rivers hold out longest)."""
        ratio = self._potential(cfg, i, maturity) / np.maximum(forage, 1e-6)
        doubt = np.clip((ratio - 1) / (cfg.farming_advantage - 1), cfg.frontier_floor, 1)
        return np.where(farms_anything, 1.0, doubt).astype(np.float32)

    def _pioneers(self, state, i, here, people, forage, farms_anything, rng):
        """Pioneers settle good river or coast land 50-300 km beyond the front, then the
        gaps fill in (the Linear Pottery farmers along central Europe's rivers, Bantu
        farmers along African rivers)."""
        cfg = state.config
        h, w = here.shape
        sources = np.flatnonzero(here & self.corridor & (neighbour_count(here) < 4))
        if not len(sources):
            return
        jumps = min(int(rng.poisson(cfg.pioneer_rate * self.scale * self.every * len(sources))), 20)
        if jumps == 0:
            return
        tamed = state.cells["tamed"]
        maturity = state.cells["maturity"][i]
        for src in rng.choice(sources, size=jumps):
            r, c = divmod(int(src), w)
            km = rng.uniform(*cfg.pioneer_km)
            angle = rng.uniform(0, 2 * np.pi)
            tr = int(round(r + km * np.sin(angle) / self.km_per_px))
            tc = int(round(c + km * np.cos(angle) / (self.km_per_px * self.cos_lat[r]))) % w
            if not 0 <= tr < h or here[tr, tc] or not people[tr, tc] or not self.corridor[tr, tc]:
                continue
            if self.landmass[tr, tc] != self.landmass[r, c] or self.suit[i][tr, tc] < 0.5:
                continue
            mat = maturity[r, c] * 0.95
            if farms_anything[tr, tc]:
                welcome = 1.0
            else:
                ratio = float(self._potential(cfg, i, mat)[tr, tc]) / max(float(forage[tr, tc]), 1e-6)
                welcome = float(np.clip((ratio - 1) / (cfg.farming_advantage - 1), cfg.frontier_floor, 1))
            if rng.random() < welcome * min(1.0, self.arable[tr, tc] / 0.6):   # good land only
                tamed[tr, tc] |= np.uint32(1 << i)
                maturity[tr, tc] = mat

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
        culture = state.cells.get("culture")
        cultures = state.tables.get("cultures")
        who = f"people near {where}"
        if culture is not None and cultures and 0 < culture[r, c] < len(cultures) and cultures[culture[r, c]]:
            who = f"the {cultures[culture[r, c]]['name']} (near {where})"
        state.event("domestication", f"{verb}: {who} tame the {sp.label}.",
                    species=sp.key, lat=lat, lon=lon, independent=independent)

    def _first_landmass_events(self, state, i, take):
        pass  # landmass-level spread events come with named regions in phase 2

    def _stats(self, state):
        tamed = state.cells["tamed"]
        records = state.tables["domestications"]
        land = ~state.world.ocean & ~state.world.lake
        crop_bits = sum(1 << i for i in range(self.n) if self.is_crop[i])
        stats = {
            "domestications": len(records),
            "farming_origins": sum(1 for r in records if r["independent"]),
            "species_tamed": len({r["species"] for r in records}),
            "farmed_land_share": float(((tamed & np.uint32(crop_bits)) != 0)[land].mean()),
            "herded_or_farmed_land_share": float((tamed != 0)[land].mean()),
            "settled_farmland_share": 0.0, "intensive_farmland_share": 0.0, "farm_density_low": 0.0,
            "farm_density_high": 0.0, "farm_density_top": 0.0,
        }
        mode = state.cells.get("subsistence")
        pop = state.cells.get("population")
        if mode is not None and pop is not None:
            farm = (mode == FARMING) & (pop > 0)
            if farm.sum() >= 10:
                dens = (pop / np.maximum(self.area, 1e-6))[farm]
                low, high, top = np.percentile(dens, [10, 90, 99])
                stats.update({
                    "settled_farmland_share": float((state.cells["intensity"][farm] >= 0.15).mean()),
                    "intensive_farmland_share": float((state.cells["intensity"][farm] >= 0.5).mean()),
                    "farm_density_low": float(low), "farm_density_high": float(high),
                    "farm_density_top": float(top),
                })
        state.stats.update(stats)


def _watered_suitability(world, is_crop, land):
    """Where each crop could grow if water were no limit (river-fed fields): rainfall and
    the wet season stop mattering; warmth, height and the rest still do."""
    from worldgen.nature import species_suitability
    out = np.zeros((len(is_crop),) + land.shape, np.float32)
    fields = ("ocean", "lake", "biome", "temperature_c", "elevation_m", "wetland")
    for rain in (800.0, 1500.0):            # inside the rain range of every crop
        proxy = SimpleNamespace(**{k: getattr(world, k) for k in fields})
        proxy.precip_mm = np.full(land.shape, rain, np.float32)
        proxy.growing_months = world.warm_months
        proxy.fresh_water = np.ones(land.shape, np.float32)      # wet-field crops too
        out = np.maximum(out, species_suitability(proxy))
    out[~is_crop] = 0
    out[:, ~land] = 0
    return out
