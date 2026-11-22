"""Settlements: villages become towns, and towns cities.

Farmers live in villages (part of each cell's population). Towns and cities grow where
three things meet:

  time      farming has been around for thousands of years: crafts, trade, temples and
            chiefs take time to appear (`town_delay_years`, `town_ramp_years`)
  people    a dense farming hinterland (the cells around) to feed townsfolk
  site      a good place: on a big river, a coast, a lake, a river mouth; plus luck

In each neighbourhood, the best site becomes the central place and draws a share of its
hinterland's people; towns grow and decline gradually, and an established town keeps an
advantage over newcomers (incumbency). Townspeople are fed by the hinterland, so a town
adds to what its cell can feed.

A place of `town_min` people or more is a named town (in the language of the culture
living there); 10,000 or more is a city. The `settlement` layer is the size of the
largest place in each cell, and `settlement_id` which named place it is.
"""
import numpy as np
from worldgen import BIOME_ID
from worldgen.generator import shift, smoothstep
from worldgen.hydrology import cell_river_flow
from worldgen.noise import Perlin3, fbm

from ..core.simulation import System
from .geo import cell_area_km2, touches
from .population import FARMING, HERDING

CITY = 10_000


class Settlements(System):
    name = "settlements"

    def __init__(self):
        self.every = None

    def setup(self, state):
        cfg, world = state.config, state.world
        self.every = cfg.step_years
        self.area = cell_area_km2(world)
        land = ~world.ocean & ~world.lake
        self.land = land
        flow = cell_river_flow(world)
        river = land & (flow > 300)
        big_river = land & (flow > 3000)
        coast = land & touches(world.ocean)
        lake = land & touches(world.lake)
        noise = np.exp(0.35 * fbm(Perlin3(world.cfg.seed * 7919 + state.seed * 17 + 9090), world.grid.xyz, 20, 2))
        self.site = ((1 + cfg.river_site * river + cfg.river_site * big_river + cfg.coast_site * coast
                      + 0.3 * lake + 0.5 * (river & coast)) * noise * land).astype(np.float32)
        self.site_mean = float(self.site[land].mean()) if land.any() else 1.0
        # What a place's name may be about.
        hills = land & (world.slope_m_per_km > 25)
        biome = world.biome
        self.features = {
            "river": river, "ford": river & ~big_river, "sea": coast, "bay": coast & ~hills,
            "lake": lake, "hill": hills, "stone": hills | (world.elevation_m > 1200),
            "forest": np.isin(biome, [BIOME_ID[b] for b in ("Temperate forest", "Boreal forest",
                                                             "Tropical rainforest", "Temperate rainforest")]),
            "marsh": biome == BIOME_ID["Wetland"],
            "desert": np.isin(biome, [BIOME_ID[b] for b in ("Desert", "Salt flat")]),
            "plain": np.isin(biome, [BIOME_ID[b] for b in ("Grassland", "Savanna")]),
        }

        state.add_layer("settlement", np.float32, record="snapshot")
        state.add_layer("settlement_id", np.int32, record="changes")
        state.add_layer("farming_since", np.int32, fill=-(2 ** 31), record="changes")
        state.tables["settlements"] = [None]      # id 0 = none
        state.saved_tables.add("settlements")
        self._by_cell = {}                         # cell -> settlement id (kept after abandonment)
        self._first_town = self._first_city = False
        self._largest = None
        self._largest_event_year = -10 ** 9
        self._stats(state)

    # ----------------------------------------------------------------- step
    def step(self, state):
        cfg = state.config
        dt = self.every
        pop = state.cells["population"]
        mode = state.cells["subsistence"]
        since = state.cells["farming_since"]
        never = since == -(2 ** 31)
        since[never & (mode == FARMING)] = state.year
        age = np.where(since > -(2 ** 31), state.year - since, 0)
        complexity = smoothstep(cfg.town_delay_years, cfg.town_delay_years + cfg.town_ramp_years, age)

        farm_pop = np.where(mode == FARMING, pop, 0) + 0.3 * np.where(mode == HERDING, pop, 0)
        r = max(1, round(cfg.hinterland_km / state.world.grid.km_per_px))
        hinter = _box(farm_pop, r, np.add)
        hinter_area = _box(self.area * self.land, r, np.add)
        density = hinter / np.maximum(hinter_area, 1e-6)
        share = cfg.urban_share_max * complexity * np.clip(density / cfg.urban_density_ref, 0, 1)
        size = state.cells["settlement"]
        score = share * hinter * self.site * (1 + cfg.incumbency * (size > 0))
        centre = (score > 0) & (score >= _box(score, r, np.maximum))
        quality = np.clip(self.site / self.site_mean, 0.5, 2.5)
        target = np.where(centre, cfg.central_place_share * share * hinter * quality, 0)
        target = np.minimum(target, 0.9 * np.maximum(pop, 0) + target * 0.1)   # can't outgrow its people for long
        size += (target - size) * (1 - np.exp(-dt / cfg.settlement_years))
        size[size < 50] = 0
        self._places(state)
        self._stats(state)

    # ---------------------------------------------------------- named places
    def _places(self, state):
        cfg = state.config
        size = state.cells["settlement"]
        ids = state.cells["settlement_id"]
        records = state.tables["settlements"]
        w = size.shape[1]
        world = state.world

        for cell in np.flatnonzero((size >= cfg.town_min) & (ids == 0)):
            r, c = divmod(int(cell), w)
            sid = self._by_cell.get(int(cell))
            if sid is None:
                sid = self._found(state, r, c)
            else:
                records[sid]["ended"] = None          # a deserted town lives again
            ids[r, c] = sid
        gone = (ids > 0) & (size < cfg.town_min * 0.4)
        for cell in np.flatnonzero(gone):
            r, c = divmod(int(cell), w)
            rec = records[ids[r, c]]
            rec["ended"] = state.year
            if rec["peak"] >= CITY:
                state.event("settlement", f"{rec['name']}, once a city of {_people(rec['peak'])}, is deserted.",
                            settlement=rec["id"], lat=rec["lat"], lon=rec["lon"])
            ids[r, c] = 0

        alive = np.flatnonzero(ids > 0)
        for cell in alive:
            rec = records[ids.flat[cell]]
            s = float(size.flat[cell])
            if s > rec["peak"]:
                rec["peak"], rec["peak_year"] = round(s), state.year

        if alive.size:
            top = alive[np.argmax(size.flat[alive])]
            biggest = records[ids.flat[top]]
            s = float(size.flat[top])
            if s >= CITY and not self._first_city:
                self._first_city = True
                state.event("settlement", f"{biggest['name']}{self._of(state, biggest)} becomes the world's first "
                                          f"city, home to {_people(s)}.", settlement=biggest["id"],
                            lat=biggest["lat"], lon=biggest["lon"])
                self._largest, self._largest_event_year = biggest["id"], state.year
            elif (s >= CITY and biggest["id"] != self._largest
                  and state.year - self._largest_event_year >= cfg.largest_city_event_years):
                self._largest, self._largest_event_year = biggest["id"], state.year
                state.event("settlement", f"{biggest['name']}{self._of(state, biggest)} is now the largest city "
                                          f"in the world, with {_people(s)}.", settlement=biggest["id"],
                            lat=biggest["lat"], lon=biggest["lon"])

    def _found(self, state, r, c):
        world = state.world
        records = state.tables["settlements"]
        culture = int(state.cells["culture"][r, c]) if "culture" in state.cells else 0
        namer = state.tables.get("naming")
        lang = namer.language_of(state, culture) if namer is not None and culture else None
        features = [k for k, mask in self.features.items() if mask[r, c]] or ["field"]
        rng = state.rng.get("names")
        name = None
        if lang is not None:
            for _ in range(20):
                name = lang.place_name(rng, features)
                if name not in namer.used_names:
                    break
            namer.used_names.add(name)
        sid = len(records)
        name = name or f"Town {sid}"
        rec = {"id": sid, "name": name, "row": r, "col": c,
               "lat": round(float(world.grid.lat[r, c]), 2), "lon": round(float(world.grid.lon[r, c]), 2),
               "culture": culture, "founded": state.year, "ended": None,
               "peak": round(float(state.cells["settlement"][r, c])), "peak_year": state.year,
               "features": features}
        records.append(rec)
        self._by_cell[r * state.shape[1] + c] = sid
        if not self._first_town:
            self._first_town = True
            state.event("settlement", f"The first town in the world grows up at {name}{self._among(state, rec)}.",
                        settlement=sid, lat=rec["lat"], lon=rec["lon"])
        return sid

    @staticmethod
    def _people_of(state, rec):
        cultures = state.tables.get("cultures")
        if cultures and rec["culture"] and cultures[rec["culture"]]:
            return cultures[rec["culture"]]["name"]
        return None

    def _of(self, state, rec):
        name = self._people_of(state, rec)
        return f" of the {name}" if name else ""

    def _among(self, state, rec):
        name = self._people_of(state, rec)
        return f", among the {name}" if name else ""

    # ----------------------------------------------------------------- stats
    def _stats(self, state):
        cfg = state.config
        size = state.cells["settlement"]
        total = max(float(state.cells["population"].sum()), 1e-9)
        towns = size >= cfg.town_min
        state.stats.update({
            "towns": int(towns.sum()),
            "cities": int((size >= CITY).sum()),
            "largest_settlement": float(size.max()) if size.size else 0.0,
            "urban_share": float(size[towns].sum() / total),
        })


def _box(a, r, op):
    """Combine each cell with its (2r+1) x (2r+1) neighbourhood (np.add sums, np.maximum
    takes the largest); wraps east-west."""
    rows = a.copy()
    for dx in range(1, r + 1):
        rows = op(op(rows, shift(a, 0, dx)), shift(a, 0, -dx))
    out = rows.copy()
    for dy in range(1, r + 1):
        out = op(op(out, _shift_rows(rows, dy)), _shift_rows(rows, -dy))
    return out


def _shift_rows(a, dy):
    """out[y] = a[y + dy], with zeros beyond the poles."""
    out = np.zeros_like(a)
    if dy > 0:
        out[:-dy] = a[dy:]
    else:
        out[-dy:] = a[:dy]
    return out


def _people(n):
    n = float(n)
    if n >= 1e6:
        return f"about {n / 1e6:.1f} million people"
    return f"about {round(n, -3 if n >= 10_000 else -2):,.0f} people"
