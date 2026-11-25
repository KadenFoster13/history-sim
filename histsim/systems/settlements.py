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

No place grows beyond what can be brought to it to eat (`feed_limit`): a few tens of
thousands where food comes on foot and by donkey, more on a navigable river or coast, and
more again with canals, sailing, roads, writing, officials and coinage. So the first
cities stay at tens of thousands and the great capitals come with the states able to feed
them (Uruk ~40-80,000; Rome, fed by sea, ~1 million).

Around each central place the rest of a region's townspeople live in a ladder of smaller
towns (rank-size: about ten times as many places above each tenfold-smaller size). Those
are not named one by one: the `small_towns` layer counts them per cell.

A place of `town_min` people or more is a named town (in the language of the culture
living there); 10,000 or more is a city. The `settlement` layer is the size of the
largest place in each cell, and `settlement_id` which named place it is.
"""
import numpy as np
from numba import njit, prange
from worldgen import BIOME_ID
from worldgen.generator import smoothstep

from .geo import shift
from worldgen.hydrology import cell_river_flow
from worldgen.noise import Perlin3, fbm

from ..core.simulation import System
from .geo import cell_area_km2, cell_scale, touches
from .population import FARMING, HERDING

CITY = 10_000


class Settlements(System):
    name = "settlements"

    def __init__(self):
        self.every = None

    def setup(self, state):
        cfg, world = state.config, state.world
        self.every = cfg.step_years
        self.scale = cell_scale(world, cfg)
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

        self.water = (river | coast | lake).astype(np.float32)          # food can come by boat
        state.add_layer("settlement", np.float32, record="snapshot")
        state.add_layer("small_towns", np.float32, record="snapshot")     # towns below the main place
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
        # Farming's age in each cell (time for crafts, trade, temples and chiefs), and the
        # farmers (and some herders) who feed townspeople. Compiled, cell by cell.
        complexity = np.empty(pop.shape)
        farm_pop = np.empty(pop.shape)
        _town_inputs(since.ravel(), mode.ravel(), pop.ravel(), state.year, FARMING, HERDING,
                     cfg.town_delay_years, cfg.town_ramp_years, complexity.ravel(), farm_pop.ravel())
        r = max(1, round(cfg.hinterland_km / state.world.grid.km_per_px))
        hinter = _box(farm_pop, r, np.add)
        if getattr(self, "_hinter_area_r", None) != r:          # land around each cell: fixed
            self._hinter_area = _box(self.area * self.land, r, np.add)
            self._hinter_area_r = r
        size = state.cells["settlement"]
        trade = state.tables.get("trade")
        traded = state.cells["trade"] if trade is not None else np.zeros(1, np.float32)
        # Each place's urban share, site (crossroads and ports count more), and its score
        # as a central place.
        share = np.empty(pop.shape)
        site = np.empty(pop.shape)
        score = np.empty(pop.shape)
        _town_scores(hinter.ravel(), self._hinter_area.ravel(), complexity.ravel(), self.site.ravel(),
                     traded.ravel(), trade is not None, cfg.trade_half, cfg.trade_site, size.ravel(),
                     cfg.urban_share_max, cfg.urban_density_ref, cfg.incumbency,
                     share.ravel(), site.ravel(), score.ravel())
        centre = (score > 0) & (score >= _box(score, r, np.maximum))
        quality = np.clip(site / self.site_mean, 0.5, 2.5)
        target = np.where(centre, cfg.central_place_share * share * hinter * quality, 0)
        # A town can't outgrow the people around it for long: its own townspeople plus the
        # farmers of an 800-px cell's ~2,500 km2 around it, whatever the map's cell size
        # (on a fine map that area spans several cells; on a coarse one it is part of one).
        people = np.maximum(pop, 0)
        if self.scale == 1:
            nearby = people
        else:
            town_people = np.minimum(size, people)
            k = int(0.5 / self.scale + 1e-9)
            if k == 0:
                nearby = town_people + (people - town_people) / self.scale ** 2
            else:
                around = _box(people, k, np.add) - town_people
                nearby = town_people + around * (1 / self.scale ** 2 - 1) / ((2 * k + 1) ** 2 - 1)
        target = np.minimum(target, 0.9 * nearby + target * 0.1)
        limit = self.feed_limit(state)
        self._capitals(state, target, dt, limit)
        target = np.minimum(target, limit)
        self._small_towns(state, share, farm_pop, limit)
        size += (target - size) * (1 - np.exp(-dt / cfg.settlement_years))
        size[size < 50] = 0
        self._places(state)
        self._stats(state)

    def feed_limit(self, state):
        """The most people a place can be fed (per cell): by what its people know of moving
        food and governing (canals, sailing, roads, writing, officials, coinage), and more
        on a river or coast."""
        from ..tech import INDEX
        cfg = state.config
        tech = state.cells.get("tech")
        keys = ("canals", "sailing", "roads", "writing", "bureaucracy", "coinage")
        bits = np.array([INDEX[k] for k in keys], np.int64)
        extra = np.array([cfg.feed_canals, cfg.feed_sailing, cfg.feed_roads, cfg.feed_writing,
                          cfg.feed_bureaucracy, cfg.feed_coinage])
        trade = state.tables.get("trade")
        traded = state.cells["trade"].ravel() if trade is not None else np.zeros(1, np.float32)
        limit = np.empty(self.water.size, np.float32)
        _feed_kernel(self.water.ravel(), tech.ravel() if tech is not None else np.zeros(1, np.uint64),
                     tech is not None, bits, extra, traded, trade is not None, cfg.trade_half, cfg.feed_base,
                     cfg.feed_water, cfg.feed_trade, limit)
        return limit.reshape(self.water.shape)

    def _small_towns(self, state, share, farm_pop, limit):
        """The towns below each region's main place: a rank-size ladder (Pareto, exponent 1)
        from `town_min` up to half what a place can be fed, holding the region's townspeople
        not in its main place. Counted per cell."""
        cfg = state.config
        hi = np.empty(limit.shape)
        people = np.empty(limit.size)
        _small_town_kernel(
            np.ascontiguousarray(share, np.float64).ravel(), np.ascontiguousarray(farm_pop, np.float64).ravel(),
            limit.ravel(), 1 - cfg.central_place_share, cfg.town_min, state.cells["small_towns"].ravel(), hi.ravel(),
            people)
        self._small_people = float(people.sum())
        self._small_hi = hi

    def _capitals(self, state, target, dt, limit=None):
        """Capitals draw people, trade and tribute from their whole state: they grow far
        beyond what their own fields could support, fed by the rest of the state, and people
        move to them from across it."""
        polity = state.tables.get("polity")
        if polity is None or not hasattr(polity, "people"):
            return
        cfg = state.config
        pop = state.cells["population"]
        own = state.cells["state"]
        records = state.tables["states"]
        flat_pop = pop.ravel()
        # Each state's cells, in map order, found with one sort instead of a scan per state.
        flat_own = own.ravel()
        order = np.argsort(flat_own, kind="stable")
        bounds = np.searchsorted(flat_own[order], np.arange(len(records) + 1))
        for sid in polity.alive_ids():
            if sid >= len(polity.people) or polity.people[sid] < 50_000:
                continue
            cap = records[sid]["capital"]
            share = cfg.capital_share * (0.5 if records[sid]["kind"] == "chiefdom" else 1.0)   # chiefs draw less
            extra = share * polity.people[sid] * np.sqrt(polity.cohesion[sid])
            target.flat[cap] += extra
            if limit is not None:
                target.flat[cap] = min(target.flat[cap], limit.flat[cap])
            # Migration toward the capital, from across the state (people are conserved).
            want = target.flat[cap] / 0.9 - flat_pop[cap]
            if want <= 0:
                continue
            others = order[bounds[sid]:bounds[sid + 1]]
            others = others[others != cap]
            pool = flat_pop[others].sum()
            if pool <= 0:
                continue
            move = min(want * (1 - np.exp(-dt / cfg.settlement_years)), 0.02 * pool)
            flat_pop[others] *= 1 - move / pool
            flat_pop[cap] += move

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
        small = state.cells["small_towns"]
        lo, hi = cfg.town_min, getattr(self, "_small_hi", 2 * cfg.town_min)
        # Smaller towns of 10,000+: the Pareto tail above 10,000 where the ladder reaches it.
        tail = np.where(hi > CITY, (1 / CITY - 1 / np.maximum(hi, CITY)) / (1 / lo - 1 / hi), 0)
        state.stats.update({
            "towns": int(towns.sum() + round(float(small.sum()))),
            "named_towns": int(towns.sum()),
            "cities": int((size >= CITY).sum() + round(float((small * tail).sum()))),
            "cities_100k": int((size >= 100_000).sum()),
            "largest_settlement": float(size.max()) if size.size else 0.0,
            "urban_share": float(size[towns].sum() + getattr(self, "_small_people", 0.0)) / total,
        })


@njit(cache=True, parallel=True)
def _town_inputs(since, mode, pop, year, farming, herding, delay, ramp, complexity, farm_pop):
    never = np.int32(-(2 ** 31))
    for i in prange(since.shape[0]):
        m = mode[i]
        if since[i] == never and m == farming:
            since[i] = year
        age = year - since[i] if since[i] != never else 0
        t = min(max((age - delay) / ramp, 0.0), 1.0)
        complexity[i] = t * t * (3 - 2 * t)
        farm_pop[i] = pop[i] if m == farming else (0.3 * pop[i] if m == herding else 0.0)


@njit(cache=True, parallel=True)
def _town_scores(hinter, hinter_area, complexity, base_site, traded, has_trade, trade_half, trade_site, size,
                 urban_share_max, urban_density_ref, incumbency, share, site, score):
    for i in prange(hinter.shape[0]):
        density = hinter[i] / max(hinter_area[i], 1e-6)
        sh = urban_share_max * complexity[i] * min(max(density / urban_density_ref, 0.0), 1.0)
        st = base_site[i]
        if has_trade:
            t = traded[i]
            st = st * (1 + trade_site * t / (t + trade_half))
        share[i] = sh
        site[i] = st
        score[i] = sh * hinter[i] * st * (1 + incumbency * (size[i] > 0))


@njit(cache=True, parallel=True)
def _feed_kernel(water, tech, has_tech, bits, extra, traded, has_trade, trade_half, base, feed_water,
                 feed_trade, out):
    one = np.uint64(1)
    for i in prange(water.shape[0]):
        v = base * (1 + feed_water * water[i])
        if has_tech:
            t = tech[i]
            for k in range(bits.shape[0]):
                if (t >> np.uint64(bits[k])) & one:
                    v *= 1 + extra[k]
        if has_trade:
            tr = traded[i]
            v *= 1 + feed_trade * tr / (tr + trade_half)
        out[i] = v


@njit(cache=True, parallel=True)
def _small_town_kernel(share, farm_pop, limit, outside, lo, count_out, hi_out, people_out):
    """Townspeople outside each area's main place, in a rank-size (Pareto 1) ladder of towns
    from `lo` to half what a place can be fed: how many towns per cell (and their people)."""
    for i in prange(share.shape[0]):
        hi = max(0.5 * limit[i], 2 * lo)
        hi_out[i] = hi
        mean = lo * hi * np.log(hi / lo) / (hi - lo)
        people = outside * share[i] * farm_pop[i]
        c = people / mean
        count_out[i] = c if c >= 0.05 else 0.0
        people_out[i] = c * mean


def _box(a, r, op):
    """Combine each cell with its (2r+1) x (2r+1) neighbourhood (np.add sums, np.maximum
    takes the largest); wraps east-west, nothing beyond the poles. (Compiled.)"""
    a = np.ascontiguousarray(a)
    if r <= 0:
        return a.copy()
    if op is np.maximum:
        return _box_max(a, r)
    return _box_sum(a.astype(np.float64), r).astype(a.dtype if a.dtype.kind == "f" else np.float64)


@njit(cache=True)
def _box_sum(a, r):
    h, w = a.shape
    rows = np.zeros((h, w))
    for y in range(h):
        s = 0.0
        for k in range(-r, r + 1):
            s += a[y, k % w]
        for x in range(w):
            rows[y, x] = s
            add = x + r + 1
            if add >= w:
                add -= w
            sub = x - r
            if sub < 0:
                sub += w
            s += a[y, add] - a[y, sub]
    out = np.zeros((h, w))
    acc = np.zeros(w)
    for k in range(0, min(h, r + 1)):          # window for row 0: rows 0..r
        for x in range(w):
            acc[x] += rows[k, x]
    for y in range(h):
        for x in range(w):
            out[y, x] = acc[x]
        if y + r + 1 < h:
            for x in range(w):
                acc[x] += rows[y + r + 1, x]
        if y - r >= 0:
            for x in range(w):
                acc[x] -= rows[y - r, x]
    return out


@njit(cache=True)
def _box_max(a, r):
    h, w = a.shape
    pad = np.empty(w + 2 * r, a.dtype)
    rows = np.empty_like(a)
    for y in range(h):
        for x in range(w + 2 * r):
            xx = x - r
            if xx < 0:
                xx += w
            elif xx >= w:
                xx -= w
            pad[x] = a[y, xx]
        for x in range(w):
            m = pad[x]
            for k in range(1, 2 * r + 1):
                if pad[x + k] > m:
                    m = pad[x + k]
            rows[y, x] = m
    out = np.empty_like(a)
    for y in range(h):
        lo, hi = max(0, y - r), min(h, y + r + 1)
        for x in range(w):
            out[y, x] = rows[lo, x]
        for k in range(lo + 1, hi):
            for x in range(w):
                if rows[k, x] > out[y, x]:
                    out[y, x] = rows[k, x]
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
