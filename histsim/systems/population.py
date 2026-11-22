"""People: where they live, how many the land can feed, growth, migration and famine.

Before year 0 a quick "peopling" pass spreads humans out of one homeland across the
ice-age world. After that, every step:

  capacity = how many people each cell can feed: foraging (from plant growth, fish and
             rivers), herding or farming (from the Farming system), in this year's climate
  growth   = slow growth toward capacity; quick die-off when people exceed it (famine)
  movement = people drift from crowded cells to roomier neighbours (and into empty land)
  coasts   = exposed ice-age plains drown as the sea rises; their people move inland
"""
import heapq

import numpy as np
from worldgen import BIOME_ID
from worldgen.generator import distance_to, shift
from worldgen.hydrology import cell_river_flow

from ..core.simulation import System
from .geo import DIRECTIONS4, ShelfSource, cell_area_km2, land_now, touches

FORAGING, HERDING, FARMING = 1, 2, 3     # values of the `subsistence` layer (0 = nobody)

# How much of the plant growth in each biome is food foragers can use (open land is best;
# dense forest hides most of its food in the canopy).
OPENNESS = {
    "Grassland": 1.0, "Savanna": 1.0, "Desert": 1.0, "Scrubland": 0.9, "Wetland": 0.9,
    "Tropical dry forest": 0.8, "Tundra": 0.8, "Temperate forest": 0.7, "Alpine": 0.6,
    "Boreal forest": 0.55, "Temperate rainforest": 0.5, "Tropical rainforest": 0.35,
    "Salt flat": 0.2, "Ice sheet": 0.0, "Glacier": 0.0,
}
# Years per km for Stone Age people to settle across each biome (inverse of spread speed).
TRAVEL = {
    "Grassland": 1.0, "Savanna": 1.0, "Scrubland": 1.1, "Tropical dry forest": 1.2,
    "Temperate forest": 1.4, "Wetland": 2.0, "Boreal forest": 2.0, "Temperate rainforest": 2.0,
    "Tropical rainforest": 2.5, "Tundra": 2.5, "Desert": 3.0, "Salt flat": 3.0, "Alpine": 5.0,
}


def _by_biome(biome, table, default):
    lookup = np.full(int(biome.max()) + 1, default, np.float32)
    for name, value in table.items():
        lookup[BIOME_ID[name]] = value
    return lookup[biome]


def plant_growth(temperature_c, precip_mm):
    """Net plant growth, g/m2/yr (the 'Miami model': limited by warmth or by rain)."""
    by_warmth = 3000 / (1 + np.exp(1.315 - 0.119 * temperature_c))
    by_rain = 3000 * (1 - np.exp(-0.000664 * np.maximum(precip_mm, 0)))
    return np.minimum(by_warmth, by_rain)


class Population(System):
    name = "population"

    def __init__(self):
        self.every = None   # set from the config in setup()

    # ----------------------------------------------------------------- setup
    def setup(self, state):
        cfg, world = state.config, state.world
        self.every = cfg.step_years
        self.area = cell_area_km2(world)
        self.shelf = ShelfSource(world)
        # Fixed per-cell facts, with exposed sea floor borrowing from the nearest land.
        self.openness = self.shelf.fill(_by_biome(world.biome, OPENNESS, 0.7), world)
        rivers = (cell_river_flow(world) > 200) | touches(world.lake)
        self.aquatic = 1 + cfg.river_bonus * rivers
        self.base_temp = self.shelf.fill(world.temperature_c.astype(np.float32), world)
        self.base_precip = self.shelf.fill(world.precip_mm.astype(np.float32), world)

        pop = state.add_layer("population", np.float32, record="snapshot")
        state.add_layer("subsistence", np.uint8, record="changes")
        state.add_layer("capacity", np.float32, record="none")
        state.add_layer("forager_capacity", np.float32, record="none")
        voyages = []
        arrival = people_the_world(world, cfg, state.rng.get("peopling"), voyages)
        state.tables["ancient_voyages"] = voyages   # prehistory: how far lands were first reached
        state.cells["arrival_year"] = arrival
        state.record_modes["arrival_year"] = "static"

        climate = _climate_now(state)
        land = land_now(world, climate.sea_level_m if climate else 0.0)
        forage = self.forager_capacity(state, land)
        settled_for = np.clip(state.year - arrival, 0, None)
        pop[:] = np.where(np.isfinite(arrival) & land, forage * (1 - np.exp(-settled_for / 3000)), 0)
        state.cells["subsistence"][:] = np.where(pop > 0, FORAGING, 0)
        self._last_drowning_event = -10 ** 9
        self._stats(state, famine=0.0, drowned=0.0)

    # ------------------------------------------------------------- capacity
    def forager_capacity(self, state, land):
        """People each cell can feed by foraging this year (whole cell, not per km2)."""
        cfg, world = state.config, state.world
        climate = state.climate
        temp = self.base_temp + (climate.temperature_offset_c if climate else 0)
        precip = self.base_precip * (climate.precip_factor if climate else 1)
        coast = land & touches(~land & ~world.lake)
        density = (cfg.forager_density_per_npp * plant_growth(temp, precip) * self.openness
                   * (self.aquatic + cfg.coast_bonus * coast))
        density = np.where(temp < -12, 0, density)        # under ice
        cap = np.where(land, density * self.area, 0).astype(np.float32)
        state.cells["forager_capacity"] = cap
        return cap

    # ----------------------------------------------------------------- step
    def step(self, state):
        cfg, world = state.config, state.world
        dt = self.every
        climate = state.climate
        land = land_now(world, climate.sea_level_m if climate else 0.0)
        forage = self.forager_capacity(state, land)
        farming = state.tables.get("farming")
        if farming is not None:
            cap, mode = farming.capacity(state, forage, land)
        else:
            cap, mode = forage, np.where(forage > 0, FORAGING, 0)
        town = state.cells.get("settlement")
        if town is not None:   # townspeople are fed by their hinterland
            cap = np.where(cap > 0, cap + town, 0).astype(np.float32)
        state.cells["capacity"] = cap
        pop = state.cells["population"]

        # The spread that began before year 0 carries on: when a cell's arrival year comes
        # round, a small founding group appears (late-settled lands, ancient voyagers).
        arrival = state.cells["arrival_year"]
        arriving = (arrival <= state.year) & (pop <= 0) & (forage > 0)
        if arriving.any():
            pop[arriving] = forage[arriving] * 0.1

        # Drowned or frozen cells: most people move to a habitable neighbour.
        lost = (cap <= 0) & (pop > 0)
        drowned = 0.0
        if lost.any():
            moving = np.where(lost, pop * 0.6, 0)
            drowned = float(pop[lost].sum() - moving.sum())
            ok = cap > 0
            n_ok = sum(shift(ok.astype(np.int8), dy, dx) for dy, dx in DIRECTIONS4)
            share = np.where(lost & (n_ok > 0), moving / np.maximum(n_ok, 1), 0)
            for dy, dx in DIRECTIONS4:   # each neighbour receives its share from lost cells
                pop += np.where(ok, shift(share, -dy, -dx), 0)
            pop[lost] = 0
            if drowned > 1000 and climate and climate.sea_level_m < 0 and state.year - self._last_drowning_event > 1000:
                state.event("environment", "Rising seas drown coastal plains; their people flee inland.",
                            people=round(drowned))
                self._last_drowning_event = state.year

        # Growth toward capacity, and famine where people exceed it.
        growth = np.where(mode >= HERDING, cfg.farmer_growth, cfg.forager_growth)
        room = np.where(cap > 0, 1 - pop / np.maximum(cap, 1e-6), 0)
        over = pop > cap
        before = pop.copy()
        pop[:] = np.where(over, cap + (pop - cap) * 0.5 ** (dt / cfg.famine_halving_years),
                          pop * np.exp(growth * dt * np.clip(room, 0, 1)))
        famine = float((before - pop)[over].sum())

        # Movement: people drift from crowded cells toward roomier neighbours.
        # Each flow is worked out once per pair of neighbours (east and south) and applied
        # to both, so people are never created or lost by moving.
        pressure = np.where(cap > 0, pop / np.maximum(cap, 1e-6), 0)
        rate = min(0.24, cfg.migration_rate * dt)
        change = np.zeros_like(pop)
        for dy, dx in ((0, 1), (1, 0)):
            nb_pop = shift(pop, dy, dx)
            flow = rate * (pressure - shift(pressure, dy, dx)) * np.minimum(cap, shift(cap, dy, dx)) / 4
            flow = np.clip(flow, -nb_pop / 4, pop / 4)
            if dy == 1:
                flow[-1] = 0          # the bottom row has no neighbour below
            change -= flow
            change += shift(flow, -dy, -dx) if dx else np.vstack([np.zeros_like(flow[:1]), flow[:-1]])
        pop += change
        np.maximum(pop, 0, out=pop)
        pop[pop < 1] = 0   # nobody left
        state.cells["subsistence"] = np.where(pop > 0, mode, 0).astype(np.uint8)
        self._stats(state, famine=famine, drowned=drowned)

    def _stats(self, state, famine, drowned):
        pop = state.cells["population"]
        mode = state.cells["subsistence"]
        land = ~state.world.ocean & ~state.world.lake
        state.stats.update({
            "world_population": float(pop.sum()),
            "foragers": float(pop[mode == FORAGING].sum()),
            "herders": float(pop[mode == HERDING].sum()),
            "farmers": float(pop[mode == FARMING].sum()),
            "inhabited_land_share": float(((pop > 0) & land).sum() / max(land.sum(), 1)),
            "famine_deaths_per_year": famine / self.every,
            "drowned_per_year": drowned / self.every,
        })


def _climate_now(state):
    climate = state.climate
    if climate is None:
        from worldgen import ClimateHistory
        climate = ClimateHistory(state.world).at(state.year)
        state.climate = climate
    return climate


# ---------------------------------------------------------------- peopling
def people_the_world(world, cfg, rng, voyages=None):
    """When people first arrived in each cell (simulation year; inf = never, by year 0).

    People leave one warm, open, well-watered homeland on a large landmass and spread
    across the ice-age world: land is wider (lower sea), ice sheets block the cold
    north and south, and boats manage open water up to `max_sea_crossing_km` (long
    crossings much later than short ones). Arrival times are scaled so the spread starts
    at `out_of_homeland_year` and has reached nearly everywhere by `settled_by_year`.

    Then rare ancient voyages: any large landmass (at least `voyage_min_area` of the
    globe) still empty but within `ancient_voyage_km` of settled land gets founders,
    landing on the coast nearest their homeland, later the longer the crossing; newly
    settled landmasses can launch further voyages. Truly remote lands and small islands
    stay empty for later seafarers. If `voyages` is a list, each voyage is appended to it.
    """
    from worldgen.generator import label_wrapped

    h, w = world.ocean.shape
    kpp = world.grid.km_per_px
    land_then = land_now(world, cfg.glacial_sea_level_m)
    shelf = ShelfSource(world)
    ice = land_then & (shelf.fill(world.temperature_c, world) < -9)
    walkable = land_then & ~ice

    travel = shelf.fill(_by_biome(world.biome, TRAVEL, 1.5), world)
    slope = shelf.fill(world.slope_m_per_km.astype(np.float32), world)
    travel = travel * (1 + slope / 20)
    to_land_km = distance_to(walkable)[0] * kpp
    crossing = ~walkable & ~ice & (to_land_km <= cfg.max_sea_crossing_km / 2)
    # Open water is slow to cross, and much slower far from any shore: short hops are
    # managed early, long ones only late in the 60,000 years (island-hopping).
    sea_cost = 6.0 * (1 + (to_land_km / cfg.sea_crossing_difficulty_km) ** 2)
    cost = np.where(walkable, travel, np.where(crossing, sea_cost, np.inf)).astype(np.float64)

    # The homeland: warm, open, watered land on one of the largest landmasses.
    lm, area = world.landmass_id, world.landmass_area
    big = np.isin(lm, np.argsort(-area)[:3]) & (area[lm] > 0.005)
    open_land = np.isin(world.biome, [BIOME_ID[b] for b in ("Savanna", "Grassland", "Tropical dry forest")])
    score = (big & open_land & (world.temperature_c > 18) & (world.temperature_c < 28)) * (0.3 + world.fresh_water)
    if not score.any():
        score = (walkable * 1.0)
    flat = score.ravel() / score.sum()
    origin = int(rng.choice(flat.size, p=flat))

    # Travel time over a coarser grid (fast), then back to full resolution.
    f = max(1, w // 400)
    ch, cw = h // f, w // f
    blocks = cost[:ch * f, :cw * f].reshape(ch, f, cw, f)
    coarse = np.where(np.isfinite(blocks), blocks, np.inf).min(axis=(1, 3))   # min keeps narrow passages open
    km_ns = kpp * f
    km_ew = km_ns * np.maximum(np.cos(np.radians(world.grid.lat[:ch * f:f, 0])), 0.05)

    def travel_from(cell):
        time = _dijkstra(coarse, (cell // w) // f, (cell % w) // f, km_ns, km_ew)
        full = np.repeat(np.repeat(time, f, axis=0), f, axis=1)
        t = np.full((h, w), np.inf)
        t[:full.shape[0], :full.shape[1]] = full
        return np.where(walkable, t, np.inf)

    t = travel_from(origin)
    reached = np.isfinite(t)
    if not reached.any():
        return t.astype(np.float32)
    scale = (cfg.settled_by_year - cfg.out_of_homeland_year) / max(np.percentile(t[reached], 97), 1e-6)
    arrival = cfg.out_of_homeland_year + t * scale

    # Rare ancient voyages to large empty landmasses within reach.
    lands, n = label_wrapped(walkable)
    cos_lat = np.cos(np.radians(world.grid.lat))
    land_area = np.bincount(lands.ravel(), weights=cos_lat.ravel(), minlength=n + 1) / cos_lat.sum()
    span = cfg.settled_by_year - cfg.out_of_homeland_year
    for _ in range(cfg.max_voyages):                     # chains of voyages, nearest first
        settled = np.isfinite(arrival)
        dist_km, iy, ix = distance_to(settled & walkable)
        dist_km = dist_km * kpp
        best = None
        for land_id in range(1, n + 1):
            if land_area[land_id] < cfg.voyage_min_area:
                continue
            cells = lands == land_id
            if settled[cells].any():
                continue
            d = np.where(cells, dist_km, np.inf)
            landing = int(np.argmin(d))
            gap = float(d.ravel()[landing])
            if gap <= cfg.ancient_voyage_km and (best is None or gap < best[1]):
                best = (land_id, gap, landing)
        if best is None:
            break
        land_id, gap, landing = best
        r, c = divmod(landing, w)
        from_cell = (iy[r, c], ix[r, c])
        departed = float(arrival[from_cell])
        # Longer voyages happen later: somewhere between the departure point's own
        # settlement and the end of the spread.
        year = departed + (cfg.settled_by_year - departed) * (0.3 + 0.6 * gap / cfg.ancient_voyage_km)
        local = travel_from(landing)
        cells = lands == land_id
        arrival = np.where(cells & np.isfinite(local), year + local * scale * 0.5, arrival)
        if voyages is not None:
            voyages.append({"year": round(year), "gap_km": round(gap),
                            "landing_lat": float(world.grid.lat[r, c]), "landing_lon": float(world.grid.lon[r, c]),
                            "from_lat": float(world.grid.lat[from_cell]), "from_lon": float(world.grid.lon[from_cell]),
                            "land_share": float(land_area[land_id])})
    return arrival.astype(np.float32)


def _dijkstra(cost, start_r, start_c, km_ns, km_ew):
    """Least travel time from one cell over a grid that wraps east-west (8 neighbours)."""
    h, w = cost.shape
    best = np.full((h, w), np.inf)
    best[start_r, start_c] = 0.0
    heap = [(0.0, start_r, start_c)]
    c = cost
    while heap:
        t, r, col = heapq.heappop(heap)
        if t > best[r, col]:
            continue
        here = c[r, col]
        for dr in (-1, 0, 1):
            rr = r + dr
            if rr < 0 or rr >= h:
                continue
            for dc in (-1, 0, 1):
                if not (dr or dc):
                    continue
                cc = (col + dc) % w
                there = c[rr, cc]
                if there == np.inf:
                    continue
                step = (km_ns if dc == 0 else (km_ew[r] if dr == 0 else (km_ns ** 2 + km_ew[r] ** 2) ** 0.5))
                nt = t + step * 0.5 * (here + there)
                if nt < best[rr, cc]:
                    best[rr, cc] = nt
                    heapq.heappush(heap, (nt, rr, cc))
    return best
