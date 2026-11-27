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
from numba import njit, prange
from worldgen import SPECIES
from worldgen.generator import distance_to

from ..core.calendar import label
from ..core.rng import draws_at, hits
from ..core.simulation import System
from .geo import DIRECTIONS4, band, cell_area_km2, cell_scale, neighbour_count, neighbour_index, rounds_for
from .population import FARMING, FORAGING, HERDING, OPENNESS, _by_biome

KEYS = [sp.key for sp in SPECIES]
# How open a cell's land lies to mounted raiders: the grass steppe most (the horse nomads'
# own country); scrub, desert edge and savanna less (raided, but farmed: the Levant, the
# Sahel); forest, marsh and hills give cover.
RAID_OPEN = {"Grassland": 1.0, "Scrubland": 0.4, "Desert": 0.4, "Savanna": 0.3, "Tundra": 0.3,
             "Salt flat": 0.4}

TSETSE_SENSITIVE = {"cattle", "horse", "camel"}
INDEPENDENT_KM = 1000      # a domestication this far from any farming counts as a new origin
# Crops that keep and can be counted and taxed: the grains, pulses and highland tubers (freeze-
# dried, as chuno in the Andes). Not tropical roots and tree crops (see Farming.storable_share).
STORABLE = {"temperate_grain", "cold_grain", "wet_grain", "tropical_grain", "slow_grain", "legume",
            "highland_tuber"}


@njit(cache=True)
def _ceiling_cell(i, tech, has_tech, bits, tamed, wet, suit_wet, irrigable, dryness, flood, flood_dry, slope,
                  stream, arable_eff, consts):
    """The intensification ceiling of one cell (see Farming.ceiling)."""
    steep_slope, rainfed, plough_c, manure_c, irrigated, paddy_c = (consts[0], consts[1], consts[2], consts[3],
                                                                    consts[4], consts[5])
    t = tech[i] if has_tech else np.uint64(0)
    one = np.uint64(1)
    canals = (t >> np.uint64(bits[0])) & one
    terraces = (t >> np.uint64(bits[1])) & one
    plough = (t >> np.uint64(bits[2])) & one
    manure = (t >> np.uint64(bits[3])) & one
    # A flood waters fields where rain is short (the Nile, the Euphrates); where rain is
    # plenty (the Amazon, the Congo) it adds silt but no water worth having, unless the
    # crop is paddy rice, which is water control in any climate. Paddies are works (bunds,
    # levelled fields, channels): they need the canals technology (without a technology
    # system, they are assumed).
    paddy = False
    if wet >= 0 and (canals or not has_tech):
        paddy = ((tamed[i] >> np.uint32(wet)) & np.uint32(1)) != 0
    use = flood_dry[i]
    if paddy and suit_wet[i] > use:
        use = min(suit_wet[i], 1.0)
    water = min(max(flood[i] / 0.4, 0.0), 1.0) * use
    dug = irrigable[i] * dryness[i] * canals
    if dug > water:
        water = dug
    # Paddies on level land are watered by streams, ponds and the monsoon rain held in
    # their bunds (Japan, Korea, the Yangtze hills, Bengal): up to stream[i] of the way to
    # the irrigated ceiling, on the level share of the cell.
    level = min(max(1 - slope[i] / (20.0 if terraces else 10.0), 0.0), 1.0)
    if paddy:
        fed = stream[i] * level * min(max(suit_wet[i], 0.0), 1.0)
        if fed > water:
            water = fed
    steep = slope[i] / (2 * steep_slope if terraces else steep_slope)
    quality = min(max(arable_eff[i] / 0.6, 0.3), 1.0) * min(max(1 - steep, 0.3), 1.0)
    rain = (rainfed - 1) * (1 + plough_c * plough + manure_c * manure)
    top = (rain + (irrigated - 1 - rain) * water) * quality
    if paddy:
        # Paddies are level fields: valley floors, plains and deltas (terraced hillsides
        # with terracing); hill rice elsewhere is dry-field rice.
        top = top * (1 + (paddy_c - 1) * min(max(suit_wet[i], 0.0), 1.0) * level)
    return 1 + top


@njit(cache=True, parallel=True)
def _ceiling_kernel(tech, has_tech, bits, tamed, wet, suit_wet, irrigable, dryness, flood, flood_dry, slope,
                    stream, arable_eff, consts, out):
    for i in prange(out.shape[0]):
        out[i] = _ceiling_cell(i, tech, has_tech, bits, tamed, wet, suit_wet, irrigable, dryness, flood, flood_dry,
                               slope, stream, arable_eff, consts)


@njit(cache=True, parallel=True)
def _capacity_kernel(land, crop_best, crop_sum, herd, forage, precip, cold,
                     tech, has_tech, bits, tamed, wet, suit_wet, irrigable, dryness, flood_c, flood_dry, slope,
                     stream, arable_eff, consts, intensity, has_intensity, arable, flood, grazing, area,
                     farmer_density, herder_density, flood_yield, cap, mode):
    for i in prange(land.shape[0]):
        if not land[i]:
            continue
        weather = min(max(precip[i], 0.3), 1.2) * (1 - 0.08 * max(0.0, -cold[i]))
        crops = crop_best[i] + 0.25 * (crop_sum[i] - crop_best[i])
        farm = farmer_density * crops * arable[i] * weather * (1 + flood_yield * flood[i])
        if has_intensity:
            ceil = _ceiling_cell(i, tech, has_tech, bits, tamed, wet, suit_wet, irrigable, dryness, flood_c,
                                 flood_dry, slope, stream, arable_eff, consts)
            farm = farm * (1 + (ceil - 1) * intensity[i])
        herding = herder_density * min(herd[i], 2.0) * grazing[i] * weather
        a = max(area[i], 1e-6)
        best, m = forage[i] / a, 1
        if herding > best:
            best, m = herding, 2
        if farm + 0.3 * herding > best:
            best, m = farm + 0.3 * herding, 3
        c = best * area[i]
        cap[i] = c
        mode[i] = m if c > 0 else 0


@njit(cache=True, parallel=True)
def _intensify_kernel(intensity, pop, cap, mode, area, own, has_own, farming_mode, dt, intensify_pressure,
                      intensify_density, intensity_neighbour, state_intensify, intensify_years, intensity_start,
                      decay_years, push_floor, known, borrow):
    h, w = intensity.shape
    old = intensity.copy()
    for yy in prange(h):
        y = np.int64(yy)
        up = y - 1 if y > 0 else y
        down = y + 1 if y < h - 1 else y
        for x in range(w):
            i0 = old[y, x]
            p, c = pop[y, x], cap[y, x]
            farming = mode[y, x] == farming_mode and p > 0
            pressure = p / max(c, 1e-6) if c > 0 else 0.0
            if farming:
                filling = min(max((pressure - intensify_pressure) / 0.1 + 0.5, 0.0), 1.0)
                density = p / max(area[y, x], 1e-6)
                # Denser farmland intensifies faster; land that is full but thin still does,
                # slowly (push_floor): Andean and Mesoamerican farmers built terraces and
                # raised fields without ever being as dense as the Ganges.
                push = filling * min(max(density / intensify_density, push_floor), 3.0)
                lead = 0.0
                for n in (old[y, x + 1 if x < w - 1 else 0], old[y, x - 1 if x > 0 else w - 1],
                          old[down, x], old[up, x]):
                    if n - i0 > lead:
                        lead = n - i0
                speed = 1 + intensity_neighbour * min(lead, 1.0)
                if has_own and own[y, x] > 0:
                    speed *= 1 + state_intensify
                # Compounding from what the farmers know: their own methods, or a share of the
                # best worked within reach (methods travel with migrants and traders: Yayoi
                # rice farmers brought Korea's paddies to Japan ready-made).
                base = max(i0, borrow * known[y, x])
                gain = dt / intensify_years * push * speed * (intensity_start + base)
                v = i0 + min(gain, 1.0) * (1 - i0)
                slack = min(max((0.5 - pressure) / 0.5, 0.0), 1.0)
            else:
                v = i0
                slack = 1.0
            intensity[y, x] = v * 0.5 ** (dt / decay_years * slack)


@njit(cache=True)
def _food_kernel(tamed, suit, maturity, food, is_crop, crop_best, crop_sum, herd):
    """Food per km2 from each cell's tamed species: the best crop, all crops, and herds."""
    n = suit.shape[0]
    for c in range(tamed.shape[0]):
        bits = tamed[c]
        if bits == 0:
            continue
        for i in range(n):
            if (bits >> np.uint32(i)) & np.uint32(1):
                v = suit[i, c] * maturity[i, c] * food[i]
                if is_crop[i]:
                    if v > crop_best[c]:
                        crop_best[c] = v
                    crop_sum[c] += v
                else:
                    herd[c] += v


@njit(cache=True)
def _mature_kernel(tamed, maturity, gains):
    """Tamed crops and animals improve toward full maturity."""
    n = maturity.shape[0]
    for c in range(tamed.shape[0]):
        bits = tamed[c]
        if bits == 0:
            continue
        for i in range(n):
            if (bits >> np.uint32(i)) & np.uint32(1):
                m = maturity[i, c]
                maturity[i, c] = m + (1 - m) * gains[i]


@njit(cache=True)
def _frontier(tamed, i, people, suit, nbr):
    """Cells species `i` could spread into (not grown there, a neighbour grows it, people
    live there, it can grow there) and how many neighbours grow it."""
    bit = np.uint32(1) << np.uint32(i)
    n = tamed.shape[0]
    out = np.empty(n, np.int64)
    near = np.empty(n, np.int8)
    k = 0
    for c in range(n):
        if (tamed[c] & bit) or not people[c] or not suit[c] > 0.3:
            continue
        cnt = 0
        for d in range(4):
            if tamed[nbr[d, c]] & bit:
                cnt += 1
        if cnt > 0:
            out[k] = c
            near[k] = cnt
            k += 1
    return out[:k], near[:k]


@njit(cache=True)
def _edge_cells(here, corridor, nbr):
    """Cells on a river or coast corridor where a species is grown and at least one
    neighbour isn't (pioneers set out from these)."""
    n = here.shape[0]
    out = np.empty(n, np.int64)
    k = 0
    for c in range(n):
        if not here[c] or not corridor[c]:
            continue
        cnt = 0
        for d in range(4):
            if here[nbr[d, c]]:
                cnt += 1
        if cnt < 4:
            out[k] = c
            k += 1
    return out[:k]


@njit(cache=True)
def _all_fronts(tamed, people, suit_ok, corridor, nbr, n):
    """For every species at once: its frontier cells (where it could spread, as `_frontier`)
    with how many neighbours grow it, and its pioneer cells (as `_edge_cells`), each sorted
    by cell and grouped by species (`*_start[i]` .. `*_start[i + 1]`)."""
    cells = tamed.shape[0]
    f_count = np.zeros(n, np.int64)
    e_count = np.zeros(n, np.int64)
    for pass_ in range(2):
        if pass_ == 1:
            f_start = np.zeros(n + 1, np.int64)
            e_start = np.zeros(n + 1, np.int64)
            for i in range(n):
                f_start[i + 1] = f_start[i] + f_count[i]
                e_start[i + 1] = e_start[i] + e_count[i]
            f_cell = np.empty(f_start[n], np.int64)
            f_near = np.empty(f_start[n], np.int8)
            e_cell = np.empty(e_start[n], np.int64)
            f_fill = f_start[:n].copy()
            e_fill = e_start[:n].copy()
        for c in range(cells):
            bits = tamed[c]
            n0 = tamed[nbr[0, c]]
            n1 = tamed[nbr[1, c]]
            n2 = tamed[nbr[2, c]]
            n3 = tamed[nbr[3, c]]
            around = n0 | n1 | n2 | n3
            if bits == 0 and around == 0:
                continue
            for i in range(n):
                b = np.uint32(1) << np.uint32(i)
                if bits & b:
                    if corridor[c]:
                        cnt = (n0 & b != 0) + (n1 & b != 0) + (n2 & b != 0) + (n3 & b != 0)
                        if cnt < 4:
                            if pass_ == 0:
                                e_count[i] += 1
                            else:
                                e_cell[e_fill[i]] = c
                                e_fill[i] += 1
                elif around & b and people[c] and suit_ok[i, c]:
                    cnt = (n0 & b != 0) + (n1 & b != 0) + (n2 & b != 0) + (n3 & b != 0)
                    if pass_ == 0:
                        f_count[i] += 1
                    else:
                        f_cell[f_fill[i]] = c
                        f_near[f_fill[i]] = cnt
                        f_fill[i] += 1
    return f_start, f_cell, f_near, e_start, e_cell


def rng_index(t, take):
    """Positions of the (sorted) cells `take` within the sorted cells `t`."""
    return np.searchsorted(t, take)


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
        # Crops are sown for a season: rice and maize in the warm half of the year (Japan,
        # Korea, the American corn belt), winter wheat in the cool half (the Mediterranean).
        self.suit = np.maximum(self.suit, _seasonal_suitability(world, self.is_crop, land))
        for i, sp in enumerate(SPECIES):
            if sp.key in TSETSE_SENSITIVE:
                self.suit[i][world.tsetse > cfg.tsetse_block] = 0
        self.suit[:, ~land] = 0
        self.grazing = _by_biome(world.biome, OPENNESS, 0.5)
        self.raid_open = _by_biome(world.biome, RAID_OPEN, 0.25).astype(np.float32)
        # River water: crops on floodplains (and, with canals, irrigable land) don't need rain.
        self.flood = getattr(world, "floodplain", None)
        self.flood = np.zeros(land.shape, np.float32) if self.flood is None else self.flood.astype(np.float32)
        self.flood = np.where(land, band(self.flood, self.scale), 0).astype(np.float32)   # same width in km
        self.irrigable = getattr(world, "irrigable", None)
        self.irrigable = self.flood if self.irrigable is None else self.irrigable.astype(np.float32)
        self.suit_dry = self.suit.copy()
        # Small streams, springs and ponds (worldgen's fresh water: 0.5-0.6 in humid lands
        # away from big rivers): what waters level paddies off the floodplains.
        fresh = getattr(world, "fresh_water", None)
        self.stream = (np.zeros(land.shape, np.float32) if fresh is None
                       else np.where(land, fresh, 0).astype(np.float32))
        self.suit_watered = _watered_suitability(world, self.is_crop, land)
        watered_arable = (world.soil_fertility * np.clip(1 - world.slope_m_per_km / 60, 0, 1)
                          * np.clip(world.warm_months / 6, 0, 1) * (1 - 0.6 * world.wetland))
        self.arable_dry = world.arable.astype(np.float32)
        self.arable_watered = np.where(land, watered_arable, 0).astype(np.float32)
        self._canals = None
        self.world_native = world.species_native
        self._apply_water(cfg)
        # Taming odds barely depend on how much land a species grows wild on: it is tamed by
        # the people who live with it, not by the size of its range. Each cell's chance is
        # scaled so a whole range's chance grows only as (wild area) ** domestication_range_exponent
        # (otherwise wide-ranging pigs and cattle always beat narrow-ranged wheat and llamas).
        wild_km2 = (self.area[None] * self.suit_dry * self.world_native).reshape(self.n, -1).sum(1)
        self.range_factor = ((cfg.domestication_reference_km2 / np.maximum(wild_km2, 1.0))
                             ** (1 - cfg.domestication_range_exponent))
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
        self.coastal = coast > 0
        patchy = np.exp(0.8 * fbm(Perlin3(world.cfg.seed * 7919 + state.seed * 31 + 271828), world.grid.xyz, 8, 3))
        self.spread_ease = ((1 + cfg.river_corridor * river + cfg.coast_corridor * coast)
                            * (0.15 + self.arable_eff) ** 1.5 * patchy
                            / (1 + world.slope_m_per_km / 15)).astype(np.float32)
        # Herds go where there is grazing, not farmland: herders spread fast across open
        # steppe (the Yamnaya across the Pontic-Caspian steppe, herders into Mongolia).
        self.herd_ease = ((1 + cfg.river_corridor * river + cfg.coast_corridor * coast)
                          * (0.15 + 0.85 * self.grazing) ** 1.5 * patchy
                          / (1 + world.slope_m_per_km / 15)).astype(np.float32)
        self.landmass = world.landmass_id
        # Malaria: the climate for its mosquitoes (worldgen's malaria field) where the parasite
        # is present. On Earth it reached the Americas only after 1492.
        malaria = getattr(world, "malaria", None)
        malaria = np.zeros(world.ocean.shape, np.float32) if malaria is None else malaria.astype(np.float32)
        from worldgen.earth import is_earth
        if is_earth(world.cfg):
            malaria = np.where(world.grid.lon < -30, 0, malaria).astype(np.float32)
        self.health = (1 - cfg.malaria_burden * np.clip(malaria, 0, 1)).astype(np.float32)
        area = np.asarray(world.landmass_area)
        self._island_mask = (area[self.landmass] <= cfg.island_max_share) & (self.landmass > 0)
        from worldgen import BIOME_ID
        # Rainforest, and the swamp forest of hot, very wet wetlands (most of the Amazon and
        # Congo basins on the map), which stone tools can't keep cleared either.
        swamp_forest = (world.biome == BIOME_ID["Wetland"]) & (world.temperature_c >= 20) & (world.precip_mm >= 1500)
        self._rainforest = np.isin(world.biome, [BIOME_ID["Tropical rainforest"], BIOME_ID["Temperate rainforest"]]) | swamp_forest
        self.km_per_px = world.grid.km_per_px
        self.dpp = world.grid.dpp
        self.cos_lat = np.maximum(np.cos(np.radians(world.grid.lat[:, 0])), 0.05)
        self.slope = world.slope_m_per_km.astype(np.float32)
        # Canals transform dry land; where rain already waters the fields they add little.
        # How much a flood's water is worth, by aridity (rain / what could evaporate): all of
        # it in deserts and dry steppe (the Nile, Mesopotamia, the Indus), none where rain
        # meets the need (the Danube, the Amazon). See _ceiling_cell.
        evap = getattr(world.cfg, "evaporation_per_degree_mm", 55.0) * np.maximum(world.temperature_c, 1.0)
        aridity = world.precip_mm / evap
        self.flood_dry = np.clip((cfg.flood_water_none_aridity - aridity)
                                 / (cfg.flood_water_none_aridity - cfg.flood_water_full_aridity), 0, 1
                                 ).astype(np.float32)
        # Canals are worth digging where it's dry for the warmth (rain well short of what could
        # evaporate): North China, the Deccan and the Ganges, not cool Russia at 700 mm a year.
        self.dryness = np.clip((cfg.canal_water_none_aridity - aridity)
                               / (cfg.canal_water_none_aridity - cfg.canal_water_full_aridity), 0, 1
                               ).astype(np.float32)
        # A flood renews the soil every year, so floodplain fields are cropped every year from
        # the start (no fallow): as productive as fully intensified rain-fed fields. As a share
        # of the irrigated ceiling, the least intensity a farmed floodplain has.
        self.flood_floor = (np.clip(self.flood / 0.4, 0, 1) * self.flood_dry
                            * (cfg.rainfed_ceiling - 1) / (cfg.irrigated_ceiling - 1)).astype(np.float32)
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
        flood = np.clip(self.flood / 0.4, 0, 1)
        water = np.maximum(flood, dug * self.irrigable)
        watered = self.suit_watered * water[None]
        # Herds graze flood-watered valleys too (cattle, sheep and goats along the Nile and
        # the Euphrates, where it hardly rains), though not canal-watered fields.
        pasture = self.suit_watered * flood[None]
        self.suit = np.where(self.is_crop[:, None, None], np.maximum(self.suit_dry, watered),
                             np.maximum(self.suit_dry, pasture))
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
        n_sp = self.n
        _food_kernel(tamed.ravel(), self.suit.reshape(n_sp, -1), maturity.reshape(n_sp, -1), self.food,
                     self.is_crop, crop_best.ravel(), crop_sum.ravel(), herd.ravel())
        # The rest, cell by cell (compiled): a varied package of crops helps; the weather;
        # floodplains renew their soil every year (silt: up to 1 + flood_yield times the
        # harvest); intensity up to each cell's ceiling; herding on open land. Each cell lives
        # by whichever of foraging, herding and farming feeds the most people.
        if climate is not None:
            precip = np.ascontiguousarray(climate.precip_factor, np.float32)
            cold = np.ascontiguousarray(climate.temperature_offset_c, np.float32)
        else:
            precip = np.ones(land.shape, np.float32)
            cold = np.zeros(land.shape, np.float32)
        intensity = state.cells.get("intensity")
        cap = np.zeros(land.shape, np.float32)
        mode = np.zeros(land.shape, np.uint8)
        _capacity_kernel(np.ascontiguousarray(land).ravel(), crop_best.ravel(), crop_sum.ravel(), herd.ravel(),
                         np.ascontiguousarray(forage, np.float32).ravel(), precip.ravel(), cold.ravel(),
                         *self._ceiling_inputs(state),
                         np.zeros(1, np.float32) if intensity is None else intensity.ravel(),
                         intensity is not None, self._raided(state, self._cleared_arable(state)).ravel(),
                         self.flood.ravel(),
                         self.grazing.ravel(),
                         self.area.ravel(), cfg.farmer_density, cfg.herder_density, cfg.flood_yield,
                         cap.ravel(), mode.ravel())
        # Malaria thins people where the parasite and its mosquitoes live (hot, wet low ground).
        cap *= self.health
        return cap, mode

    def storable_share(self, state):
        """How much of each cell's crop food keeps and can be counted and taxed (0..1): the
        best storable crop's yield over the best crop's, squared. Grain ripens at once and keeps for
        years; roots, tubers and tree crops are dug or picked as needed and rot (cells with
        no crops count as 1)."""
        tamed = state.cells["tamed"]
        mat = state.cells["maturity"]
        best = np.zeros(tamed.shape, np.float32)
        best_store = np.zeros(tamed.shape, np.float32)
        for i in np.flatnonzero(self._present(tamed) & self.is_crop & (self.food > 0)):
            v = np.where((tamed & np.uint32(1 << i)) != 0, self.suit[i] * mat[i] * self.food[i], 0)
            np.maximum(best, v, out=best)
            if SPECIES[i].key in STORABLE:
                np.maximum(best_store, v, out=best_store)
        # Squared: people mostly eat the staple that yields best, so a storable crop well
        # behind the root crops (beans beside manioc) counts for little.
        return np.where(best > 0, (best_store / np.maximum(best, 1e-9)) ** 2, 1.0).astype(np.float32)

    def _known_methods(self, state, intensity):
        """The most intensive farming within intensity_reach_km of each cell (across narrow
        seas too), recomputed every 50 years."""
        cfg = state.config
        if cfg.intensity_borrow <= 0:
            return np.zeros(intensity.shape, np.float32)
        if getattr(self, "_known_year", None) is None or state.year - self._known_year >= 50:
            from scipy.ndimage import maximum_filter
            self._known_year = state.year
            r = max(1, int(round(cfg.intensity_reach_km / state.world.grid.km_per_px)))
            self._known = maximum_filter(intensity, size=2 * r + 1, mode=("nearest", "wrap")).astype(np.float32)
        return self._known

    def _raided(self, state, arable):
        """Farmland share left worth sowing where mounted herders raid: open land within
        raid_km of herders who ride is raided in proportion to how much of the land around
        is theirs (the Pontic and Kazakh steppe stayed pasture until the 1700s, held by
        Scythians, Huns, Cumans and Tatars; farming stopped at the forest edge). A kingdom or
        empire shields its farmers in part (the Great Wall, Kievan forts)."""
        cfg = state.config
        tech = state.cells.get("tech")
        if cfg.raid_burden <= 0 or tech is None or "horse" not in KEYS:
            return arable
        from ..tech import INDEX
        riding = (tech & np.uint64(1 << INDEX["riding"])) != 0
        mounted = (state.cells["subsistence"] == HERDING) & riding & self.has(state, KEYS.index("horse"))
        if not mounted.any():
            return arable
        key = (state.year // 50, int(np.count_nonzero(mounted)))
        if getattr(self, "_raid_key", None) != key:
            self._raid_key = key
            from scipy.ndimage import uniform_filter
            r = max(1, int(round(cfg.raid_km / state.world.grid.km_per_px)))
            share = uniform_filter(mounted.astype(np.float32), size=2 * r + 1, mode=("nearest", "wrap"))
            pressure = np.clip(share / cfg.raid_share_full, 0, 1) * self.raid_open
            # Floodplain farmland is packed with walled villages and towns (Mesopotamia, the Nile).
            pressure = pressure * (1 - 0.7 * np.clip(self.flood / 0.4, 0, 1))
            own = state.cells.get("state")
            if own is not None:
                records = state.tables.get("states") or [None]
                big = np.array([r is not None and r["kind"] in ("kingdom", "empire") for r in records], bool)
                pressure = np.where(big[own], pressure * (1 - cfg.raid_state_shield), pressure)
            self._raid = (1 - cfg.raid_burden * pressure).astype(np.float32)
        return (arable * self._raid).astype(np.float32)

    def _cleared_arable(self, state):
        """Farmland share people can keep in crops: in rainforest, felling and keeping back
        the forest with stone tools holds it to (1 - rainforest_clearing) of the land, until
        they know iron (steel axes fell trees many times faster; Amazonia, with no metal
        tools, stayed thinly farmed away from its rivers; iron came with farmers into the
        Congo forest). Rechecked when the iron-using area changes."""
        cfg = state.config
        if not self._rainforest.any() or cfg.rainforest_clearing <= 0:
            return self.arable
        iron = self.knows(state, "iron") if "tech" in state.cells else np.zeros(state.shape, bool)
        key = int(np.count_nonzero(iron & self._rainforest))
        if getattr(self, "_cleared_key", None) != key:
            self._cleared_key = key
            slow = self._rainforest & ~iron
            self._cleared = np.where(slow, self.arable * (1 - cfg.rainforest_clearing), self.arable
                                     ).astype(np.float32)
        return self._cleared

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
        more for rice-like paddies; less on poor or steep land (terraces halve that).
        (Worked out cell by cell in `_ceiling_cell`, shared with `capacity`.)"""
        out = np.zeros(state.shape, np.float32)
        _ceiling_kernel(*self._ceiling_inputs(state), out.ravel())
        return out

    def _ceiling_inputs(self, state):
        from ..tech import INDEX
        cfg = state.config
        tech = state.cells.get("tech")
        tech = np.zeros(1, np.uint64) if tech is None else tech.ravel()
        wet = self.wet_grain if self.wet_grain is not None else -1
        suit_wet = self.suit[wet].ravel() if wet >= 0 else np.zeros(1, np.float32)
        bits = np.array([INDEX["canals"], INDEX["terraces"], INDEX["plough"], INDEX["manuring"]], np.int64)
        consts = np.array([cfg.steep_slope, cfg.rainfed_ceiling, cfg.plough_ceiling, cfg.manuring_ceiling,
                           cfg.irrigated_ceiling, cfg.paddy_ceiling], np.float64)
        return (tech, len(tech) > 1, bits, state.cells["tamed"].ravel(), wet, suit_wet, self.irrigable.ravel(),
                self.dryness.ravel(), self.flood.ravel(), self.flood_dry.ravel(), self.slope.ravel(),
                self.stream.ravel(), self.arable_eff.ravel(), consts)

    def _intensify(self, state):
        """Crowded farmland is worked harder; emptied or abandoned land lets it go."""
        from .geo import shift
        cfg, dt = state.config, self.every
        intensity = state.cells["intensity"]
        cap = state.cells.get("capacity")
        mode = state.cells.get("subsistence")
        if cap is None or mode is None:
            return
        pop = state.cells["population"]
        own = state.cells.get("state")
        # Farmers intensify once their land is filling up, and the more of them there are per
        # km2 the faster (Boserup: density drives it, and intensive land holds more people,
        # so dense places run ahead while thin shifting-farming land stays as it is); faster
        # next to more intensive land and inside a state; slow to start, then faster as
        # methods, works and markets build on each other. Emptied land (people well below
        # what it feeds) or land no longer farmed decays. (Compiled, cell by cell.)
        _intensify_kernel(intensity, pop, cap, mode, self.area,
                          own if own is not None else np.zeros(pop.shape, np.uint16), own is not None,
                          FARMING, dt, cfg.intensify_pressure, cfg.intensify_density, cfg.intensity_neighbour,
                          cfg.state_intensify, cfg.intensify_years, cfg.intensity_start, cfg.intensity_decay_years,
                          cfg.intensify_floor, self._known_methods(state, intensity), cfg.intensity_borrow)
        farmed = (mode == FARMING) & (pop > 0)
        np.maximum(intensity, np.where(farmed, self.flood_floor, 0), out=intensity)

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
        gains = (1 - np.exp(-dt / (cfg.crop_maturity_years * np.sqrt(self.difficulty)))).astype(np.float32)
        _mature_kernel(tamed.ravel(), maturity.reshape(self.n, -1), gains)

        tamed_flat = tamed.ravel()
        # Domestication.
        holocene = np.clip((state.year - cfg.holocene_start_year) / 500, 0, 1)
        if state.climate is not None and state.climate.global_temperature_c < -1.5:
            holocene *= 0.1
        if holocene > 0:
            crowding = np.clip(pop / np.maximum(state.cells["forager_capacity"], 1e-6), 0, 1.5)
            lived = np.flatnonzero(people)
            for i, sp in enumerate(SPECIES):
                # Only cells where it can happen.
                cells = lived[self.can_tame[i].ravel()[lived]]
                cells = cells[(tamed_flat[cells] & np.uint32(1 << i)) == 0]
                if not len(cells):
                    continue
                # Needs real crowding (pressure to find more food), and a species worth
                # the effort: food plants and animals before fibre.
                hazard = (cfg.domestication_rate * self.range_factor[i] * self.scale ** 2 * holocene
                          * self.suit[i].ravel()[cells]
                          * crowding.ravel()[cells] ** 3 / sp.difficulty * (0.15 + sp.food)
                          * np.where(farms_anything.ravel()[cells], cfg.farmer_domestication_bonus, 1.0))
                # One origin per species per step at most: whether any cell tames it (the
                # chance that at least one does), and if so which (in proportion to each
                # cell's chance). Two random numbers instead of one per cell.
                total = float(hazard.sum()) * dt
                if rng.random() < 1 - np.exp(-total):
                    chance = 1 - np.exp(-hazard * dt)
                    cell = int(cells[min(int(np.searchsorted(np.cumsum(chance), rng.random() * chance.sum())),
                                         len(cells) - 1)])
                    self._domesticate(state, i, cell)

        # Canals water dry irrigable land where they are known (rechecked every 50 years).
        if state.year - getattr(self, "_water_year", -10 ** 9) >= 50:
            self._water_year = state.year
            self._apply_water(cfg, self.knows(state, "canals") if "tech" in state.cells else None)
        # Spread to neighbours (in several rounds on fine maps: see geo.rounds_for).
        from .geo import shift
        forage = state.cells["forager_capacity"] / np.maximum(self.area, 1e-6)
        rounds = rounds_for(self.scale)
        pace = 1 / (self.scale * rounds)
        # Every species' frontier and pioneer cells, in one pass over the map (a species'
        # spread this step changes only its own cells, so these hold until its turn).
        if getattr(self, "_nbr", None) is None:
            self._nbr = neighbour_index(tamed.shape)
        if getattr(self, "_suit_ok_of", None) is not self.suit:
            self._suit_ok = np.ascontiguousarray((self.suit > 0.3).reshape(self.n, -1))
            self._suit_ok_of = self.suit
        f_start, f_cell, f_near, e_start, e_cell = _all_fronts(
            tamed.ravel(), np.ascontiguousarray(people).ravel(), self._suit_ok, self.corridor.ravel(), self._nbr,
            self.n)
        for i in np.flatnonzero(self._present(tamed)):
            here = (tamed & np.uint32(1 << i)) != 0          # where it was before this step
            first = (f_cell[f_start[i]:f_start[i + 1]], f_near[f_start[i]:f_start[i + 1]])
            for r in range(rounds):
                self._spread(state, i, pace, forage, people, farms_anything, rng, first if r == 0 else None)
            self._pioneers(state, i, here, people, forage, farms_anything, rng, e_cell[e_start[i]:e_start[i + 1]])
        # Crops pass between farming peoples by contact and trade (checked every century).
        if cfg.crop_exchange_rate > 0 and state.year - getattr(self, "_exchange_year", -10 ** 9) >= cfg.crop_exchange_every:
            self._exchange_year = state.year
            self._exchange(state, rng)
        self._stats(state)

    def _exchange(self, state, rng):
        """Farmers take up a crop or animal that does well on their land when farmers within
        `crop_exchange_km` keep it: seeds and stock pass between farming peoples by contact
        and trade far faster than farming itself spreads (wheat and barley reached Egypt's
        farmers across Sinai by ~5500 BCE, wheat reached China by ~2500 BCE, sorghum crossed
        to India). Unlike the frontier, this needs no chain of good land in between.

        Islands where hardly anyone farms yet are settled the same way by coastal farmers
        within reach, by boat: farming crosses straits with people in boats, as
        colonists, whether or not the foragers there would take it up (Yayoi farmers from
        Korea to Japan, Austronesians through Indonesia, Neolithic farmers to Britain),
        which the frontier, on land, never can."""
        cfg = state.config
        tamed = state.cells["tamed"]
        people = state.cells["population"] > 0
        # Peoples who live by farming or herding (a fibre plot among foragers is no farm).
        farms = (state.cells["subsistence"] >= HERDING) & people
        if not farms.any():
            return
        radius = cfg.crop_exchange_km / self.km_per_px          # in rows (km north-south)
        if radius < 1:
            return
        maturity = state.cells["maturity"]
        # Land masses where hardly anyone farms or herds yet (under a quarter of the cells
        # with people), and who can sail there.
        lm = self.landmass.ravel()
        n_lm = int(lm.max()) + 1
        share = (np.bincount(lm, weights=farms.ravel(), minlength=n_lm)
                 / np.maximum(np.bincount(lm, weights=people.ravel(), minlength=n_lm), 1))
        # Islands only (up to about New Guinea or Borneo): on a continent the frontier
        # carries farming, and a thinly farmed continent is not open to boat colonists.
        island = self._island_mask
        unfarmed = people & ~farms & (share[self.landmass] < 0.25) & (self.landmass > 0) & island
        sailors = self.coastal              # paddled boats cross a strait; sails aren't needed
        for i in np.flatnonzero(self._present(tamed)):
            bit = np.uint32(1 << i)
            here = (tamed & bit) != 0
            good = self.suit[i] >= 0.5
            cand = farms & ~here & good
            if cand.any():
                near = self._within(here, cand, radius)
                # By sea, coast to coast, contact reaches twice as far (the Yellow Sea, the
                # Mediterranean, the Indian Ocean).
                far = cand & ~near & self.coastal
                if far.any() and (here & self.coastal).any():
                    near |= self._within(here & self.coastal, far, 2 * radius) & self.coastal
                cand &= near
            isle = unfarmed & good
            if isle.any() and (here & sailors).any():
                isle &= self._within(here & sailors, isle, radius)
            else:
                isle = np.zeros_like(cand)
            idx = np.flatnonzero(cand | isle)
            if not len(idx):
                continue
            chance = 1 - np.exp(-cfg.crop_exchange_rate * cfg.crop_exchange_every * self.suit[i].ravel()[idx])
            take = idx[rng.random(len(idx)) < chance]
            if len(take):
                tamed.ravel()[take] |= bit
                mat = 0.9 * float(np.median(maturity[i][here]))
                if self.is_crop[i]:
                    # Bred for where it is mostly grown: far north or south of that it must be
                    # bred again (maize took millennia to suit the Mississippi valley).
                    home_row = float(np.median(np.nonzero(here)[0]))
                    rows = take // tamed.shape[1]
                    mat = mat * np.clip(1 - cfg.latitude_adaptation * np.abs(rows - home_row) * self.dpp, 0, 1)
                maturity[i].ravel()[take] = mat
                self._first_landmass_events(state, i, take)

    def _within(self, src, where, radius):
        """Cells of `where` within `radius` rows' distance (in km, so wider in columns toward
        the poles) of any cell of `src`; worked out in bands of latitude, wrapping east-west."""
        from scipy.ndimage import binary_dilation
        h, w = src.shape
        reach = int(np.ceil(radius))
        near = np.zeros_like(src)
        for r0 in range(0, h, 16):
            r1 = min(r0 + 16, h)
            if not where[r0:r1].any():
                continue
            cos = float(max(self.cos_lat[(r0 + r1) // 2], 0.2))
            wide = min(int(np.ceil(radius / cos)), w // 2)
            yy, xx = np.ogrid[-reach:reach + 1, -wide:wide + 1]
            shape = (yy / radius) ** 2 + (xx * cos / radius) ** 2 <= 1
            a0, a1 = max(r0 - reach, 0), min(r1 + reach, h)
            padded = np.pad(src[a0:a1], ((0, 0), (wide, wide)), mode="wrap")
            grown = binary_dilation(padded, structure=shape)[:, wide:-wide]
            near[r0:r1] = grown[r0 - a0:r1 - a0]
        return near

    def _spread(self, state, i, pace, forage, people, farms_anything, rng, front=None):
        """One round of species i spreading to neighbouring cells (worked out only on the
        cells it could reach; the random draw still covers the whole map)."""
        cfg, dt = state.config, self.every
        tamed = state.cells["tamed"]
        maturity = state.cells["maturity"]
        here = (tamed & np.uint32(1 << i)) != 0
        if getattr(self, "_nbr", None) is None:
            self._nbr = neighbour_index(here.shape)
        if front is not None:
            t, near_t = front
        else:
            t, near_t = _frontier(tamed.ravel(), i, np.ascontiguousarray(people).ravel(), self.suit[i].ravel(),
                                  self._nbr)
        if len(t):
            # The variety the neighbours grow (newcomers get it slightly less adapted).
            if getattr(self, "_nbr", None) is None:
                self._nbr = neighbour_index(here.shape)
            mat, held = maturity[i].ravel(), here.ravel()
            best = np.zeros(len(t), maturity.dtype)
            w = here.shape[1]
            for k in range(4):
                nb = self._nbr[k][t]
                # A crop moved north or south must be bred again for the new seasons and day
                # lengths (it spread east-west far more easily: Diamond's continental axes).
                fit = 1.0
                if self.is_crop[i]:
                    fit = np.clip(1 - cfg.latitude_adaptation * np.abs(t // w - nb // w) * self.dpp, 0, 1)
                best = np.maximum(best, np.where(held[nb], mat[nb] * fit, 0))
            welcome = self._welcome_at(cfg, i, t, best, forage, farms_anything)
            chance = 1 - np.exp(-cfg.spread_rate * pace * dt * (near_t / 4) * (0.5 + self.suit[i].ravel()[t])
                                * (self.spread_ease if self.is_crop[i] else self.herd_ease).ravel()[t] * welcome)
            take = t[draws_at(rng, here.size, t) < chance]
            if len(take):
                tamed.ravel()[take] |= np.uint32(1 << i)
                mat[take] = best[rng_index(t, take)] * 0.97
                self._first_landmass_events(state, i, take)

    def _welcome_at(self, cfg, i, t, maturity, forage, farms_anything):
        """`_welcome` for the flat cells `t` only."""
        value = self.suit[i].ravel()[t] * maturity * self.food[i]
        if self.is_crop[i]:
            pot = cfg.farmer_density * value * self.arable.ravel()[t] * (1 + cfg.flood_yield * self.flood.ravel()[t])
        else:
            pot = cfg.herder_density * np.minimum(value, 2.0) * self.grazing.ravel()[t]
        ratio = pot / np.maximum(forage.ravel()[t], 1e-6)
        doubt = np.clip((ratio - 1) / (cfg.farming_advantage - 1), cfg.frontier_floor, 1)
        return np.where(farms_anything.ravel()[t], 1.0, doubt).astype(np.float32)

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

    def _pioneers(self, state, i, here, people, forage, farms_anything, rng, sources=None):
        """Pioneers settle good river or coast land 50-300 km beyond the front, then the
        gaps fill in (the Linear Pottery farmers along central Europe's rivers, Bantu
        farmers along African rivers)."""
        cfg = state.config
        h, w = here.shape
        if getattr(self, "_nbr", None) is None:
            self._nbr = neighbour_index(here.shape)
        if sources is None:
            sources = _edge_cells(np.ascontiguousarray(here).ravel(), self.corridor.ravel(), self._nbr)
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
            if self.is_crop[i]:
                mat *= max(1 - cfg.latitude_adaptation * abs(tr - r) * self.dpp, 0.0)
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


def _seasonal_suitability(world, is_crop, land):
    """Rain-fed crops judged by their growing season instead of the whole year's mean
    temperature: warm-season crops by the warm half, cool-season crops by the cool half
    (a crop fits if the year or its season suits it); rain, the growing season and the
    rest as they are. Animals and wild ranges are unchanged."""
    from worldgen.nature import species_suitability
    out = np.zeros((len(is_crop),) + land.shape, np.float32)
    warmest = getattr(world, "warmest_month_c", None)
    coldest = getattr(world, "coldest_month_c", None)
    if warmest is None or coldest is None:
        return out
    fields = ("ocean", "lake", "biome", "elevation_m", "wetland", "precip_mm", "growing_months", "fresh_water")
    # Warm-season crops (comfortable from 14 C up: rice, maize, sorghum, tropical roots) gain
    # from summer heat; cool-season crops (comfortable up to 20 C: wheat, rye) from winter
    # sowing in warm climates. Not the other way round: Siberia's summer doesn't make wheat
    # fit a land whose winters kill it.
    warm_crop = np.array([s.kind == "crop" and s.temp[0] >= 14 for s in SPECIES])
    cool_crop = np.array([s.kind == "crop" and s.temp[1] <= 20 for s in SPECIES])
    # Tropical roots (comfortable from 20 C: yams, taro, manioc) are perennials that frost
    # kills: a hot summer is no use to them where winters freeze.
    tender = np.array([s.kind == "crop" and s.temp[0] >= 20 for s in SPECIES])
    frost_free = coldest >= 10
    for temp, which in (((world.temperature_c + warmest) / 2, warm_crop), ((world.temperature_c + coldest) / 2, cool_crop)):
        proxy = SimpleNamespace(**{k: getattr(world, k) for k in fields})
        proxy.temperature_c = temp
        season = species_suitability(proxy)
        season[tender] = np.where(frost_free, season[tender], 0)
        out[which] = np.maximum(out[which], season[which])
    out[~is_crop] = 0
    out[:, ~land] = 0
    return out


def _watered_suitability(world, is_crop, land):
    """Where each crop could grow (or herd graze) if water were no limit (river-fed fields): rainfall and
    the wet season stop mattering; warmth, height and the rest still do. Watered fields
    can also be sown in the cool half of the year (winter wheat and barley on the Nile,
    Euphrates and Indus floods), so a crop counts as fitting if either half suits it."""
    from worldgen.nature import species_suitability
    out = np.zeros((len(is_crop),) + land.shape, np.float32)
    fields = ("ocean", "lake", "biome", "elevation_m", "wetland")
    from worldgen.generator import BIOME_ID
    desert, scrub = BIOME_ID["Desert"], BIOME_ID["Scrubland"]
    coldest = getattr(world, "coldest_month_c", None)
    seasons = [world.temperature_c]
    if coldest is not None:
        seasons.append((world.temperature_c + coldest) / 2)    # mean of the cool half-year
    for temp in seasons:
        for rain in (800.0, 1500.0):            # inside the rain range of every crop
            proxy = SimpleNamespace(**{k: getattr(world, k) for k in fields})
            proxy.temperature_c = temp
            # River water adds to the rain; it can't take rain away (a rainforest floodplain
            # stays too wet for dry-country grains like sorghum).
            proxy.precip_mm = np.maximum(world.precip_mm, rain).astype(np.float32)
            proxy.growing_months = world.warm_months
            proxy.fresh_water = np.ones(land.shape, np.float32)      # wet-field crops too
            out = np.maximum(out, species_suitability(proxy))
            # A watered desert valley is green pasture (cattle on the Nile and the Euphrates):
            # for herds, a desert floodplain is scrub.
            proxy.biome = np.where(world.biome == desert, scrub, world.biome)
            out[~is_crop] = np.maximum(out[~is_crop], species_suitability(proxy)[~is_crop])
    out[:, ~land] = 0
    return out
