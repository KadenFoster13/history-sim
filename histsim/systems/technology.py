"""Technology: knowledge found, spread and lost, cell by cell (phase 4 step 1).

The web of technologies is in histsim/tech.py. Each cell keeps what its people know as a
bit mask (the `tech` layer), so knowledge has a geography:

  found     where a technology's needs and prerequisites are met: a small chance each
            step, rising with the people there, towns, and how many people live within
            ~300 km (more minds and more exchange: Kremer, Henrich). Hard technologies
            need big connected populations, so they are found once or a few times and
            then spread; easy ones arise in several places. Only a find far from anyone
            who already knows it (`tech_independent_km`) is a new invention; nearer, the
            knowledge would have spread anyway.
  spread    to neighbouring cells that can use it, ~2 km a year overland, faster along
            coasts and big rivers and inside states
  lost      where what keeps it alive is gone (no towns for writing, no tin for bronze,
            nobody left), fading over a few centuries

Each invention is recorded (the `inventions` table) and named in the inventors' own
language. Step 1 adds no effects: states, farming and war don't use technology yet.
"""
import numpy as np
from numba import njit, prange
from .geo import shift
from worldgen.hydrology import cell_river_flow

from ..core.simulation import System
from ..tech import INDEX, NEEDS, TECHS
from .geo import DIRECTIONS4, cell_scale, great_circle_km, rounds_for, touches
from .population import FARMING
from .settlements import _box

ALL = np.uint64(2 ** 64 - 1)
SETTLED = 0.15          # farmland this intensive counts as settled farming (as in the viewer)


class Technology(System):
    name = "technology"

    def __init__(self):
        self.every = None

    # ----------------------------------------------------------------- setup
    def setup(self, state):
        cfg, world = state.config, state.world
        self.every = cfg.tech_every
        self.scale = cell_scale(world, cfg)
        self.rng = state.rng.get("technology")
        self.kpp = world.grid.km_per_px
        h, w = world.ocean.shape
        self.n = h * w
        self.xyz = np.stack(world.grid.xyz, axis=-1).reshape(-1, 3)
        land = ~world.ocean & ~world.lake
        # Neighbour cell of every cell in each direction (clamped at the poles, wrapping).
        index = np.arange(self.n).reshape(h, w)
        self.neighbours = np.stack([shift(index, dy, dx).ravel() for dy, dx in DIRECTIONS4])
        # Fixed needs: water, hills, ore and river land within reach.
        river = cell_river_flow(world) > 1000
        static = {
            "water": land & (touches(world.ocean) | river),
            "hills": land & (world.slope_m_per_km > 8),
            "copper_near": self._near(world.copper > 0.15, cfg.copper_reach_km),
            "tin_near": self._near(world.tin > 0.02, cfg.tin_reach_km),
            "iron_near": self._near(world.iron > 0.3, cfg.iron_reach_km),
            "precious_near": self._near((world.gold > 0.1) | (world.silver > 0.1), cfg.precious_reach_km),
            "irrigable_near": self._near(land & (getattr(world, "irrigable", np.zeros_like(land, float)) > 0.3),
                                         cfg.irrigable_reach_km),
        }
        self.static = {k: v.ravel() for k, v in static.items()}
        # Bits of the technologies that need (or are kept by) each need.
        self.need_bits = {n: np.uint64(sum(1 << i for i, t in enumerate(TECHS) if n in t.needs)) for n in NEEDS}
        self.keep_bits = {n: np.uint64(sum(1 << i for i, t in enumerate(TECHS) if n in t.keep_needs)) for n in NEEDS}
        self.find_bits = {n: np.uint64(sum(1 << i for i, t in enumerate(TECHS) if n in t.find_needs)) for n in NEEDS}
        self.prereq = [(np.uint64(1 << i), np.uint64(sum(1 << INDEX[p] for p in t.prereqs)))
                       for i, t in enumerate(TECHS) if t.prereqs]
        from worldgen import SPECIES
        keys = [sp.key for sp in SPECIES]
        self.cattle = keys.index("cattle") if "cattle" in keys else None
        self.horse = keys.index("horse") if "horse" in keys else None

        state.add_layer("tech", np.uint64, record="changes")
        state.tables["inventions"] = self.inventions = []
        state.saved_tables.add("inventions")
        state.tables["technology"] = self
        self.first = {}                # tech key -> year first found
        self._stats(state, np.flatnonzero(state.cells["population"].ravel() > 0), None)

    def _near(self, mask, km):
        r = max(0, int(round(km / self.kpp)))
        return _box(mask.astype(np.uint8), r, np.maximum).astype(bool) if r else mask

    # ------------------------------------------------------------- needs
    def _needs(self, state, idx):
        """Each need, met or not, in the cells `idx` (flat indices of cells with people)."""
        cfg = state.config
        mode = state.cells["subsistence"].ravel()[idx]
        farming = mode == FARMING
        intensity = state.cells.get("intensity")
        settled = farming & ((intensity.ravel()[idx] >= SETTLED) if intensity is not None else True)
        tamed = state.cells["tamed"].ravel()[idx]
        cattle = (tamed & np.uint32(1 << self.cattle)) != 0 if self.cattle is not None else np.zeros(len(idx), bool)
        horse = (tamed & np.uint32(1 << self.horse)) != 0 if self.horse is not None else np.zeros(len(idx), bool)
        size = state.cells.get("settlement")
        if size is not None:
            r = max(0, int(round(cfg.tech_town_km / self.kpp)))
            towns = size >= cfg.town_min
            towns = (_box(towns.astype(np.uint8), r, np.maximum) > 0) if r else towns
            towns = towns.ravel()[idx]
        else:
            towns = np.zeros(len(idx), bool)
        own = state.cells.get("state")
        if own is not None:
            own = own.ravel()[idx]
            records = state.tables.get("states") or [None]
            big = np.array([r is not None and r["kind"] in ("kingdom", "empire") for r in records], bool)
            in_state, kingdom = own > 0, big[own]
        else:
            in_state = kingdom = np.zeros(len(idx), bool)
        smiths = (state.cells["tech"].ravel()[idx] & np.uint64(1 << INDEX["copper"])) != 0
        needs = {"people": np.ones(len(idx), bool), "farming": farming, "settled": settled, "smiths": smiths,
                 "herds": cattle | horse, "cattle": cattle, "horse": horse, "towns": towns,
                 "state": in_state, "kingdom": kingdom}
        needs.update({k: v[idx] for k, v in self.static.items()})
        trade = state.tables.get("trade")
        if trade is not None:                      # ores and metals that come by trade
            for k, v in trade.goods.items():
                needs[k] = needs[k] | v[idx]
        return needs

    def _allowed(self, needs, known, bits):
        """Bit mask per cell of the technologies whose needs (`bits`: need_bits, find_bits or
        keep_bits) and prerequisites are met."""
        names = [n for n in needs if bits[n]]
        met = np.stack([np.asarray(needs[n], bool) for n in names]) if names else np.zeros((0, len(known)), bool)
        masks = np.array([bits[n] for n in names], np.uint64)
        if not hasattr(self, "_pre_bits"):
            self._pre_bits = np.array([b for b, _ in self.prereq], np.uint64)
            self._pre_need = np.array([p for _, p in self.prereq], np.uint64)
        ok = np.empty(len(known), np.uint64)
        _allowed_kernel(met, masks, self._pre_bits, self._pre_need, np.ascontiguousarray(known), ok)
        return ok

    # ----------------------------------------------------------------- step
    def step(self, state):
        cfg = state.config
        dt = self.every
        rng = self.rng
        tech = state.cells["tech"].ravel()
        pop = state.cells["population"].ravel()
        tech[pop <= 0] = 0                                  # nobody left to remember
        idx = np.flatnonzero(pop > 0)
        if not len(idx):
            return
        needs = self._needs(state, idx)

        # Loss: knowledge whose keepers are gone fades.
        known = tech[idx]
        keep = self._allowed(needs, known, self.keep_bits)
        lost = (known & ~keep) != 0
        if lost.any():
            fade = 1 - 0.5 ** (dt / cfg.tech_loss_half_life)
            go = np.flatnonzero(lost)[rng.random(int(lost.sum())) < fade]
            known[go] &= keep[go]
            tech[idx] = known

        # Spread to neighbours that can use it.
        usable = self._allowed(needs, known, self.need_bits)
        ease = (1 + cfg.tech_corridor * needs["water"]) * (1 + cfg.tech_state_spread * needs["state"])
        rounds = rounds_for(self.scale)
        rate = cfg.tech_spread_km_year * dt / rounds / self.kpp
        trade = state.tables.get("trade")
        offer = trade.offer if trade is not None else None
        for _ in range(rounds):
            near = np.zeros_like(tech)
            for k in range(4):
                near |= tech[self.neighbours[k]]
            if offer is not None:
                near |= offer                      # what trading partners know
            cand = near[idx] & ~known & usable
            j = np.flatnonzero(cand)
            if not len(j):
                break
            cells = idx[j]
            around = tech[self.neighbours[:, cells]]                     # (4, cells)
            gained = np.zeros(len(j), np.uint64)
            for i in range(len(TECHS)):
                b = np.uint64(1 << i)
                sel = np.flatnonzero(cand[j] & b)
                if not len(sel):
                    continue
                count = ((around[:, sel] & b) != 0).sum(axis=0)
                chance = 1 - np.exp(-rate * count / 2 * ease[j[sel]])
                if offer is not None:              # by trade: a chance per step, by how busy
                    traded = (offer[cells[sel]] & b) != 0
                    w = trade.offer_weight[cells[sel]] * traded
                    chance = 1 - (1 - chance) * np.exp(-cfg.trade_tech * w * dt / rounds / 10)
                hit = sel[rng.random(len(sel)) < chance]
                gained[hit] |= b
            known[j] |= gained
            tech[idx] = known

        # Discovery: a chance in every cell where it could happen.
        r = max(1, int(round(cfg.tech_contact_km / self.kpp)))
        contact = _box(state.cells["population"], r, np.add).ravel()[idx] / 1e6
        base = pop[idx] / 1e6 * (1 + cfg.tech_town_boost * needs["towns"])
        usable = self._allowed(needs, known, self.find_bits)
        for i, t in enumerate(TECHS):
            b = np.uint64(1 << i)
            where = np.flatnonzero(usable & ~known & b)
            if not len(where):
                continue
            weight = base[where] * np.maximum(contact[where], 1e-9) ** t.crowd
            hazard = float(weight.sum()) / (t.rarity * cfg.tech_years)      # per year
            hazard = hazard / (1 + hazard * t.lag)          # never faster than its methods mature
            if rng.random() >= 1 - np.exp(-hazard * dt):
                continue
            pick = int(where[rng.choice(len(where), p=weight / weight.sum())])
            cell = int(idx[pick])
            knowers = np.flatnonzero(tech & b)
            if len(knowers) and great_circle_km(self.xyz[knowers], self.xyz[cell]).min() < cfg.tech_independent_km:
                continue                     # near others who know it: it would just spread
            known[pick] |= b
            tech[cell] |= b
            self._invent(state, t, cell)

        self._stats(state, idx, known)

    # ------------------------------------------------------------ chronicle
    def _invent(self, state, t, cell):
        w = state.shape[1]
        r, c = divmod(cell, w)
        world = state.world
        culture = int(state.cells["culture"].flat[cell]) if "culture" in state.cells else 0
        namer = state.tables.get("naming")
        lang = namer.language_of(state, culture) if namer is not None and culture else None
        local = lang.spell(lang.word(t.key), capital=False) if lang is not None else None
        cultures = state.tables.get("cultures") or [None]
        people = cultures[culture]["name"] if culture and cultures[culture] else None
        sid = int(state.cells["state"].flat[cell]) if "state" in state.cells else 0
        states = state.tables.get("states") or [None]
        realm = states[sid]["name"] if sid and states[sid] else None
        first = t.key not in self.first
        if first:
            self.first[t.key] = state.year
        rec = {"id": len(self.inventions), "tech": t.key, "name": t.name, "local_name": local,
               "culture": culture, "state": sid, "row": r, "col": c,
               "lat": round(float(world.grid.lat[r, c]), 2), "lon": round(float(world.grid.lon[r, c]), 2),
               "first": first, "founded": state.year, "ended": None}
        self.inventions.append(rec)
        who = f"The {people}" if people else "People"
        where = f" of {realm}" if realm else ""
        word = f" (in their tongue, {local})" if local else ""
        if first:
            text = f"{who}{where} are the first in the world to {t.verb}{word}."
        else:
            text = f"{who}{where} {t.verb}{word}, far from anyone who knew how."
        state.event("technology", text, tech=t.key, first=first, invention=rec["id"],
                    lat=rec["lat"], lon=rec["lon"])

    # ----------------------------------------------------------------- stats
    def _stats(self, state, idx, known):
        pop = state.cells["population"].ravel()
        total = max(float(pop.sum()), 1e-9)
        stats = {}
        if known is None:
            known = state.cells["tech"].ravel()[idx]
        people = pop[idx]
        for i, t in enumerate(TECHS):
            has = (known & np.uint64(1 << i)) != 0
            stats[f"know_{t.key}"] = float(people[has].sum()) / total
        count = np.zeros(len(known), np.int16)
        for i in range(len(TECHS)):
            count += ((known >> np.uint64(i)) & np.uint64(1)).astype(np.int16)
        stats["inventions"] = len(self.inventions)
        stats["technologies_most"] = int(count.max()) if len(count) else 0
        stats["technologies_mean"] = float((count * people).sum()) / total
        state.stats.update(stats)


@njit(cache=True, parallel=True)
def _allowed_kernel(met, masks, pre_bits, pre_need, known, ok):
    full = ~np.uint64(0)
    for c in prange(known.shape[0]):
        v = full
        for k in range(masks.shape[0]):
            if not met[k, c]:
                v &= ~masks[k]
        kn = known[c]
        for k in range(pre_bits.shape[0]):
            if (kn & pre_need[k]) != pre_need[k]:
                v &= ~pre_bits[k]
        ok[c] = v


def tech_names(mask):
    """English names of the technologies in one cell's mask."""
    return [t.name for i, t in enumerate(TECHS) if int(mask) >> i & 1]
