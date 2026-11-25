"""Cultures and languages: who people are, what they speak, and how that changes.

At year 0 the land people reached long ago is a patchwork of many small cultures (old
homelands are the most diverse), grouped into a few dozen language families; lands
reached late are more uniform. Each culture speaks its own language, a daughter of its
family's proto-language, and has a generated name.

Every step:
  arrivals      newly settled cells take the culture planned for them at year 0 (late
                settlement) or that of their strongest neighbour
  expansion     when farming or herding reaches foragers, the newcomers often bring their
                culture too (demic spread, like the Bantu or early Indo-European
                expansions); otherwise the locals adopt farming and keep their own
  assimilation  strong neighbours (dense, farming or herding, with towns) slowly absorb
                weaker ones; small enclaves surrounded by one culture are absorbed fastest

Every `culture_check_years`:
  splits        a culture spread too wide for its people to stay in touch splits in two
                (along natural barriers); a piece cut off by sea or other peoples
                becomes its own culture after a while. The new culture's language is a
                daughter of the old one: regular sound changes and some new words.
  extinction    cultures left with no people end
"""
import colorsys

import numpy as np
from numba import njit, prange
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from worldgen.generator import distance_to

from .geo import shift
from worldgen.noise import Perlin3, fbm

from ..core.rng import draws_at, hits
from ..core.simulation import System
from ..names import Language
from .geo import (DIRECTIONS4, cell_scale, cells_for, rounds_for, ShelfSource, cell_area_km2, cell_graph, great_circle_km, land_now,
                  nearest_seed, spaced_sample, subgraph)
from .population import FARMING, FORAGING, HERDING, TRAVEL, _by_biome

MAX_CULTURES = 65_000          # the culture layer is uint16


class Cultures(System):
    name = "cultures"

    def __init__(self):
        self.every = None

    # ----------------------------------------------------------------- setup
    def setup(self, state):
        cfg, world = state.config, state.world
        self.every = cfg.step_years
        self.scale = cell_scale(state.world, cfg)
        self.rng = state.rng.get("cultures")
        self.name_rng = state.rng.get("names")
        h, w = world.ocean.shape
        self.n = h * w
        self.area = cell_area_km2(world)
        self.km_per_px = world.grid.km_per_px
        self.xyz = np.stack(world.grid.xyz, axis=-1).reshape(-1, 3)
        # Travel effort between cells: the biome, slope and some local noise (so borders
        # follow ridges and rough country rather than straight lines).
        shelf = ShelfSource(world)
        travel = shelf.fill(_by_biome(world.biome, TRAVEL, 1.5), world)
        slope = shelf.fill(world.slope_m_per_km.astype(np.float32), world)
        noise = np.exp(1.1 * fbm(Perlin3(world.cfg.seed * 7919 + state.seed * 131 + 4242), world.grid.xyz, 9, 5))
        cost = (travel * (1 + slope / 10) * noise).astype(np.float64)
        ever_land = land_now(world, cfg.glacial_sea_level_m) | (~world.ocean & ~world.lake)
        self.graph, self.edges = cell_graph(world, cost, ever_land)

        state.add_layer("culture", np.uint16, record="changes")
        state.tables["cultures"] = [None]       # id 0 = nobody
        state.tables["languages"] = self.language_table = []
        state.saved_tables.update({"cultures", "languages"})
        state.tables["naming"] = self           # other systems name things through the cultures
        self.langs = []                         # Language objects, by language id
        self.used_names = set()
        self._prev_mode = state.cells["subsistence"].copy()
        self._initial_cultures(state)
        self._stats(state)

    def _initial_cultures(self, state):
        cfg = state.config
        arrival = state.cells["arrival_year"].ravel()
        reach = np.isfinite(arrival)
        pop = state.cells["population"].ravel()
        area = self.area.ravel()
        forage = state.cells["forager_capacity"].ravel()
        span = max(state.year - cfg.out_of_homeland_year, 1)
        depth = np.where(reach, np.clip((state.year - np.where(reach, arrival, 0)) / span, 0, 1), 0)
        area_mkm2 = float(area[reach].sum()) / 1e6

        # Language families: a few large regions, more of them in long-settled lands.
        n_fam = max(1, round(area_mkm2 * cfg.families_per_mkm2))
        fam_seeds = spaced_sample(self.rng, np.where(reach, area * (0.3 + depth), 0), n_fam,
                                  self.xyz, cfg.family_min_km)
        family = nearest_seed(self.graph, fam_seeds, self.n)
        family = np.where(reach, family, -1)
        family, fam_seeds = self._cover_unreached(family, reach, fam_seeds, pop, join_km=600)

        # Cultures: many more, small where people are many and settlement is old.
        n_cul = max(len(fam_seeds), round(area_mkm2 * cfg.cultures_per_mkm2))
        dens = forage / np.maximum(area, 1e-6)
        rel = np.sqrt(dens / max(np.median(dens[reach]) if reach.any() else 1, 1e-9))
        cul_seeds = spaced_sample(self.rng, np.where(reach, area * (0.4 + 1.6 * depth) * rel, 0), n_cul,
                                  self.xyz, cfg.culture_seed_min_km)
        cul_seeds = list(dict.fromkeys(fam_seeds + cul_seeds))   # every family has at least one
        same_family = subgraph(self.graph, self.edges, lambda a, b: (family[a] == family[b]) & (family[a] >= 0))
        label = nearest_seed(same_family, cul_seeds, self.n)
        label = np.where(reach, label, -1)
        label, cul_seeds = self._cover_unreached(label, reach, cul_seeds, pop, join_km=0)

        # Name them: one proto-language per family, each culture speaks a daughter of it.
        protos = {}
        planned = np.zeros(self.n, np.int32)
        fam_hues = {}
        for i, seed_cell in enumerate(cul_seeds):
            f = int(family[seed_cell])
            if f not in protos:
                protos[f] = self._new_language(None)
                fam_hues[f] = (len(fam_hues) * 0.618034 + self.rng.random() * 0.1) % 1
            cid = self._new_culture(state, protos[f], seed_cell, parent=None,
                                    hue=fam_hues[f] + self.rng.uniform(-0.06, 0.06))
            planned[label == i] = cid
        # Family names come from their most populous culture.
        tally = np.bincount(planned, weights=pop, minlength=len(state.tables["cultures"]))
        for f, proto in protos.items():
            members = [c for c in state.tables["cultures"][1:] if c["family"] == proto.id]
            best = max(members, key=lambda c: tally[c["id"]])
            proto.name = best["name"]
            state.tables["languages"][proto.id]["name"] = f"Proto-{best['name']}"
            state.tables["languages"][proto.id]["family_name"] = f"{best['name']} languages"
        self.planned = planned.reshape(state.shape)
        state.cells["culture"][:] = np.where(state.cells["population"] > 0, self.planned, 0)

    def _cover_unreached(self, label, reach, seeds, pop, join_km):
        """Reached cells no seed could reach overland (islands): join a nearby region, or
        found a new one in each piece."""
        missing = reach & (label < 0)
        if not missing.any():
            return label, seeds
        shape = self.area.shape
        if join_km > 0:
            dist, iy, ix = distance_to((label >= 0).reshape(shape))
            near = (dist.ravel() * self._kpp() <= join_km) & missing
            src = (iy * shape[1] + ix).ravel()
            label = label.copy()
            label[near] = label[src[near]]
            missing = reach & (label < 0)
        if missing.any():
            only = subgraph(self.graph, self.edges, lambda a, b: missing[a] & missing[b])
            n_comp, comp = connected_components(only, directed=False)
            seeds = list(seeds)
            for k in np.unique(comp[missing]):
                cells = np.flatnonzero(missing & (comp == k))
                seed_cell = int(cells[np.argmax(pop[cells] + 1e-3)])
                label[cells] = len(seeds)
                seeds.append(seed_cell)
        return label, seeds

    def _kpp(self):
        return self.km_per_px

    # -------------------------------------------------------------- entities
    def _new_language(self, parent):
        """A new language (a proto-language if no parent), recorded in the languages table
        (row = language id)."""
        lang = Language(len(self.langs), seed=int(self.rng.integers(2 ** 31)), parent=parent, rng=self.rng)
        self.langs.append(lang)
        self.language_table.append(lang.to_dict())
        return lang

    def _new_culture(self, state, family_lang, seed_cell, parent, hue=None, colour=None, year=None):
        cultures = state.tables["cultures"]
        if len(cultures) >= MAX_CULTURES:
            return None
        lang = self._new_language(family_lang)
        name = self._unique(lambda: lang.people_name(self.name_rng))
        lang.name = name
        self.language_table[lang.id]["name"] = name
        if colour is None:
            colour = colorsys.hsv_to_rgb(hue % 1, self.rng.uniform(0.45, 0.85), self.rng.uniform(0.62, 0.95))
        r, c = divmod(int(seed_cell), self.area.shape[1])
        cid = len(cultures)
        cultures.append({
            "id": cid, "name": name, "language": lang.id, "family": lang.family,
            "parent": parent, "founded": state.year if year is None else year, "ended": None,
            "colour": [round(float(v), 3) for v in colour], "origin": [r, c],
            "peak_population": 0.0,
        })
        return cid

    def _unique(self, make):
        for _ in range(30):
            name = make()
            if name not in self.used_names:
                break
        self.used_names.add(name)
        return name

    def language_of(self, state, culture_id):
        """The Language object a culture speaks (for naming places)."""
        rec = state.tables["cultures"][int(culture_id)]
        return self.langs[rec["language"]] if rec else None

    # ----------------------------------------------------------------- step
    def step(self, state):
        cfg = state.config
        dt = self.every
        rng = self.rng
        cult = state.cells["culture"]
        pop = state.cells["population"]
        mode = state.cells["subsistence"]
        people = pop > 0
        cult[~people] = 0

        strength = self._strength(state)
        # Newly settled cells.
        new = people & (cult == 0)
        if new.any():
            best_c, _, _ = _strongest_neighbour(cult, strength, new)
            take = np.where(self.planned > 0, self.planned, best_c)
            cult[new] = take[new]

        # Farming or herding arriving among foragers: often the newcomers' culture too.
        converted = (self._prev_mode == FORAGING) & (mode >= HERDING)
        if converted.any():
            farmer_strength = np.where(mode >= HERDING, strength, 0)
            best_c, _, _ = _strongest_neighbour(cult, farmer_strength, converted)
            take = hits(rng, converted & (best_c > 0), cfg.farmers_replace)
            cult[take] = best_c[take]
        self._prev_mode = mode.copy()

        # Assimilation by stronger neighbours; enclaves absorbed (fronts move as many km at
        # any map size: more rounds on fine maps).
        rounds = rounds_for(self.scale)
        for _ in range(rounds):
            best_c, best_s, count = _strongest_neighbour(cult, strength, people, different=True)
            # Pressure to give way: how much stronger the strongest neighbouring culture is,
            # plus being surrounded by it (compiled, only where it can happen).
            chance = np.zeros(cult.shape)
            _assimilation_chance(best_c.ravel(), best_s.ravel(), strength.ravel(), count.ravel(),
                                 people.ravel(), cfg.assimilation_rate, cfg.cohesion_rate,
                                 self.scale * rounds, dt, chance.ravel())
            flip = hits(rng, people & (best_c > 0), chance)
            cult[flip] = best_c[flip]

        if (state.year - getattr(self, "_last_check", -10 ** 9)) >= cfg.culture_check_years:
            self._last_check = state.year
            self._splits(state)
        self._stats(state)

    def _strength(self, state):
        """How strongly each cell's culture holds and spreads: people per km2, raised for
        herders and farmers (organisation, prestige) and for towns."""
        cfg = state.config
        pop = state.cells["population"]
        mode = state.cells["subsistence"]
        prestige = np.array([0, cfg.forager_prestige, cfg.herder_prestige, cfg.farmer_prestige], np.float32)
        town = state.cells.get("settlement")
        polity = state.tables.get("polity")
        own = state.cells.get("state")
        ruled = polity is not None and own is not None and len(polity.ruler_culture) > int(own.max())
        out = np.empty(pop.shape)
        _strength_kernel(pop.ravel(), np.ascontiguousarray(mode).ravel(), self.area.ravel(), prestige,
                         town.ravel() if town is not None else np.zeros(1, np.float32), town is not None,
                         own.ravel() if ruled else np.zeros(1, np.uint16),
                         polity.ruler_culture if ruled else np.zeros(1, np.int64), ruled,
                         state.cells["culture"].ravel(), cfg.ruling_prestige, out.ravel())
        return out

    # ---------------------------------------------------------------- splits
    def _splits(self, state):
        cfg = state.config
        cult = state.cells["culture"].ravel()
        pop = state.cells["population"].ravel()
        mode = state.cells["subsistence"].ravel()
        records = state.tables["cultures"]
        n_ids = len(records)
        alive_cells = cult > 0
        counts = np.bincount(cult, minlength=n_ids)
        people = np.bincount(cult, weights=pop, minlength=n_ids)

        # Endings, and peaks.
        for cid in range(1, n_ids):
            rec = records[cid]
            if rec["ended"] is None:
                if counts[cid] == 0 and rec.get("_seen"):
                    rec["ended"] = state.year
                    if rec["peak_population"] > 50_000:
                        state.event("culture", f"The {rec['name']} are no more: their last communities "
                                               f"have merged into neighbouring peoples.", culture=cid)
                elif counts[cid] > 0:
                    rec["_seen"] = True
                    rec["peak_population"] = max(rec["peak_population"], round(float(people[cid])))

        # Too wide to hold together: split along the easiest line between core and edge.
        w = np.where(alive_cells, pop + 1e-3, 0)
        centre = np.zeros((n_ids, 3))
        for k in range(3):
            centre[:, k] = np.bincount(cult, weights=w * self.xyz[:, k], minlength=n_ids)
        norm = np.linalg.norm(centre, axis=1, keepdims=True)
        centre = centre / np.maximum(norm, 1e-12)
        idx = np.flatnonzero(alive_cells)
        dist = great_circle_km(self.xyz[idx], centre[cult[idx]])
        reach = np.zeros(n_ids)
        np.maximum.at(reach, cult[idx], dist)
        herders = np.bincount(cult, weights=np.where(mode == HERDING, pop, 0), minlength=n_ids)
        limit = cfg.culture_radius_km * (1 + herders / np.maximum(people, 1e-9))
        p_split = 1 - np.exp(-cfg.culture_check_years / cfg.culture_split_years)
        wide = np.flatnonzero((reach > limit) & (counts >= cells_for(6, self.scale)))
        for cid in wide:
            if self.rng.random() < p_split:
                cells = idx[cult[idx] == cid]
                d = dist[cult[idx] == cid]
                self._split_wide(state, int(cid), cells, d, pop)

        # Pieces cut off from the rest of their people.
        cult = state.cells["culture"].ravel()
        a, b = self.edges
        same = (cult[a] == cult[b]) & (cult[a] > 0)
        g = coo_matrix((np.ones(same.sum()), (a[same], b[same])), shape=(self.n, self.n))
        _, comp = connected_components(g, directed=False)
        idx = np.flatnonzero(cult > 0)
        if not len(idx):
            return
        key = cult[idx].astype(np.int64) * (self.n + 1) + comp[idx]
        uniq, inverse, size = np.unique(key, return_inverse=True, return_counts=True)
        piece_cult = uniq // (self.n + 1)
        # The biggest piece of each culture is its homeland; others may break away.
        order = np.lexsort((-size, piece_cult))
        main = np.zeros(len(uniq), bool)
        first = np.ones(len(order), bool)
        first[1:] = piece_cult[order][1:] != piece_cult[order][:-1]
        main[order[first]] = True
        p_iso = 1 - np.exp(-cfg.culture_check_years / cfg.isolation_split_years)
        for k in np.flatnonzero(~main & (size >= cells_for(cfg.min_culture_cells, self.scale))):
            if self.rng.random() < p_iso:
                cells = idx[inverse == k]
                self._break_away(state, int(piece_cult[k]), cells, pop)

    def _split_wide(self, state, cid, cells, dist, pop):
        core = cells[np.argmin(dist)]
        far_pool = cells[dist >= 0.75 * dist.max()]
        weights = pop[far_pool] + 1e-3
        far = int(self.rng.choice(far_pool, p=weights / weights.sum()))
        sub = self.graph[cells][:, cells]
        local = {int(c): i for i, c in enumerate(cells)}
        _, _, src = dijkstra(sub, directed=False, indices=[local[int(core)], local[far]],
                             min_only=True, return_predecessors=True)
        new_cells = cells[src == local[far]]
        if 0 < len(new_cells) < len(cells):
            self._break_away(state, cid, new_cells, pop, seed_cell=far, reason="spread")

    def _break_away(self, state, cid, cells, pop, seed_cell=None, reason="cut_off"):
        records = state.tables["cultures"]
        parent = records[cid]
        lang = self.langs[parent["language"]]
        if seed_cell is None:
            seed_cell = int(cells[np.argmax(pop[cells])])
        h, s, v = colorsys.rgb_to_hsv(*parent["colour"])
        colour = colorsys.hsv_to_rgb((h + self.rng.uniform(-0.05, 0.05)) % 1,
                                     np.clip(s + self.rng.uniform(-0.2, 0.2), 0.35, 0.95),
                                     np.clip(v + self.rng.uniform(-0.2, 0.2), 0.5, 0.98))
        new = self._new_culture(state, lang, seed_cell, parent=cid, colour=colour)
        if new is None:
            return
        state.cells["culture"].ravel()[cells] = new
        people = float(pop[cells].sum())
        records[new]["peak_population"] = round(people)
        if people > 30_000:
            r, c = divmod(int(seed_cell), self.area.shape[1])
            lat, lon = float(state.world.grid.lat[r, c]), float(state.world.grid.lon[r, c])
            why = "have grown apart from" if reason == "spread" else "long cut off, have become a people apart from"
            state.event("culture", f"The {records[new]['name']} {why} the {parent['name']}; "
                                   f"their speech is now a language of its own.", culture=new, parent=cid,
                        lat=lat, lon=lon)

    # ----------------------------------------------------------------- stats
    def _stats(self, state):
        cult = state.cells["culture"]
        pop = state.cells["population"]
        records = state.tables["cultures"]
        people = np.bincount(cult.ravel(), weights=pop.ravel(), minlength=len(records))
        alive = np.flatnonzero(people[1:] > 0) + 1
        families = {records[c]["family"] for c in alive}
        total = max(float(pop.sum()), 1e-9)
        state.stats.update({
            "cultures": len(alive),
            "language_families": len(families),
            "largest_culture_share": float(people[1:].max() / total) if len(alive) else 0.0,
        })


@njit(cache=True, parallel=True)
def _strength_kernel(pop, mode, area, prestige, town, has_town, own, ruler_culture, ruled, culture,
                     ruling_prestige, out):
    """People per km2, raised for herders and farmers (organisation, prestige), for towns,
    and for the rulers' own culture in their state."""
    for i in prange(pop.shape[0]):
        v = pop[i] / max(area[i], 1e-6) * prestige[mode[i]]
        if has_town:
            v *= 1 + np.log1p(town[i] / 1000)
        if ruled and own[i] > 0 and culture[i] == ruler_culture[own[i]]:
            v *= 1 + ruling_prestige
        out[i] = v


@njit(cache=True, parallel=True)
def _assimilation_chance(best_c, best_s, strength, count, people, assimilation_rate, cohesion_rate, divide,
                         dt, out):
    for i in prange(best_c.shape[0]):
        if not people[i] or best_c[i] == 0:
            continue
        ratio = best_s[i] / max(strength[i], 1e-9)
        pressure = assimilation_rate * min(max(np.log(max(ratio, 1.0)), 0.0), 3.0)
        if count[i] >= 3:
            pressure += cohesion_rate
        pressure /= divide
        out[i] = 1 - np.exp(-pressure * dt)


def _strongest_neighbour(cult, strength, where, different=False):
    """For each cell: the culture of its strongest neighbour (optionally only neighbours of
    another culture), that neighbour's strength, and how many neighbours share that culture.
    (Compiled; the same results as comparing the four shifted maps in turn.)"""
    from .geo import neighbour_index
    key = cult.shape
    if key not in _NEIGHBOURS:
        _NEIGHBOURS[key] = np.ascontiguousarray(neighbour_index(cult.shape))
    strength = np.ascontiguousarray(strength)
    best_s = np.zeros(cult.shape, np.result_type(np.float32, strength.dtype))
    best_c = np.zeros_like(cult)
    count = np.zeros(cult.shape, np.int8)
    _strongest_kernel(cult.ravel(), strength.ravel(), np.ascontiguousarray(where).ravel(), different,
                      _NEIGHBOURS[key], best_c.ravel(), best_s.ravel(), count.ravel())
    return best_c, best_s, count


_NEIGHBOURS = {}


@njit(cache=True)
def _strongest_kernel(cult, strength, where, different, nbr, best_c, best_s, count):
    n = cult.shape[0]
    for i in range(n):
        if not where[i]:
            continue
        for k in range(4):
            j = nbr[k, i]
            nc = cult[j]
            ns = strength[j]
            if nc > 0 and ns > best_s[i] and (not different or nc != cult[i]):
                best_c[i] = nc
                best_s[i] = ns
        if best_c[i] > 0:
            c = 0
            for k in range(4):
                if cult[nbr[k, i]] == best_c[i]:
                    c += 1
            count[i] = c
