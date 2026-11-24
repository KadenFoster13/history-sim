"""States: chiefdoms, city-states, kingdoms and empires; territory, war, peace and breakup.

Formation   where towns have grown among farmers, a chief may rise and found a chiefdom
            around one; dense herders may form a confederation of their own. Farmland
            hemmed in by desert, mountains or sea forms states sooner and more often
            (Carneiro's circumscription: losers have nowhere to flee, so wars end in
            subjects and tribute, not migration - the Nile, Mesopotamia, coastal Peru).
Territory   a state's grip on a cell is its strength (people, weighted for herders'
            horsemanship, times its cohesion) fading with travel distance from the
            capital: rough country, forests and mountains cost more distance than open
            plains, so borders follow the land. Strong states take in stateless villages
            nearby; how far they reach grows with their size.
War         rulers decide (see agents/) to attack a neighbour when clearly stronger along
            the shared border, more readily when it is busy elsewhere. Along the front each
            cell may change hands, depending on both sides' local strength and on defence:
            hills, rivers, towns and capitals. Conquest costs lives. Wars end in peace (and a
            truce of some decades), when one side loses its capital (a small state is
            annexed, a large one moves its court), or when the fighting has no front left.
            Allies join defensive wars.
Cohesion    (Turchin's "asabiya") grows in states facing other peoples on long frontiers
            and fades in comfortable cores. Young frontier states rise; old empires, mostly
            interior, lose cohesion, and with it the strength to hold distant provinces.
Breakup     distant regions, especially of other cultures, may rebel and become
            independent; the weaker the state's cohesion, the more readily.
Kinds       chiefdom, city-state (a single city and its fields), kingdom, empire (millions
            of people or many peoples); named in the rulers' language.
"""
import colorsys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from worldgen.generator import shift
from worldgen.hydrology import cell_river_flow

from ..agents import Neighbour, Situation, StateAI
from ..core.simulation import System
from .geo import (DIRECTIONS4, ShelfSource, cell_area_km2, cell_graph, cell_scale, cells_for, rounds_for,
                  great_circle_km, land_now)
from .population import FARMING, FORAGING, HERDING, TRAVEL, _by_biome

TITLES = {"chiefdom": "Chiefdom of {}", "city-state": "City of {}", "kingdom": "Kingdom of {}",
          "empire": "{} Empire"}
DIRECTIONS8 = DIRECTIONS4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))
MAX_STATES = 65_000          # the state layer is uint16
NOTABLE = 300_000            # people: wars and events involving states this big are chronicled


class States(System):
    name = "states"

    def __init__(self):
        self.every = None

    # ----------------------------------------------------------------- setup
    def setup(self, state):
        cfg, world = state.config, state.world
        self.every = cfg.step_years
        self.scale = cell_scale(world, cfg)
        self.rng = state.rng.get("states")
        self.name_rng = state.rng.get("names")
        h, w = world.ocean.shape
        self.n = h * w
        self.w = w
        self.area = cell_area_km2(world)
        self.xyz = np.stack(world.grid.xyz, axis=-1).reshape(-1, 3)
        shelf = ShelfSource(world)
        travel = shelf.fill(_by_biome(world.biome, TRAVEL, 1.5), world)
        slope = shelf.fill(world.slope_m_per_km.astype(np.float32), world)
        self.cost = (travel * (1 + slope / 40)).astype(np.float64)   # ~1 on open plains, 2-5 in rough country
        ever_land = land_now(world, cfg.glacial_sea_level_m) | (~world.ocean & ~world.lake)
        self.graph, self.edges = cell_graph(world, self.cost, ever_land)
        self.circumscribed = self._circumscription(state)
        river = (cell_river_flow(world) > 1000) & ~world.ocean
        self.defence = (cfg.hill_defence * np.clip(slope / 30, 0, 1.5) + cfg.river_defence * river).astype(np.float32)
        # Hills cover a whole cell; a river line, walls or a capital only a ~50 km core of it.
        self.hill_defence = (cfg.hill_defence * np.clip(slope / 30, 0, 1.5)).astype(np.float32)
        self.river_defence = (cfg.river_defence * river).astype(np.float32)

        state.add_layer("state", np.uint16, record="changes")
        state.add_layer("state_distance", np.float32, fill=np.inf, record="none")
        state.tables["states"] = self.records = [None]
        state.tables["wars"] = []
        state.saved_tables.update({"states", "wars"})
        state.tables["polity"] = self
        self.ai = StateAI(cfg, self.rng)
        from worldgen import SPECIES
        self.horse = next((i for i, sp in enumerate(SPECIES) if sp.key == "horse"), None)
        self.wars = {}               # war id -> record (active wars)
        self.pair_war = {}           # (a, b) with a < b -> war id
        self.truces = {}             # (a, b) -> year the truce ends
        self.alliances = {}          # (a, b) -> year the alliance ends
        self.cohesion = np.zeros(1)
        self.aggression = np.zeros(1)
        self.ruler_culture = np.zeros(1, np.int64)
        self._last_admin = self._last_decide = -10 ** 9
        self._first_kingdom = self._first_empire = False
        self._largest, self._largest_year = None, -10 ** 9
        self._deaths = 0.0
        self._world_people = float(state.cells["population"].sum())
        self._notable_share = cfg.notable_share
        self._summary(state)
        self._stats(state)

    # ------------------------------------------------------------- summaries
    def _summary(self, state):
        """Per-state totals and each cell's local strength (for this step)."""
        cfg = state.config
        own = state.cells["state"]
        pop = state.cells["population"]
        mode = state.cells["subsistence"]
        n = len(state.tables["states"])
        flat = own.ravel()
        self.people = np.bincount(flat, weights=pop.ravel(), minlength=n)
        self.cell_count = np.bincount(flat, minlength=n)
        military = pop * (1 + cfg.herder_military * ((mode == HERDING) & self._horses(state)))
        self.military = np.bincount(flat, weights=military.ravel(), minlength=n)
        self.power = self.military * self.cohesion[:n]
        self.reach = cfg.reach_km * (1 + cfg.reach_growth * np.log10(1 + self.people / 1e5))
        chief = np.array([r is not None and r["kind"] == "chiefdom" for r in self.records[:n]])
        self.reach[:len(chief)][chief] *= cfg.chiefdom_reach         # chiefs rule only nearby
        dist = state.cells["state_distance"]
        if self.scale != 1:
            # Measured to cell centres, which lie half a cell inside a border: on coarse maps
            # 50 km in, on the 800-px map 25 km. Strength along a border is as on that map.
            dist = np.maximum(dist + 25.0 * (self.scale - 1), 0)
        decay = np.exp(-np.where(np.isfinite(dist), dist, 1e9) / self.reach[own])
        self.local = np.where(own > 0, np.sqrt(self.power[own]) * decay, 0).astype(np.float32)

    def _circumscription(self, state):
        """0..1: good farmland whose surroundings (~200 km) are mostly poor land or sea."""
        from worldgen.generator import blur
        world, cfg = state.world, state.config
        farming = state.tables.get("farming")
        arable = getattr(farming, "arable_eff", None)
        if arable is None:
            arable = world.arable
        good = (arable >= cfg.circumscription_arable) & ~world.ocean & ~world.lake
        share = blur(good.astype(np.float32), cfg.circumscription_km / world.grid.km_per_px)
        return np.where(good, np.clip((0.75 - share) / 0.5, 0, 1), 0).astype(np.float32)

    # ----------------------------------------------------------------- step
    def step(self, state):
        cfg = state.config
        own = state.cells["state"]
        pop = state.cells["population"]
        own[pop <= 0] = 0
        self._deaths = 0.0
        self._world_people = float(pop.sum())
        self._notable_share = cfg.notable_share
        self._summary(state)
        self._update_cohesion(state)
        self._form(state)
        rounds = rounds_for(self.scale)         # fronts cross more cells on fine maps
        for k in range(rounds):
            if k:
                self._summary(state)
            self._absorb(state, rounds)
            self._fight(state, rounds)
        if state.year - self._last_decide >= cfg.decision_years:
            self._last_decide = state.year
            self._decide(state)
        self._end_wars(state)
        if state.year - self._last_admin >= cfg.admin_years:
            self._last_admin = state.year
            self._admin(state)
        self._stats(state)

    # ------------------------------------------------------------- cohesion
    def _update_cohesion(self, state):
        """Cohesion grows where a state faces other peoples, fades in its quiet core."""
        cfg = state.config
        dt = self.every
        own = state.cells["state"]
        cult = state.cells.get("culture")
        records = state.tables["states"]
        n = len(records)
        if n <= 1 or cult is None:
            return
        families = self._family_lookup(state)
        fam = families[cult]
        frontier = np.zeros(own.shape, bool)    # facing peoples of another language family
        for dy, dx in DIRECTIONS4:
            nf = shift(fam, dy, dx)
            frontier |= (nf != fam) & (nf > 0)
        for _ in range(max(0, int(round(1 / self.scale)) - 1)):   # a ~50 km band on maps of any size
            wider = frontier.copy()
            for dy, dx in DIRECTIONS4:
                wider |= shift(frontier, dy, dx)
            frontier = wider
        pop = state.cells["population"]
        front_people = np.bincount(own.ravel(), weights=(pop * frontier).ravel(), minlength=n)
        f = np.minimum(front_people / np.maximum(self.people, 1e-9) / max(self.scale, 1), 1.0)   # coarse cells: part of a cell
        a = self.cohesion[:n]
        # Big states also overextend: elites multiply and quarrel, provinces drift apart.
        overextension = 1 + self.people / cfg.overextension_people
        da = dt * (cfg.cohesion_gain * f * a * (1 - a) - cfg.cohesion_loss * overextension * (1 - f) * a)
        self.cohesion[:n] = np.clip(a + da, 0.02, 1.0)

    def _horses(self, state):
        """Cells whose people keep horses (riders: herders become formidable)."""
        farming = state.tables.get("farming")
        if farming is None or self.horse is None:
            return np.zeros(state.shape, bool)
        return farming.has(state, self.horse)

    def _family_lookup(self, state):
        cultures = state.tables.get("cultures") or [None]
        if getattr(self, "_fam_n", 0) != len(cultures):
            self._fam = np.array([c["family"] + 1 if c else 0 for c in cultures], np.int64)
            self._fam_n = len(cultures)
        return self._fam

    # ------------------------------------------------------------- formation
    def _form(self, state):
        cfg = state.config
        dt = self.every
        own = state.cells["state"]
        pop = state.cells["population"]
        mode = state.cells["subsistence"]
        size = state.cells.get("settlement")
        rng = self.rng
        since = state.cells.get("farming_since")
        circ = self.circumscribed
        delay = cfg.state_delay_years * (1 - cfg.circumscription_sooner * circ)
        old_farming = (since > -(2 ** 31)) & (state.year - since >= delay) if since is not None else True
        if size is not None:
            towns = (size >= cfg.state_town_min) & (own == 0) & old_farming
            if towns.any():
                chance = 1 - np.exp(-cfg.state_formation_rate * dt * np.clip(size / 5000, 0.5, 3)
                                    * (1 + cfg.circumscription_states * circ))
                for cell in np.flatnonzero(towns & (rng.random(own.shape) < chance)):
                    if own.flat[cell] == 0:
                        self._found(state, int(cell), radius=max(1, round(1 / self.scale)))   # ~150 km across
        density = pop / np.maximum(self.area, 1e-6)
        herders = (mode == HERDING) & (own == 0) & (density >= cfg.herder_state_density) & self._horses(state)
        if herders.any() and rng.random() < 1 - np.exp(-cfg.herder_state_rate * dt):
            cells = np.flatnonzero(herders)
            cell = int(rng.choice(cells, p=pop.flat[cells] / pop.flat[cells].sum()))
            sid = self._found(state, cell, radius=max(1, round(2 / self.scale)))
            if sid is not None:
                self.records[sid]["nomadic"] = True

    def _found(self, state, cell, radius, culture=None, parent=None, cells=None, cohesion=None):
        """A new state with its capital at `cell`, holding the stateless people around it
        (or exactly `cells`)."""
        records = state.tables["states"]
        if len(records) >= MAX_STATES:
            return None
        own = state.cells["state"]
        pop = state.cells["population"]
        r, c = divmod(cell, self.w)
        if cells is None:
            rows = slice(max(r - radius, 0), r + radius + 1)
            cols = [(c + k) % self.w for k in range(-radius, radius + 1)]
            block = np.zeros(own.shape, bool)
            block[rows, cols] = True
            if self.scale > 1:      # coarse cells: a plus, not a square three cells wide
                dr = np.abs(np.arange(own.shape[0]) - r)[:, None]
                dc = np.abs((np.arange(self.w) - c + self.w // 2) % self.w - self.w // 2)[None, :]
                block &= dr + dc <= radius
            cells = np.flatnonzero(block & (own == 0) & (pop > 0))
        sid = len(records)
        if culture is None:
            cult = state.cells.get("culture")
            culture = int(cult.flat[cell]) if cult is not None else 0
        name, capital_name = self._name(state, culture, cell)
        h, s, v = (self.rng.random(), self.rng.uniform(0.5, 0.85), self.rng.uniform(0.6, 0.92))
        if parent is not None and records[parent]:
            ph, ps, pv = colorsys.rgb_to_hsv(*records[parent]["colour"])
            h = (ph + self.rng.uniform(-0.12, 0.12)) % 1
        records.append({
            "id": sid, "name": name, "kind": "chiefdom", "title": TITLES["chiefdom"].format(name),
            "culture": culture, "capital": int(cell), "capital_name": capital_name,
            "founded": state.year, "ended": None, "fate": None, "parent": parent,
            "colour": [round(float(x), 3) for x in colorsys.hsv_to_rgb(h, s, v)],
            "peak_people": 0.0, "peak_cells": 0, "peak_year": state.year,
        })
        self._grow_arrays(len(records))
        self.cohesion[sid] = cohesion if cohesion is not None else self.rng.uniform(0.5, 0.8)
        self.aggression[sid] = float(np.exp(self.rng.normal(0, 0.45)))
        self.ruler_culture[sid] = culture
        own.flat[cells] = sid
        own.flat[cell] = sid if pop.flat[cell] > 0 else own.flat[cell]
        self._distances(state, only=sid)
        return sid

    def _grow_arrays(self, n):
        if len(self.cohesion) < n:
            extra = n - len(self.cohesion) + 64
            self.cohesion = np.concatenate([self.cohesion, np.zeros(extra)])
            self.aggression = np.concatenate([self.aggression, np.ones(extra)])
            self.ruler_culture = np.concatenate([self.ruler_culture, np.zeros(extra, np.int64)])

    def _name(self, state, culture, cell):
        """A short name for a new state, in its rulers' language, and its capital's name."""
        places = state.tables.get("settlements") or []
        ids = state.cells.get("settlement_id")
        capital_name = None
        if ids is not None and 0 < ids.flat[cell] < len(places) and places[ids.flat[cell]]:
            capital_name = places[ids.flat[cell]]["name"]
        namer = state.tables.get("naming")
        lang = namer.language_of(state, culture) if namer is not None and culture else None
        cultures = state.tables.get("cultures") or []
        taken = {s["name"] for s in state.tables["states"][1:] if s}
        rng = self.name_rng
        for attempt in range(12):
            pick = rng.random()
            if capital_name and pick < 0.5 and attempt == 0:
                name = capital_name
            elif culture and pick < 0.75 and attempt < 2 and culture < len(cultures) and cultures[culture]:
                name = cultures[culture]["name"]
            elif lang is not None:
                name = lang.place_name(rng, ("land", "plain", "river", "hill"))
            else:
                name = f"State {len(state.tables['states'])}"
            if name not in taken:
                break
        if capital_name is None:
            capital_name = name
        return name, capital_name

    # ------------------------------------------------------------ absorption
    def _absorb(self, state, rounds=1):
        """Strong states take in stateless villages next to them."""
        cfg = state.config
        dt = self.every
        own = state.cells["state"]
        pop = state.cells["population"]
        mode = state.cells["subsistence"]
        free = (own == 0) & (pop > 0)
        if not free.any() or not (own > 0).any():
            return
        best_owner = np.zeros_like(own)
        best_local = np.zeros(own.shape, np.float32)
        for dy, dx in DIRECTIONS4:
            no, nl = shift(own, dy, dx), shift(self.local, dy, dx)
            better = free & (no > 0) & (nl > best_local)
            best_owner = np.where(better, no, best_owner)
            best_local = np.where(better, nl, best_local)
        # Resistance from how many people live there per km2 (the same at any map resolution).
        density = pop / np.maximum(self.area, 1e-6)
        ratio = best_local / (cfg.absorb_resistance * np.sqrt(np.maximum(density, 1e-3) * 2500))
        way = np.array([0, cfg.absorb_foragers, cfg.absorb_herders, 1.0], np.float32)[mode]
        # Borders fill in rather than reach out: a village half surrounded by a state joins
        # readily, one at the tip of a thin finger hardly at all (no stringy borders).
        hold = self._surrounded(own, best_owner)
        compact = np.clip((hold - 1) / 2, 0, 2)
        chance = 1 - np.exp(-cfg.absorb_rate / (self.scale * rounds) * dt * np.clip(ratio - 1, 0, 1) * way * compact)
        take = free & (best_owner > 0) & (self.rng.random(own.shape) < chance)
        if take.any():
            own[take] = best_owner[take]
            self._patch_distances(state, take)

    # ------------------------------------------------------------------ war
    def _war_keys(self):
        if not self.pair_war:
            return np.zeros(0, np.int64)
        pairs = np.array(list(self.pair_war.keys()), np.int64)
        return np.concatenate([pairs[:, 0] * 65536 + pairs[:, 1], pairs[:, 1] * 65536 + pairs[:, 0]])

    def _fight(self, state, rounds=1):
        """Along every front, cells may change hands."""
        if not self.wars:
            return
        cfg = state.config
        dt = self.every
        own = state.cells["state"]
        pop = state.cells["population"]
        keys = self._war_keys()
        size = state.cells.get("settlement")
        town = np.log1p(size / 2000) if size is not None else 0
        records = state.tables["states"]
        capitals = np.zeros(own.shape, bool)
        for sid in self.alive_ids():
            capitals.flat[records[sid]["capital"]] = True
        defence = self.defence + cfg.town_defence * town + cfg.capital_defence * capitals
        open_defence = self.hill_defence
        best_p = np.zeros(own.shape, np.float32)
        best_owner = np.zeros_like(own)
        for dy, dx in DIRECTIONS4:
            attacker = shift(own, dy, dx)
            at_war = (attacker > 0) & (own > 0) & (attacker != own)
            if not at_war.any():
                continue
            pair = attacker.astype(np.int64) * 65536 + own
            at_war &= np.isin(pair, keys)
            if not at_war.any():
                continue
            push = cfg.war_edge * np.log(np.maximum(shift(self.local, dy, dx), 1e-9)
                                         / np.maximum(self.local, 1e-9))
            if self.scale > 1:
                p = self._coarse_take(cfg.war_rate * dt, push, open_defence, defence, capitals)
            else:
                edge = push - defence
                p = np.minimum(cfg.war_rate / (self.scale * rounds) * dt / (1 + np.exp(-3 * edge)), 1.0)
            # Fronts move as lines: cells the attacker half surrounds fall sooner than
            # spearheads (otherwise conquests grow as diamonds along the map grid).
            p = p * np.clip(self._surrounded(own, attacker) / 3, 1 / 3, 5 / 3)
            better = at_war & (p > best_p)
            best_p = np.where(better, p, best_p)
            best_owner = np.where(better, attacker, best_owner)
        flip = (best_owner > 0) & (self.rng.random(own.shape) < best_p)
        if not flip.any():
            return
        cells = np.flatnonzero(flip)
        losers, winners = own.flat[cells].copy(), best_owner.flat[cells]
        dead = pop.flat[cells] * cfg.conquest_loss
        pop.flat[cells] -= dead
        intensity = state.cells.get("intensity")
        if intensity is not None:          # fields and works damaged in the fighting
            intensity.flat[cells] *= 1 - cfg.conquest_intensity_loss
        self._deaths += float(dead.sum())
        own.flat[cells] = winners
        for a, b in zip(winners, losers):
            war = self.wars.get(self.pair_war.get((min(a, b), max(a, b))))
            if war is not None:
                side = 1 if a in war["attackers"] else -1
                war["net"] += side
                war["cells"] += 1
        for b in np.unique(losers):
            cap = records[b]["capital"]
            if own.flat[cap] != b:
                self._capital_falls(state, int(b), int(own.flat[cap]))
        self._patch_distances(state, flip)

    def _coarse_take(self, rate, push, open_defence, defence, capitals=None):
        """Chance a coarse cell falls this step. It is `scale` ~50 km cells deep: the front
        crosses open country first and then the defended core (river, walls, capital), so
        the time to take it is the sum of the two, as on the 800-px map. A capital falls
        when the front reaches its town, on average halfway through the open part."""
        q_open = rate / (1 + np.exp(np.clip(-3 * (push - open_defence), -50, 50)))
        q_core = rate / (1 + np.exp(np.clip(-3 * (push - defence), -50, 50)))
        depth = np.full(np.shape(push), self.scale - 1, np.float32)
        if capitals is not None:
            depth = np.where(capitals, (self.scale - 1) / 2, depth)
        return np.minimum(1 / (depth / np.maximum(q_open, 1e-12) + 1 / np.maximum(q_core, 1e-12)), 1.0)

    @staticmethod
    def _surrounded(own, owner):
        """How many of each cell's 8 neighbours belong to `owner` (an array of state ids)."""
        count = np.zeros(own.shape, np.float32)
        for dy, dx in DIRECTIONS8:
            count += (shift(own, dy, dx) == owner) & (owner > 0)
        return count

    def _capital_falls(self, state, loser, winner):
        records = state.tables["states"]
        own = state.cells["state"]
        lose, win = records[loser], records[winner]
        people_left = float(state.cells["population"][own == loser].sum())
        cells_left = int((own == loser).sum())
        if cells_left == 0 or people_left < 0.3 * self.people[winner] or cells_left < cells_for(8, self.scale):
            own[own == loser] = winner          # the whole state is taken
            self._end_state(state, loser, f"conquered by {win['title']}")
            self._patch_distances(state, own == winner)
            if self._notable(loser, winner):
                state.event("war", f"The {win['title']} conquers the {lose['title']}.",
                            winner=winner, loser=loser)
            return
        old = lose["capital_name"]
        cell = self._best_capital(state, loser)
        lose["capital"] = cell
        lose["capital_name"] = self._place_name(state, cell) or lose["capital_name"]
        self.cohesion[loser] = max(0.02, self.cohesion[loser] - 0.15)
        self._distances(state, only=loser)
        for war in self.wars.values():
            if loser in war["attackers"] + war["defenders"]:
                war["capital_fell"] = True
        if self._notable(loser, winner):
            state.event("war", f"The {win['title']} takes {old}, capital of the {lose['title']}; "
                               f"its rulers flee to {lose['capital_name']}.", winner=winner, loser=loser)

    def _best_capital(self, state, sid):
        own = state.cells["state"]
        cells = np.flatnonzero(own == sid)
        size = state.cells.get("settlement")
        score = state.cells["population"].flat[cells] + (size.flat[cells] * 10 if size is not None else 0)
        return int(cells[np.argmax(score)])

    def _place_name(self, state, cell):
        places = state.tables.get("settlements") or []
        ids = state.cells.get("settlement_id")
        if ids is not None and 0 < ids.flat[cell] < len(places) and places[ids.flat[cell]]:
            return places[ids.flat[cell]]["name"]
        return None

    # ------------------------------------------------------------ decisions
    def _borders(self, state):
        """Pairs of neighbouring states: border length and each side's strength on it."""
        own = state.cells["state"]
        a_list, b_list, la, lb = [], [], [], []
        for dy, dx in ((0, 1), (1, 0)):
            other = shift(own, dy, dx)
            m = (own > 0) & (other > 0) & (own != other)
            if m.any():
                a_list.append(own[m]); b_list.append(other[m])
                la.append(self.local[m]); lb.append(shift(self.local, dy, dx)[m])
        if not a_list:
            return {}
        a, b = np.concatenate(a_list).astype(np.int64), np.concatenate(b_list).astype(np.int64)
        la, lb = np.concatenate(la), np.concatenate(lb)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        l_lo = np.where(a == lo, la, lb)
        l_hi = np.where(a == lo, lb, la)
        key = lo * 65536 + hi
        uniq, inv, count = np.unique(key, return_inverse=True, return_counts=True)
        s_lo = np.bincount(inv, weights=l_lo)
        s_hi = np.bincount(inv, weights=l_hi)
        return {(int(k // 65536), int(k % 65536)): (int(n), float(x), float(y))
                for k, n, x, y in zip(uniq, count, s_lo, s_hi)}

    def _decide(self, state):
        cfg = state.config
        self._summary(state)              # include states founded this step
        borders = self._borders(state)
        records = state.tables["states"]
        families = {}
        cultures = state.tables.get("cultures") or []
        for sid in self.alive_ids():
            c = records[sid]["culture"]
            families[sid] = cultures[c]["family"] if c and c < len(cultures) and cultures[c] else -sid
        neighbours = {}
        for (a, b), (count, s_a, s_b) in borders.items():
            neighbours.setdefault(a, []).append((b, count, s_a / max(s_b, 1e-9)))
            neighbours.setdefault(b, []).append((a, count, s_b / max(s_a, 1e-9)))
        wars_of = {}
        for war in self.wars.values():
            for sid in war["attackers"] + war["defenders"]:
                wars_of[sid] = wars_of.get(sid, 0) + 1
        year = state.year
        for sid, nbs in neighbours.items():
            if self.people[sid] < cfg.min_war_people:
                continue
            sit = Situation(id=sid, year=year, power=float(self.power[sid]), cohesion=float(self.cohesion[sid]),
                            aggression=float(self.aggression[sid]), wars=wars_of.get(sid, 0))
            for other, count, ratio in nbs:
                pair = (min(sid, other), max(sid, other))
                sit.neighbours.append(Neighbour(
                    id=other, border=count * self.scale, ratio=ratio, at_war=pair in self.pair_war,
                    truce=self.truces.get(pair, -1) > year, allied=self.alliances.get(pair, -1) > year,
                    their_wars=wars_of.get(other, 0), same_family=families.get(sid) == families.get(other),
                    threat=float(self.power[other] / max(self.power[sid], 1e-9))))
            for action, target in self.ai.decide(sit, cfg.decision_years):
                pair = (min(sid, target), max(sid, target))
                if action == "war" and pair not in self.pair_war:
                    self._declare(state, sid, target)
                elif action == "annex" and pair not in self.pair_war:
                    self._submit(state, target, sid)
                elif action == "ally" and pair not in self.pair_war:
                    self.alliances[pair] = year + int(self.rng.integers(40, 150))

    def _submit(self, state, weak, strong):
        """A small neighbour bows to an overwhelming power and is absorbed without war."""
        records = state.tables["states"]
        if records[weak]["ended"] is not None or records[strong]["ended"] is not None:
            return
        if any(weak in w["attackers"] + w["defenders"] for w in self.wars.values()):
            return
        own = state.cells["state"]
        mask = own == weak
        own[mask] = strong
        self._end_state(state, weak, f"submitted to the {records[strong]['title']}")
        self._patch_distances(state, mask)
        if self._notable(strong, weak) and self.people[weak] >= NOTABLE:
            state.event("state", f"The {records[weak]['title']} submits to the {records[strong]['title']}.",
                        state_id=strong, absorbed=weak)

    def _declare(self, state, attacker, defender):
        records = state.tables["states"]
        year = state.year
        allies = [s for s in self.alive_ids()
                  if s not in (attacker, defender)
                  and self.alliances.get((min(s, defender), max(s, defender)), -1) > year
                  and (min(s, attacker), max(s, attacker)) not in self.pair_war]
        war = {"id": len(state.tables["wars"]), "attackers": [int(attacker)],
               "defenders": [int(defender)] + [int(s) for s in allies],
               "started": year, "ended": None, "outcome": None, "net": 0, "cells": 0,
               "capital_fell": False, "_weariness": 0.0}
        state.tables["wars"].append(war)
        self.wars[war["id"]] = war
        for d in war["defenders"]:
            self.pair_war[(min(attacker, d), max(attacker, d))] = war["id"]
        if self._notable(attacker, defender):
            names = [records[s]["name"] for s in allies[:3]]
            joined = ""
            if len(names) == 1:
                joined = f", whose ally {names[0]} joins in"
            elif names:
                joined = ", whose allies " + ", ".join(names[:-1]) + f" and {names[-1]} join in"
            state.event("war", f"The {records[attacker]['title']} goes to war against the "
                               f"{records[defender]['title']}{joined}.", war=war["id"])

    def _end_wars(self, state):
        cfg = state.config
        records = state.tables["states"]
        own = state.cells["state"]
        dt = self.every
        for wid in list(self.wars):
            war = self.wars[wid]
            sides = war["attackers"] + war["defenders"]
            alive = all(records[s]["ended"] is None for s in war["attackers"]) and \
                any(records[s]["ended"] is None for s in war["defenders"])
            war["_weariness"] += dt / cfg.war_years
            chance = 1 - np.exp(-dt / cfg.war_years * war["_weariness"])
            if alive and not war["capital_fell"] and self.rng.random() >= chance:
                continue
            war["ended"] = state.year
            net = war["net"]
            a, d = war["attackers"][0], war["defenders"][0]
            war["outcome"] = "attacker" if net > 0 else "defender" if net < 0 else "stalemate"
            for s in war["defenders"]:
                pair = (min(a, s), max(a, s))
                self.pair_war.pop(pair, None)
                self.truces[pair] = state.year + int(self.rng.integers(*cfg.truce_years))
            del self.wars[wid]
            if alive and war["cells"] > 0 and self._notable(a, d):
                ra, rd = records[a], records[d]
                if net > 0:
                    text = f"Peace: the {ra['title']} wins land from the {rd['title']}."
                elif net < 0:
                    text = f"Peace: the {rd['title']} beats off the {ra['title']} and wins land from it."
                else:
                    text = f"The war between the {ra['title']} and the {rd['title']} ends in a draw."
                state.event("war", text, war=wid)

    # --------------------------------------------------- administration
    def _distances(self, state, only=None):
        """Travel distance from every cell to its own state's capital (km, weighted by terrain)."""
        own = state.cells["state"].ravel()
        records = state.tables["states"]
        ids = [only] if only is not None else list(self.alive_ids())
        if not ids:
            return
        caps = [records[s]["capital"] for s in ids if own[records[s]["capital"]] == s]
        if not caps:
            return
        if only is None:
            sub = self._same_state_graph(own)
            dist = dijkstra(sub, directed=False, indices=caps, min_only=True)
            out = np.where(own > 0, dist, np.inf)
        else:
            cells = np.flatnonzero(own == only)
            g = self.graph[cells][:, cells]
            local = {int(c): i for i, c in enumerate(cells)}
            if records[only]["capital"] not in local:
                return
            d = dijkstra(g, directed=False, indices=local[records[only]["capital"]])
            out = state.cells["state_distance"].ravel().copy()
            out[cells] = d
        # Cut-off pockets (no land route to the capital): a long way, as the crow flies.
        lost = (own > 0) & ~np.isfinite(out)
        if lost.any():
            cap_xyz = self.xyz[np.array([records[s]["capital"] if records[s] else 0 for s in range(len(records))])]
            out[lost] = 2 * great_circle_km(self.xyz[lost], cap_xyz[own[lost]])
        state.cells["state_distance"] = out.reshape(state.shape).astype(np.float32)

    def _same_state_graph(self, own):
        a, b = self.edges
        same = (own[a] == own[b]) & (own[a] > 0)
        data = np.asarray(self.graph[a[same], b[same]], dtype=np.float64).ravel() if same.any() else np.zeros(0)
        return coo_matrix((np.concatenate([data, data]), (np.concatenate([a[same], b[same]]),
                                                         np.concatenate([b[same], a[same]]))),
                          shape=(self.n, self.n)).tocsr()

    def _patch_distances(self, state, changed):
        """Newly gained cells: distance through their nearest cell of the same state."""
        own = state.cells["state"]
        dist = state.cells["state_distance"]
        best = np.full(own.shape, np.inf, np.float32)
        step = (state.world.grid.km_per_px * self.cost).astype(np.float32)
        for dy, dx in DIRECTIONS4:
            same = shift(own, dy, dx) == own
            best = np.where(changed & same, np.minimum(best, shift(dist, dy, dx) + step), best)
        dist[changed] = np.where(np.isfinite(best[changed]), best[changed], 1e5)

    def _admin(self, state):
        cfg = state.config
        records = state.tables["states"]
        self._tidy(state)
        self._distances(state)
        self._summary(state)
        self._secessions(state)
        self._collapses(state)
        self._summary(state)
        n = len(records)
        cult = state.cells.get("culture")
        ruled_by_own = None
        if cult is not None:
            own = state.cells["state"]
            match = cult == self.ruler_culture[own]
            ruled_by_own = np.bincount(own.ravel(), weights=(state.cells["population"] * match).ravel(),
                                       minlength=n)
        size = state.cells.get("settlement")
        for sid in range(1, n):
            rec = records[sid]
            if rec is None or rec["ended"] is not None:
                continue
            if self.cell_count[sid] == 0:
                self._end_state(state, sid, rec["fate"] or "absorbed by its neighbours")
                continue
            people = float(self.people[sid])
            if people > rec["peak_people"]:
                rec["peak_people"], rec["peak_cells"], rec["peak_year"] = round(people), int(self.cell_count[sid]), state.year
            own_share = ruled_by_own[sid] / max(people, 1e-9) if ruled_by_own is not None else 1.0
            cap_size = float(size.flat[rec["capital"]]) if size is not None else 0.0
            kind = rec["kind"]
            # Promotions at the thresholds; demotions only well below them (no flickering).
            if people >= cfg.empire_people or (people >= cfg.empire_people / 2 and own_share < 0.5):
                kind = "empire"
            elif kind == "empire" and people >= cfg.empire_people / 4:
                pass
            elif people >= cfg.kingdom_people and (cap_size >= cfg.kingdom_capital_min
                                                    or people >= 2 * cfg.kingdom_people):
                kind = "kingdom"
            elif kind in ("kingdom", "empire") and people >= cfg.kingdom_people / 2:
                kind = "kingdom"
            elif self.cell_count[sid] <= cells_for(3, self.scale) and cap_size >= 5000:
                kind = "city-state"
            else:
                kind = "chiefdom"
            if kind != rec["kind"]:
                self._new_kind(state, sid, kind)
        self._largest_event(state)

    def _new_kind(self, state, sid, kind):
        rec = state.tables["states"][sid]
        old = rec["title"]
        if kind == "empire" and rec.get("empire_from") is None:
            rec["empire_from"] = state.year
        if rec["kind"] == "empire" and kind != "empire":
            rec["empire_to"] = state.year
        rec["kind"] = kind
        rec["title"] = TITLES[kind].format(rec["name"])
        if kind == "kingdom" and not self._first_kingdom:
            self._first_kingdom = True
            state.event("state", f"The {rec['title']}, ruled from {rec['capital_name']}, is the world's first kingdom.",
                        state_id=sid)
        elif kind == "empire":
            first = "" if self._first_empire else " the world's first empire,"
            self._first_empire = True
            state.event("state", f"The {old} becomes the {rec['title']},{first} ruling "
                                 f"{_people(self.people[sid])}.", state_id=sid)

    def _largest_event(self, state):
        alive = list(self.alive_ids())
        if not alive:
            return
        top = max(alive, key=lambda s: self.people[s])
        if (top != self._largest and self.people[top] >= 1e6
                and state.year - self._largest_year >= state.config.largest_state_event_years):
            self._largest, self._largest_year = top, state.year
            rec = state.tables["states"][top]
            state.event("state", f"The {rec['title']} is now the largest state in the world, "
                                 f"ruling {_people(self.people[top])}.", state_id=top)

    def _end_state(self, state, sid, fate):
        rec = state.tables["states"][sid]
        if rec["ended"] is None:
            rec["ended"] = state.year
            rec["fate"] = fate
            if rec["kind"] == "empire" and rec.get("empire_to") is None:
                rec["empire_to"] = state.year
        for pair in [p for p in self.pair_war if sid in p]:
            war = self.wars.get(self.pair_war[pair])
            if war is not None:
                war["capital_fell"] = True

    def _tidy(self, state):
        """Enclaves surrounded by one state join it, and tiny statelets (a cell or two, not a
        real city-state) are swallowed by their strongest neighbour."""
        cfg = state.config
        own = state.cells["state"]
        records = state.tables["states"]
        surround = np.zeros_like(own)
        count = np.zeros(own.shape, np.int8)
        for dy, dx in DIRECTIONS4:
            nb = shift(own, dy, dx)
            same = (nb > 0) & (nb != own)
            first = same & (surround == 0)
            surround = np.where(first, nb, surround)
            count += same & (nb == surround)
        enclave = (own > 0) & (count >= 3) & (self.rng.random(own.shape) < cfg.enclave_absorb)
        capitals = np.zeros(own.shape, bool)
        for sid in self.alive_ids():
            capitals.flat[records[sid]["capital"]] = True
        enclave &= ~capitals
        if enclave.any():
            own[enclave] = surround[enclave]
        n = len(records)
        cells = np.bincount(own.ravel(), minlength=n)
        size = state.cells.get("settlement")
        for sid in list(self.alive_ids()):
            rec = records[sid]
            if not 0 < cells[sid] <= cells_for(2, self.scale) or (size is not None and size.flat[rec["capital"]] >= 10_000):
                continue
            if self.rng.random() >= cfg.statelet_absorb:
                continue
            mask = own == sid
            best, best_power = 0, 0.0
            for dy, dx in DIRECTIONS4:
                nb = shift(own, dy, dx)[mask]
                for other in np.unique(nb[(nb > 0) & (nb != sid)]):
                    p = self.power[other] if other < len(self.power) else 0
                    if p > best_power:
                        best, best_power = int(other), p
            if best and best_power > 3 * (self.power[sid] if sid < len(self.power) else 0):
                own[mask] = best
                self._end_state(state, sid, f"absorbed by the {records[best]['title']}")

    # --------------------------------------------------------- secessions
    def _collapses(self, state):
        """A big state whose cohesion has run out falls apart at once into successor states,
        each around one of its great towns (as empires broke up into kingdoms)."""
        cfg = state.config
        own = state.cells["state"]
        records = state.tables["states"]
        size = state.cells.get("settlement")
        pop = state.cells["population"].ravel()
        for sid in list(self.alive_ids()):
            if (sid >= len(self.cell_count) or self.cell_count[sid] < cells_for(cfg.collapse_min_cells, self.scale)
                    or self.cohesion[sid] >= cfg.collapse_cohesion):
                continue
            cells = np.flatnonzero(own.ravel() == sid)
            k = int(np.clip(len(cells) * self.scale ** 2 / cfg.collapse_cells_per_successor, 2, 8))
            score = pop[cells] + (size.ravel()[cells] * 10 if size is not None else 0)
            # Successor seats: the capital, then great towns far from those already chosen.
            seeds = [records[sid]["capital"]]
            gap = great_circle_km(self.xyz[cells], self.xyz[seeds[0]])
            for _ in range(k - 1):
                pick = cells[np.argmax(score * (gap > 250) * gap)] if (gap > 250).any() else None
                if pick is None or pick in seeds:
                    break
                seeds.append(int(pick))
                gap = np.minimum(gap, great_circle_km(self.xyz[cells], self.xyz[pick]))
            if len(seeds) < 2:
                continue
            sub = self.graph[cells][:, cells]
            local = {int(c): i for i, c in enumerate(cells)}
            _, _, src = dijkstra(sub, directed=False, indices=[local[s] for s in seeds],
                                 min_only=True, return_predecessors=True)
            seat_of = {local[s]: s for s in seeds}
            cult = state.cells.get("culture")
            born = []
            for seed in seeds[1:]:
                part = cells[src == local[seed]]
                if len(part) == 0:
                    continue
                culture = int(np.argmax(np.bincount(cult.ravel()[part], weights=pop[part]))) if cult is not None else 0
                new = self._found(state, seed, 0, culture=culture, parent=sid, cells=part,
                                  cohesion=self.rng.uniform(0.5, 0.8))
                if new is None:
                    break
                if pop[part].sum() >= 2 * cfg.kingdom_people:
                    rec = records[new]
                    rec["kind"], rec["title"] = "kingdom", TITLES["kingdom"].format(rec["name"])
                pair = (min(sid, new), max(sid, new))
                self.truces[pair] = state.year + int(self.rng.integers(*cfg.truce_years))
                born.append(new)
            # Stray pockets with no route to any seat stay with the old state.
            self.cohesion[sid] = self.rng.uniform(0.35, 0.55)
            rec = records[sid]
            rec["collapses"] = rec.get("collapses", 0) + 1
            if born and (self._notable(sid, sid) or len(born) >= 3):
                names = [records[b]["name"] for b in born[:4]]
                state.event("state", f"The {rec['title']} falls apart: {', '.join(names)}"
                                     f"{' and others' if len(born) > 4 else ''} go their own way.",
                            state_id=sid, successors=born)
            self._distances(state, only=sid)

    def _secessions(self, state):
        """Distant provinces, above all of other peoples, break away from weak states."""
        cfg = state.config
        own = state.cells["state"]
        cult = state.cells.get("culture")
        if cult is None or not (own > 0).any():
            return
        records = state.tables["states"]
        dist = state.cells["state_distance"]
        reach = self.reach[own]
        families = self._family_lookup(state)
        ruler = self.ruler_culture[own]
        gap = np.where(cult == ruler, 0.0, np.where(families[cult] == families[ruler], cfg.related_gap,
                                                    cfg.foreign_gap))
        sep = (np.where(np.isfinite(dist), dist, 1e6) / np.maximum(reach, 1) + gap) * (1.5 - self.cohesion[own])
        # (never so small that the new state counts as a statelet, on coarse maps too)
        min_rebel = max(cells_for(cfg.min_rebel_cells, self.scale), cells_for(2, self.scale) + 1)
        restless = (own > 0) & (sep > cfg.secession_threshold) & (self.cell_count[own] >= min_rebel * 2)
        if not restless.any():
            return
        flat_own = own.ravel()
        flat_rest = restless.ravel()
        a, b = self.edges
        keep = flat_rest[a] & flat_rest[b] & (flat_own[a] == flat_own[b])
        g = coo_matrix((np.ones(keep.sum()), (a[keep], b[keep])), shape=(self.n, self.n))
        _, comp = connected_components(g, directed=False)
        idx = np.flatnonzero(flat_rest)
        labels, inverse, counts = np.unique(comp[idx], return_inverse=True, return_counts=True)
        p_years = cfg.admin_years / cfg.rebellion_years
        pop = state.cells["population"].ravel()
        for k in np.flatnonzero(counts >= min_rebel):
            cells = idx[inverse == k]
            sid = int(flat_own[cells[0]])
            if records[sid]["ended"] is not None or self.cell_count[sid] - len(cells) < 1:
                continue
            strength = float(sep.ravel()[cells].mean())
            if self.rng.random() >= 1 - np.exp(-p_years * strength):
                continue
            size = state.cells.get("settlement")
            score = pop[cells] + (size.ravel()[cells] * 10 if size is not None else 0)
            capital = int(cells[np.argmax(score)])
            local = np.bincount(cult.ravel()[cells], weights=pop[cells])
            culture = int(np.argmax(local))
            new = self._found(state, capital, 0, culture=culture, parent=sid, cells=cells,
                              cohesion=min(1.0, self.cohesion[sid] + 0.35))
            if new is None:
                return
            if pop[cells].sum() >= 2 * cfg.kingdom_people:    # a big province breaks away as a kingdom
                rec = records[new]
                rec["kind"], rec["title"] = "kingdom", TITLES["kingdom"].format(rec["name"])
            pair = (min(sid, new), max(sid, new))
            self.truces[pair] = state.year + int(self.rng.integers(*cfg.truce_years))
            if self._notable(sid, sid) or pop[cells].sum() >= max(NOTABLE, self._world_people * self._notable_share) / 3:
                rec = records[new]
                state.event("state", f"The lands around {rec['capital_name']} break away from the "
                                     f"{records[sid]['title']} as the {rec['title']}.", state_id=new, parent=sid)

    # ------------------------------------------------------------- helpers
    def alive_ids(self):
        recs = self.records
        return (s for s in range(1, len(recs)) if recs[s] is not None and recs[s]["ended"] is None)

    def _notable(self, a, b):
        """Worth a line in the chronicle: a big state (in absolute terms and in its world)."""
        bar = max(NOTABLE, self._world_people * self._notable_share)
        return self.people[a] >= bar or self.people[b] >= bar

    # ----------------------------------------------------------------- stats
    def _stats(self, state):
        own = state.cells["state"]
        pop = state.cells["population"]
        total = max(float(pop.sum()), 1e-9)
        records = state.tables["states"]
        alive = [s for s in range(1, len(records)) if records[s] and records[s]["ended"] is None
                 and s < len(self.cell_count) and self.cell_count[s] > 0]
        people = self.people if len(self.people) == len(records) else np.bincount(own.ravel(), weights=pop.ravel(), minlength=len(records))
        state.stats.update({
            "states": len(alive),
            "kingdoms": sum(1 for s in alive if records[s]["kind"] in ("kingdom", "empire")),
            "empires": sum(1 for s in alive if records[s]["kind"] == "empire"),
            "people_in_states": float(pop[own > 0].sum() / total),
            "largest_state_share": float(max((people[s] for s in alive), default=0) / total),
            "wars": len(self.wars),
            "wars_started": len(state.tables["wars"]),
            "war_deaths_per_year": self._deaths / self.every,
        })


def _people(n):
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.1f} million people"
    return f"{round(n, -3):,.0f} people"
