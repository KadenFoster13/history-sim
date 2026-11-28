"""Checks for phase 3: states, war, peace and breakup. Run: python -m unittest discover tests"""
import unittest

import numpy as np
import worldgen

from histsim import Simulation, default_systems
from histsim.agents import Neighbour, Situation, StateAI
from histsim.config import DEFAULT

WORLD = worldgen.generate_world(worldgen.WorldConfig(seed=22, width=400))
SIM = Simulation(WORLD, seed=1, systems=default_systems())
SIM.run(9000)        # to 1000 BCE: states, wars and breakups have happened


class StateTests(unittest.TestCase):
    def test_states_rule_people(self):
        st = SIM.state
        own, pop = st.cells["state"], st.cells["population"]
        self.assertGreater(st.stats["states"], 5)
        self.assertGreater(st.stats["kingdoms"], 1)
        self.assertTrue((pop[own > 0] > 0).all())          # no state rules empty land
        records = st.tables["states"]
        for sid in np.unique(own[own > 0]):
            rec = records[sid]
            self.assertIsNone(rec["ended"])
            self.assertEqual(own.flat[rec["capital"]], sid)  # every living state holds its capital
            self.assertTrue(rec["name"] and rec["title"])

    def test_hemmed_in_farmland(self):
        sim = Simulation(WORLD, seed=2, systems=default_systems())
        sim._setup()
        circ = sim.state.tables["polity"].circumscribed
        self.assertTrue(((circ >= 0) & (circ <= 1)).all())
        self.assertGreater(circ.max(), 0.5)                  # some farmland is hemmed in
        self.assertEqual(circ[WORLD.ocean].max(), 0)

    def test_no_states_before_towns(self):
        sim = Simulation(WORLD, seed=1, systems=default_systems())
        sim.run(4000)
        self.assertEqual(sim.state.stats["states"], 0)

    def test_wars_are_recorded(self):
        wars = SIM.state.tables["wars"]
        self.assertTrue(wars)
        for war in wars:
            self.assertFalse(set(war["attackers"]) & set(war["defenders"]))
            if war["ended"] is not None:
                self.assertIn(war["outcome"], ("attacker", "defender", "stalemate"))
                self.assertGreaterEqual(war["ended"], war["started"])

    def test_ended_states_hold_nothing(self):
        own = SIM.state.cells["state"]
        ended = [s["id"] for s in SIM.state.tables["states"][1:] if s and s["ended"] is not None]
        self.assertFalse(np.isin(own, ended).any())


    def test_empire_bar_rises_with_the_world(self):
        from histsim.systems.states import empire_bar
        self.assertAlmostEqual(empire_bar(DEFAULT, DEFAULT.empire_world), DEFAULT.empire_people)
        small, big = empire_bar(DEFAULT, 30e6), empire_bar(DEFAULT, 300e6)
        self.assertLess(small, DEFAULT.empire_people)
        self.assertGreater(big, DEFAULT.empire_people)
        self.assertLess(big / small, 10)            # more slowly than the world's people

    def test_coarse_fronts(self):
        # A 100 km cell is two ~50 km cells deep: open country, then its defended core.
        from histsim.systems.states import States
        s = States()
        s.scale = 2.0
        push = np.zeros(3, np.float32)
        open_d = np.zeros(3, np.float32)
        core = np.array([0.0, 0.5, 2.0], np.float32)
        p = s._coarse_take(0.6, push, open_d, core)
        self.assertAlmostEqual(float(p[0]), 0.15, places=5)     # undefended: half the rate
        q_open, q_core = 0.3, 0.6 / (1 + np.exp(3 * 2.0))
        self.assertAlmostEqual(float(p[2]), 1 / (1 / q_open + 1 / q_core), places=5)
        self.assertTrue(p[0] > p[1] > p[2])


class GrowthTests(unittest.TestCase):
    """Phase 4 step 2: the technology of rule, expansion in pulses, land by its worth,
    chiefdoms that dissolve, land falling back."""

    def test_technology_of_rule(self):
        from histsim.tech import bit
        st = SIM.state
        polity = st.tables["polity"]
        polity._summary(st)
        sid = max(polity.alive_ids(), key=lambda s: polity.people[s])
        cap = st.tables["states"][sid]["capital"]
        tech = st.cells["tech"]
        saved = tech.flat[cap]
        try:
            tech.flat[cap] = 0
            polity._summary(st)
            reach0, army0 = polity.reach[sid], polity.military[sid]
            tech.flat[cap] = np.uint64(bit("writing") | bit("bureaucracy") | bit("roads") | bit("bronze")
                                       | bit("copper") | bit("counting") | bit("wheel"))
            polity._summary(st)
            self.assertGreater(polity.reach[sid], reach0 * 1.5)       # records, officials, roads
            self.assertGreater(polity.military[sid], army0 * 1.2)     # bronze weapons
            self.assertEqual(polity.knows["bureaucracy"][sid], 1)
        finally:
            tech.flat[cap] = saved
            polity._summary(st)

    def test_expansion_comes_in_pulses(self):
        polity = SIM.state.tables["polity"]
        sid = next(polity.alive_ids())
        drive, gained = polity.drive[sid], polity.gained[sid]
        try:
            polity.drive[sid], polity.gained[sid] = 1.0, 0.0
            self.assertEqual(float(polity._expanding(np.array([sid]))[0]), 1.0)     # a young dynasty
            polity.gained[sid] = 1.0
            self.assertLess(float(polity._expanding(np.array([sid]))[0]), 0.5)      # digesting its gains
            polity.drive[sid], polity.gained[sid] = 0.1, 0.0
            self.assertEqual(float(polity._expanding(np.array([sid]))[0]), 0.0)     # an old one holds
        finally:
            polity.drive[sid], polity.gained[sid] = drive, gained

    def test_land_is_taken_for_its_worth(self):
        from histsim.systems.population import FARMING, FORAGING
        st = SIM.state
        polity = st.tables["polity"]
        polity._summary(st)
        mode, pop = st.cells["subsistence"], st.cells["population"]
        farm, forage = (mode == FARMING) & (pop > 0), (mode == FORAGING) & (pop > 0)
        self.assertGreater(np.median(polity.worth[farm]), 5 * np.median(polity.worth[forage]))
        self.assertTrue(((polity.worth >= 0) & (polity.worth <= 1)).all())

    def test_kinds_are_recorded(self):
        for rec in SIM.state.tables["states"][1:]:
            self.assertEqual(rec["kinds"][0][0], rec["founded"])
            self.assertEqual(rec["kinds"][-1][1], rec["kind"])

    def test_chiefdoms_dissolve(self):
        sim = Simulation(WORLD, seed=3, systems=default_systems())
        sim.run(7000)
        st = sim.state
        polity = st.tables["polity"]
        polity._summary(st)
        chiefs = [s for s in polity.alive_ids() if polity.chief[s]]
        while not chiefs and st.year < 9000:         # until one is alive (none may be at 7000)
            sim.run(st.year + 250)
            polity._summary(st)
            chiefs = [s for s in polity.alive_ids() if polity.chief[s]]
        self.assertTrue(chiefs)
        sid = chiefs[0]
        polity.cohesion[sid] = 0.01
        polity._fall_back(st)
        self.assertEqual(st.tables["states"][sid]["fate"], "dissolved")
        self.assertFalse((st.cells["state"] == sid).any())        # its villages govern themselves


class DecisionTests(unittest.TestCase):
    def test_strong_rulers_attack_weak_neighbours(self):
        ai = StateAI(DEFAULT, np.random.default_rng(0))
        weak = Neighbour(id=2, border=10, ratio=5.0, at_war=False, truce=False, allied=False,
                         their_wars=0, same_family=False, threat=0.2)
        sit = Situation(id=1, year=0, power=1e6, cohesion=0.8, aggression=1.0, wars=0, neighbours=[weak])
        attacks = sum(any(a == ("war", 2) for a in ai.decide(sit, 10)) for _ in range(200))
        self.assertGreater(attacks, 20)
        truce = Neighbour(**{**weak.__dict__, "truce": True})
        sit.neighbours = [truce]
        self.assertFalse(any(ai.decide(sit, 10) for _ in range(100)))


if __name__ == "__main__":
    unittest.main()
