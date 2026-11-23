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
