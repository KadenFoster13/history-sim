"""Checks for phase 4 step 1: technology found, spread and lost. Run: python -m unittest discover tests"""
import unittest

import numpy as np
import worldgen

from histsim import Simulation, default_systems
from histsim.config import DEFAULT
from histsim.tech import INDEX, NEEDS, TECHS, age_of, bit, known_list

WORLD = worldgen.generate_world(worldgen.WorldConfig(seed=22, width=400))


def run(config=DEFAULT, years=600, systems=None, seed=1):
    sim = Simulation(WORLD, seed=seed, systems=systems or default_systems(), config=config)
    sim.run(years)
    return sim


class WebTests(unittest.TestCase):
    def test_web_is_consistent(self):
        self.assertLessEqual(len(TECHS), 64)
        for i, t in enumerate(TECHS):
            self.assertEqual(INDEX[t.key], i)
            for p in t.prereqs:
                self.assertLess(INDEX[p], i)           # prerequisites come first
            for n in t.find_needs + t.keep_needs:
                self.assertIn(n, NEEDS)
        mask = bit("copper") | bit("bronze") | bit("writing")
        self.assertEqual([t.key for t in known_list(mask)], ["copper", "bronze", "writing"])
        self.assertEqual(int(age_of(mask)), 2)          # bronze age
        self.assertEqual(int(age_of(bit("pottery"))), 0)


class TechnologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Technologies found a thousand times more readily, so a short run shows them.
        cls.sim = run(DEFAULT.copy(tech_years=1.0), years=600)

    def test_found_and_spread(self):
        st = self.sim.state
        tech = st.cells["tech"]
        pottery = (tech & np.uint64(bit("pottery"))) != 0
        self.assertTrue(pottery.any())
        self.assertGreater(st.stats["know_pottery"], 0.05)
        inventions = st.tables["inventions"]
        self.assertTrue(any(i["tech"] == "pottery" and i["first"] for i in inventions))
        self.assertTrue(any(e["kind"] == "technology" for e in st.events) or inventions)
        # Every invention is far from the others of the same technology.
        from histsim.systems.geo import great_circle_km
        xyz = np.stack(WORLD.grid.xyz, axis=-1)
        pots = [i for i in inventions if i["tech"] == "pottery"]
        for a in range(len(pots)):
            for b in range(a + 1, len(pots)):
                d = great_circle_km(xyz[pots[a]["row"], pots[a]["col"]], xyz[pots[b]["row"], pots[b]["col"]])
                self.assertGreater(float(d), DEFAULT.tech_independent_km - 1)

    def test_only_where_it_can_be_used(self):
        st = self.sim.state
        tech = st.cells["tech"]
        self.assertFalse((tech[st.cells["population"] <= 0] != 0).any())   # nobody, no knowledge
        system = st.tables["technology"]
        for key, need in (("copper", "copper_near"), ("bronze", "tin_near")):
            has = ((tech & np.uint64(bit(key))) != 0).ravel()
            self.assertFalse((has & ~system.static[need]).any(), key)

    def test_lost_without_keepers(self):
        st = self.sim.state
        system = st.tables["technology"]
        tech = st.cells["tech"]
        tech[:] = np.where(st.cells["population"] > 0, np.uint64(bit("writing") | bit("counting")), 0)
        st.cells["settlement"][:] = 0                       # no towns anywhere: nobody keeps writing
        before = int(((tech & np.uint64(bit("writing"))) != 0).sum())
        for _ in range(int(1000 / system.every)):          # ~7 half-lives
            system.step(st)
            st.cells["settlement"][:] = 0
        after = int(((tech & np.uint64(bit("writing"))) != 0).sum())
        self.assertLess(after, 0.03 * before)

    def test_pottery_alone_changes_nothing(self):
        # In the first centuries only pottery is known, which has no effect yet: the rest of
        # history is the same with or without the technology system.
        without = [s for s in default_systems() if s.name != "technology"]
        a = run(DEFAULT.copy(tech_years=1.0), years=300)
        b = run(DEFAULT.copy(tech_years=1.0), years=300, systems=without)
        np.testing.assert_array_equal(a.state.cells["population"], b.state.cells["population"])
        np.testing.assert_array_equal(a.state.cells["culture"], b.state.cells["culture"])


class FarmingAndTownTests(unittest.TestCase):
    """Phase 4 step 3: farming technology, places fed by what can be brought to them, and
    the ladder of smaller towns."""

    @classmethod
    def setUpClass(cls):
        cls.sim = run(years=8000)                 # to 2000 BCE: farming, towns, states

    def test_farming_technology_raises_ceilings(self):
        st = self.sim.state
        farming = st.tables["farming"]
        tech = st.cells["tech"]
        saved = tech.copy()
        try:
            tech[:] = 0
            before = farming.ceiling(st)
            tech[:] = np.uint64(bit("plough") | bit("manuring") | bit("canals") | bit("terraces"))
            after = farming.ceiling(st)
            self.assertTrue((after >= before - 1e-5).all())
            rainfed = (farming.flood < 0.05) & (farming.irrigable < 0.05) & (before > 1.5)
            self.assertTrue((after[rainfed] > before[rainfed]).all())          # plough, manure
            dry = (farming.irrigable > 0.5) & (farming.dryness > 0.5) & (farming.flood < 0.05)
            if dry.any():
                self.assertGreater(float(after[dry].mean()), 2 * float(before[dry].mean()))   # canals
        finally:
            tech[:] = saved

    def test_places_are_fed(self):
        st = self.sim.state
        system = next(s for s in self.sim.systems if s.name == "settlements")
        limit = system.feed_limit(st)
        size = st.cells["settlement"]
        self.assertTrue((size <= limit * 1.001 + 1).all())               # nobody outgrows their food
        tech = st.cells["tech"]
        saved = tech.copy()
        try:
            tech[:] = 0
            low = system.feed_limit(st)
            tech[:] = np.uint64(bit("writing") | bit("roads") | bit("sailing"))
            self.assertTrue((system.feed_limit(st) > low).all())
        finally:
            tech[:] = saved
        self.assertGreater(float(low[system.water > 0].min()), float(low[system.water == 0].max()))

    def test_ladder_of_towns(self):
        st = self.sim.state
        small = st.cells["small_towns"]
        self.assertTrue((small >= 0).all())
        self.assertGreater(float(small.sum()), 0)
        self.assertGreater(st.stats["towns"], st.stats["named_towns"])    # many unnamed small towns
        self.assertGreaterEqual(st.stats["cities"], st.stats["cities_100k"])


if __name__ == "__main__":
    unittest.main()
