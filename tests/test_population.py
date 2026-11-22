"""Checks for phase 1: peopling, population, farming. Run: python -m unittest discover tests"""
import unittest

import numpy as np
import worldgen

from histsim import Simulation, default_systems
from histsim.config import DEFAULT, PEOPLING_LEVELS, peopling
from histsim.systems.population import FARMING, FORAGING, people_the_world

WORLD = worldgen.generate_world(worldgen.WorldConfig(seed=22, width=400))
LAND = ~WORLD.ocean & ~WORLD.lake


def start(config=DEFAULT, seed=1):
    sim = Simulation(WORLD, seed=seed, systems=default_systems(), config=config)
    sim._setup()
    return sim


class PeoplingTests(unittest.TestCase):
    def test_spread_from_one_homeland(self):
        arrival = people_the_world(WORLD, DEFAULT, np.random.default_rng(1))
        reached = np.isfinite(arrival)
        self.assertTrue(reached[LAND].any())
        self.assertAlmostEqual(float(np.nanmin(arrival[reached])), DEFAULT.out_of_homeland_year, delta=1)
        # Never on ice or open sea far from any land.
        self.assertFalse((reached & WORLD.ocean & (WORLD.elevation_m < DEFAULT.glacial_sea_level_m)).any())

    def test_peopling_levels(self):
        reached = {level: LAND[np.isfinite(people_the_world(WORLD, peopling(level), np.random.default_rng(1)))].sum()
                   for level in PEOPLING_LEVELS}
        self.assertLess(reached["sparse"], reached["some"])
        self.assertLessEqual(reached["some"], reached["most"])
        self.assertLessEqual(reached["most"], reached["all"])
        with self.assertRaises(ValueError):
            peopling("crowded")

    def test_late_arrivals_get_founders(self):
        sim = start()
        arrival = sim.state.cells["arrival_year"]
        late = (arrival > 0) & (arrival < 400) & (sim.state.cells["forager_capacity"] > 0)
        self.assertTrue(late.any())
        self.assertFalse((sim.state.cells["population"][late] > 0).any())
        sim.run(400)
        self.assertTrue((sim.state.cells["population"][late] > 0).mean() > 0.9)

    def test_year_zero_population(self):
        sim = start()
        pop = sim.state.cells["population"]
        self.assertGreater(pop.sum(), 2e5)
        self.assertLess(pop.sum(), 2e7)
        self.assertTrue((sim.state.cells["subsistence"][pop > 0] == FORAGING).all())
        self.assertFalse((pop[~np.isfinite(sim.state.cells["arrival_year"])] > 0).any())


class DynamicsTests(unittest.TestCase):
    def test_migration_conserves_people(self):
        # With growth and famine switched off, moving people around never creates or loses anyone.
        cfg = DEFAULT.copy(forager_growth=0.0, farmer_growth=0.0, famine_halving_years=1e12,
                           domestication_rate=0.0)
        sim = Simulation(WORLD, seed=1, start_year=8000, systems=default_systems(), config=cfg)
        sim._setup()   # sea at today's level: no drowning
        before = sim.state.cells["population"].sum()
        sim.run(8100)
        self.assertAlmostEqual(sim.state.cells["population"].sum() / before, 1.0, places=3)

    def test_farming_emerges_and_spreads(self):
        cfg = DEFAULT.copy(domestication_rate=DEFAULT.domestication_rate * 20)
        sim = start(cfg)
        sim.run(2500)
        st = sim.state.stats
        self.assertGreater(st["domestications"], 0)
        self.assertGreater(st["farmed_land_share"], 0.01)
        mode = sim.state.cells["subsistence"]
        self.assertTrue((mode == FARMING).any())
        # Farmers live more densely than foragers.
        pop = sim.state.cells["population"]
        area = (WORLD.grid.km_per_px ** 2) * np.cos(np.radians(WORLD.grid.lat))
        dens = pop / area
        self.assertGreater(np.median(dens[mode == FARMING]), np.median(dens[mode == FORAGING]))

    def test_no_farming_before_holocene(self):
        sim = start(DEFAULT.copy(domestication_rate=1e-3))
        sim.run(DEFAULT.holocene_start_year)
        self.assertEqual(sim.state.stats["domestications"], 0)


if __name__ == "__main__":
    unittest.main()
