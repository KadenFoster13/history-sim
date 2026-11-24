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

    def test_river_water_and_frontiers(self):
        sim = start()
        farming = sim.state.tables["farming"]
        crops = farming.is_crop
        # Floodplains: crops grow on the flood where rain alone wouldn't do.
        flood = WORLD.floodplain >= 0.4
        self.assertTrue((farming.suit[crops] >= farming.suit_dry[crops] - 1e-6).all())
        if flood.any():
            gain = farming.suit[crops][:, flood].max(axis=0) - farming.suit_dry[crops][:, flood].max(axis=0)
            self.assertGreaterEqual(gain.max(), 0)
        dry_river = flood & (WORLD.precip_mm < 300)
        if dry_river.any():
            self.assertGreater(farming.suit[crops][:, dry_river].max(), 0.3)
        # Canals (a later technology) water more land.
        before = farming.suit[crops].sum()
        farming._apply_water(DEFAULT.copy(canal_irrigation=1.0))
        self.assertGreater(farming.suit[crops].sum(), before)
        farming._apply_water(DEFAULT)
        # Foragers who live well hold out against farming; poor ones take it up readily.
        i = int(np.flatnonzero(crops)[0])
        none = np.zeros(LAND.shape, bool)
        rich = farming._welcome(DEFAULT, i, np.ones(LAND.shape, np.float32), np.full(LAND.shape, 1e3), none)
        poor = farming._welcome(DEFAULT, i, np.ones(LAND.shape, np.float32), np.full(LAND.shape, 1e-3), none)
        good = farming.suit[i] > 0.5
        self.assertTrue(np.allclose(rich[good], DEFAULT.frontier_floor))
        self.assertTrue(np.allclose(poor[good], 1.0))

    def test_rules_scale_with_cell_size(self):
        from histsim.systems.geo import cell_scale, cells_for
        scale = cell_scale(WORLD, DEFAULT)              # a 400-px world: cells twice as wide
        self.assertAlmostEqual(scale, 2.0)
        self.assertEqual(cells_for(8, scale), 2)        # 8 cells of 50 km = 2 of 100 km
        self.assertEqual(cells_for(8, 0.5), 32)
        self.assertEqual(cells_for(2, scale), 1)        # never below one cell
        # River and coast bands keep their width in km: wider on fine maps, part of a cell on coarse.
        from histsim.systems.geo import band
        line = np.zeros((5, 5), bool)
        line[2] = True
        self.assertEqual(int(band(line, 0.5).sum()), 15)
        self.assertTrue(np.allclose(band(line, 2.0)[2], 0.5))
        self.assertIs(band(line, 1.0), line)
        # Migration stays conservative with the extra rounds on finer maps.
        cfg = DEFAULT.copy(forager_growth=0.0, farmer_growth=0.0, famine_halving_years=1e12,
                           domestication_rate=0.0, reference_width=1600)
        sim = Simulation(WORLD, seed=1, start_year=8000, systems=default_systems(), config=cfg)
        sim._setup()
        before = sim.state.cells["population"].sum()
        sim.run(8050)
        self.assertAlmostEqual(sim.state.cells["population"].sum() / before, 1.0, places=3)

    def test_intensification(self):
        # Crowded farmland is worked harder (it feeds more); emptied land lets it go.
        sim = start()
        st, cfg = sim.state, DEFAULT
        farming = st.tables["farming"]
        ceiling = farming.ceiling(st)
        self.assertTrue((ceiling >= 1).all())
        self.assertLessEqual(float(ceiling.max()), cfg.irrigated_ceiling * cfg.paddy_ceiling + 1e-3)
        cells = LAND & (farming.arable_eff > 0.3)
        st.cells["subsistence"][:] = np.where(cells, FARMING, st.cells["subsistence"])
        area = farming.area
        cap = np.where(cells, 4.0 * area, 0).astype(np.float32)       # 4 people per km2
        st.cells["capacity"] = cap
        intensity = st.cells["intensity"]
        st.cells["population"][:] = np.where(cells, 0.95 * cap, 0)   # dense and crowded
        for _ in range(int(3000 / cfg.step_years)):              # 3,000 years
            farming._intensify(st)
        crowded = float(intensity[cells].mean())
        self.assertGreater(crowded, 0.05)
        self.assertLessEqual(float(intensity.max()), 1.0)
        self.assertEqual(float(intensity[~cells].max()), 0.0)
        st.cells["population"][:] = np.where(cells, 0.1 * cap, 0)    # emptied
        for _ in range(100):
            farming._intensify(st)
        self.assertLess(float(intensity[cells].mean()), crowded / 2)
        # Thin farming stays shifting: few people per km2 barely intensify.
        intensity[:] = 0
        st.cells["population"][:] = np.where(cells, 0.95 * cap / 20, 0)
        st.cells["capacity"] = cap / 20
        for _ in range(int(3000 / cfg.step_years)):
            farming._intensify(st)
        self.assertLess(float(intensity[cells].mean()), crowded / 3)
        # Intensity multiplies what farmland feeds.
        st.cells["population"][:] = np.where(cells, 0.95 * cap, 0)
        forage = st.cells["forager_capacity"]
        before, _ = farming.capacity(st, forage, LAND)
        intensity[cells] = 1.0
        after, _ = farming.capacity(st, forage, LAND)
        self.assertGreaterEqual(float(after.sum()), float(before.sum()))

    def test_no_farming_before_holocene(self):
        sim = start(DEFAULT.copy(domestication_rate=1e-3))
        sim.run(DEFAULT.holocene_start_year)
        self.assertEqual(sim.state.stats["domestications"], 0)


if __name__ == "__main__":
    unittest.main()
