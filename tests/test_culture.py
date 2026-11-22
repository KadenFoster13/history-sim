"""Checks for phase 2: languages, cultures, settlements, saved tables, the viewer.
Run: python -m unittest discover tests"""
import os
import tempfile
import unittest

import numpy as np
import worldgen

from histsim import Simulation, default_systems
from histsim.names import Language
from histsim.record import RunReader, RunRecorder

WORLD = worldgen.generate_world(worldgen.WorldConfig(seed=22, width=400))


def start(seed=1):
    sim = Simulation(WORLD, seed=seed, systems=default_systems())
    sim._setup()
    return sim


class LanguageTests(unittest.TestCase):
    def test_words_are_stable(self):
        a, b = Language(0, 1234), Language(0, 1234)
        self.assertEqual(a.word("river"), b.word("river"))
        self.assertNotEqual(a.word("river"), Language(0, 999).word("river"))

    def test_daughters_share_cognates(self):
        proto = Language(0, 77)
        daughter = Language(1, 78, parent=proto, rng=np.random.default_rng(1))
        self.assertEqual(daughter.family, proto.family)
        same_length = sum(abs(len(proto.word(m)) - len(daughter.word(m))) <= 1
                          for m in ("water", "sun", "big", "stone", "people", "hill", "red", "moon"))
        self.assertGreaterEqual(same_length, 5)    # mostly inherited words, a little changed

    def test_names_are_readable(self):
        rng = np.random.default_rng(3)
        for seed in range(20):
            lang = Language(0, seed)
            for _ in range(5):
                for name in (lang.people_name(rng), lang.place_name(rng, ("river", "hill"))):
                    self.assertTrue(3 <= len(name) <= 13, name)
                    self.assertTrue(name[0].isupper(), name)


class CultureTests(unittest.TestCase):
    def test_everyone_belongs_to_a_culture(self):
        sim = start()
        pop, cult = sim.state.cells["population"], sim.state.cells["culture"]
        self.assertTrue((cult[pop > 0] > 0).all())
        self.assertTrue((cult[pop <= 0] == 0).all())
        cultures = sim.state.tables["cultures"]
        self.assertGreater(sim.state.stats["cultures"], 20)
        self.assertGreater(sim.state.stats["language_families"], 3)
        names = [c["name"] for c in cultures[1:]]
        self.assertEqual(len(names), len(set(names)))   # every people has its own name

    def test_cultures_split_into_related_peoples(self):
        sim = start()
        sim.run(3000)
        cultures = sim.state.tables["cultures"]
        langs = sim.state.tables["languages"]
        daughters = [c for c in cultures[1:] if c["parent"] is not None]
        self.assertTrue(daughters)
        for c in daughters[:20]:
            parent = cultures[c["parent"]]
            self.assertEqual(c["family"], parent["family"])
            self.assertEqual(langs[c["language"]]["parent"], parent["language"])
        cult, pop = sim.state.cells["culture"], sim.state.cells["population"]
        self.assertTrue((cult[pop > 0] > 0).all())


class SettlementTests(unittest.TestCase):
    def test_towns_appear_with_names(self):
        sim = start()
        sim.run(8000)
        size = sim.state.cells["settlement"]
        ids = sim.state.cells["settlement_id"]
        places = sim.state.tables["settlements"]
        self.assertGreater(sim.state.stats["towns"], 0)
        town = size >= sim.state.config.town_min
        self.assertTrue((ids[town] > 0).all())
        for sid in np.unique(ids[town]):
            self.assertTrue(places[sid]["name"])
            self.assertIsNone(places[sid]["ended"])
        self.assertLess(sim.state.stats["urban_share"], 0.3)


class RecordingTests(unittest.TestCase):
    def test_tables_saved_and_replayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim = Simulation(WORLD, seed=2, systems=default_systems())
            rec = RunRecorder(tmp, sim, keyframe_every=50)
            sim.run(300)
            rec.close()
            run = RunReader(tmp)
            tables = run.tables()
            self.assertEqual(len(tables["cultures"]), len(sim.state.tables["cultures"]))
            self.assertEqual(tables["cultures"][1]["name"], sim.state.tables["cultures"][1]["name"])
            self.assertFalse(any(k.startswith("_") for c in tables["cultures"][1:] for k in c))
            layers = run.layers_at(299)
            self.assertTrue(np.array_equal(layers["culture"], sim.state.cells["culture"]))

    def test_viewer_draws(self):
        import matplotlib
        matplotlib.use("Agg")
        import worldgen.viewer as wv
        from histsim.viewer import HistoryViewer, LiveSource
        with tempfile.TemporaryDirectory() as tmp:
            wv.PREFERENCES_FILE = os.path.join(tmp, "prefs.json")
            sim = Simulation(WORLD, seed=3, systems=default_systems())
            viewer = HistoryViewer(WORLD, LiveSource(sim), "test")
            viewer.step()
            for mode in ("Cultures", "Language families", "Background only", "Arrival"):
                viewer._on_mode(mode)
            for bg in ("Biomes", "White", "Terrain"):
                viewer._on_background(bg)
            viewer.toggle_dark()
            self.assertTrue(os.path.exists(viewer.save(os.path.join(tmp, "v.png"))))
            self.assertIn("N", viewer._describe(200, 100) + "N")


if __name__ == "__main__":
    unittest.main()
