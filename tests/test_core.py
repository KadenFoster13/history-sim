"""Checks for the simulator core. Run: python -m unittest discover tests"""
import os
import tempfile
import unittest

import numpy as np
import worldgen

from histsim import RunReader, RunRecorder, Simulation, System, default_systems, label, real_label
from histsim.core.rng import RngStreams

WORLD = worldgen.generate_world(worldgen.WorldConfig(seed=7, width=400))


def run_history(seed, years, out=None, observer=None):
    sim = Simulation(WORLD, seed=seed, systems=default_systems())
    recorder = RunRecorder(out, sim, keyframe_every=50, chunk_years=40) if out else None
    if observer:
        sim.add_observer(observer)
    sim.run(years)
    if recorder:
        recorder.close()
    return sim


class CalendarTests(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(real_label(0), "10000 BCE")      # year 0 is 10,000 BCE
        self.assertEqual(real_label(9_999), "1 BCE")
        self.assertEqual(real_label(10_000), "1 CE")      # no year 0 between BCE and CE
        self.assertEqual(real_label(12_024), "2025 CE")
        self.assertIn("year 12,024", label(12_024))


class RngTests(unittest.TestCase):
    def test_streams_are_reproducible_and_independent(self):
        a, b = RngStreams(5), RngStreams(5)
        a.get("war").random(10)                      # using one stream...
        self.assertEqual(a.get("religion").random(), b.get("religion").random())  # ...doesn't shift another
        self.assertNotEqual(RngStreams(6).get("religion").random(), RngStreams(5).get("religion").random())


class SimulationTests(unittest.TestCase):
    def test_systems_run_on_their_schedule(self):
        class Counter(System):
            name, every = "counter", 10

            def setup(self, state):
                self.calls = []

            def step(self, state):
                self.calls.append(state.year)

        counter = Counter()
        Simulation(WORLD, seed=1, systems=[counter]).run(35)
        self.assertEqual(counter.calls, [0, 10, 20, 30])

    def test_same_seed_same_history(self):
        a = run_history(3, 120).state
        b = run_history(3, 120).state
        for name in a.cells:
            np.testing.assert_array_equal(a.cells[name], b.cells[name])

    def test_sea_rises_and_floods_land_bridges(self):
        early = []
        sim = run_history(1, 2, observer=lambda s: early.append(s.cells["exposed_shelf"].sum()))
        late = Simulation(WORLD, seed=1, start_year=8000, systems=default_systems())
        late.run(8001)
        self.assertGreater(early[0], 0)
        self.assertEqual(late.state.cells["exposed_shelf"].sum(), 0)

    def test_timings_and_yearly_climate(self):
        temps = []
        sim = run_history(1, 12, observer=lambda s: temps.append(s.stats["global_temperature_c"]))
        names = {s.name for s in sim.systems} | {"observers"}
        self.assertEqual(set(sim.timings), names)
        self.assertTrue(all(t >= 0 for t in sim.timings.values()))
        self.assertIn("seconds_farming", sim.state.stats)
        # The climate's maps are rebuilt every few years, its global numbers every year.
        self.assertEqual(len(set(temps)), 12)


class RecordingTests(unittest.TestCase):
    def test_replay_matches_live_state(self):
        live = {}

        def remember(state):
            if state.year in (0, 37, 50, 99, 149):
                live[state.year] = {k: v.copy() for k, v in state.cells.items()
                                    if state.record_modes.get(k) != "none"}
                self.modes = dict(state.record_modes)

        with tempfile.TemporaryDirectory() as out:
            run_history(2, 150, out=out, observer=remember)
            run = RunReader(out)
            for year, layers in live.items():
                replay = run.layers_at(year)
                for name, arr in layers.items():
                    if self.modes[name] == "snapshot" and year % 50:
                        continue  # snapshot-only layers (population) exist at snapshot years
                    np.testing.assert_array_equal(replay[name], arr, err_msg=f"{name} at year {year}")
            stats = run.stats()
            self.assertEqual(len(stats["year"]), 150)
            self.assertTrue(run.meta["finished"])
            self.assertTrue(os.path.exists(os.path.join(out, "events.jsonl")))

    def test_graphs(self):
        from histsim.analysis import plot_run
        with tempfile.TemporaryDirectory() as out:
            run_history(2, 60, out=out)
            path = plot_run(out)
            self.assertTrue(os.path.getsize(path) > 10_000)


if __name__ == "__main__":
    unittest.main()
