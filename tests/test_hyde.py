"""Earth's 1300 CE targets and the HYDE reader (no HYDE files needed: a tiny grid is made here)."""
import os
import tempfile
import unittest
import zipfile

import numpy as np

from histsim.analysis.earth_targets import REGIONS, region_owner, targets_1300, world_1300
from histsim.analysis.hyde import _regrid, blended_map, load_hyde, score


class TargetTests(unittest.TestCase):
    def test_targets_are_medians_within_their_range(self):
        t = targets_1300()
        self.assertEqual(set(t), set(REGIONS))
        for r in t.values():
            self.assertLessEqual(r["lo"], r["target"])
            self.assertLessEqual(r["target"], r["hi"])
        self.assertAlmostEqual(t["India"]["target"], (88.0 + 94.5) / 2)     # six sources: middle two
        self.assertEqual(world_1300()["target"], 400.0)

    def test_regions_do_not_overlap(self):
        owner = region_owner((400, 800))
        d = 360 / 800
        r, c = int((90 - 36.5) / d), int((20 + 180) / d)                   # Sicily strait: Europe and N Africa boxes
        self.assertEqual(owner[r, c], "Europe (W+C)")                       # the first region listed owns it
        self.assertEqual(owner[int((90 - 0) / d), int((-150 + 180) / d)], "")   # mid-Pacific


class HydeTests(unittest.TestCase):
    def _grid(self):
        """A 1-degree world grid (as HYDE's layout) with people in two cells."""
        data = np.zeros((180, 360), np.float32)
        data[90 - 26, 180 + 83] = 1e6          # 25.5 N, 83.5 E: the Ganges
        data[90 - 49, 180 + 2] = 5e5           # 48.5 N, 2.5 E: Paris
        return data

    def test_regrid_keeps_people_and_places(self):
        out = _regrid(self._grid(), 1.0, 800)
        self.assertAlmostEqual(out.sum(), 1.5e6)
        d = 360 / 800
        self.assertEqual(out[int((90 - 25.5) / d), int((83.5 + 180) / d)], 1e6)

    def test_load_from_zip_and_score(self):
        data = self._grid()
        header = "ncols 360\nnrows 180\nxllcorner -180\nyllcorner -90\ncellsize 1.0\nNODATA_value -9999\n"
        body = "\n".join(" ".join(f"{v:g}" for v in row) for row in data)
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(os.path.join(tmp, "1300AD_pop.zip"), "w") as z:
                z.writestr("popc_1300AD.asc", header + body)
            grid = load_hyde(tmp, 1300, width=800, cache_dir=os.path.join(tmp, "cache"))
            self.assertAlmostEqual(float(grid.sum()), 1.5e6, places=0)
            self.assertTrue(os.path.exists(os.path.join(tmp, "cache", "hyde_1300AD_800.npy")))
        blend = blended_map(grid)
        owner = region_owner(grid.shape)
        self.assertAlmostEqual(blend[owner == "India"].sum() / 1e6, targets_1300()["India"]["target"], places=3)
        lines, nums = score(blend, grid)
        self.assertGreaterEqual(nums["regions_in_range"], 2)                 # India and Europe hit their targets exactly
        self.assertTrue(any(line.startswith("India") for line in lines))


if __name__ == "__main__":
    unittest.main()
