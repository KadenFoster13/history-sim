"""Reading a saved run: statistics, events, and the map at any year."""
import bisect
import csv
import glob
import json
import os

import numpy as np


class RunReader:
    def __init__(self, run_dir):
        self.dir = run_dir
        with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.shape = tuple(self.meta["shape"])
        self._keyframes = sorted(int(os.path.basename(p)[2:9])
                                 for p in glob.glob(os.path.join(run_dir, "keyframes", "k_*.npz")))
        self._chunks = sorted(int(os.path.basename(p)[2:9])
                              for p in glob.glob(os.path.join(run_dir, "changes", "c_*.npz")))
        self._chunk_cache = {}

    @property
    def first_year(self):
        return self.meta["first_year"]

    @property
    def last_year(self):
        return self.meta["last_year"]

    def stats(self):
        """Yearly statistics as {name: numpy array}, including 'year'."""
        path = os.path.join(self.dir, "stats.csv")
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        out = {}
        for key in (rows[0].keys() if rows else []):
            out[key] = np.array([float(r[key]) if r[key] not in ("", None) else np.nan for r in rows])
        return out

    def events(self, start=None, end=None, kind=None):
        """Events from the chronicle, optionally limited to a year range and kind."""
        out = []
        with open(os.path.join(self.dir, "events.jsonl"), encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                if start is not None and e["year"] < start:
                    continue
                if end is not None and e["year"] > end:
                    continue
                if kind is not None and e["kind"] != kind:
                    continue
                out.append(e)
        return out

    def layers_at(self, year, names=None):
        """Every per-cell layer (or just `names`) as it was at the end of `year`."""
        if not self._keyframes or year < self._keyframes[0]:
            raise ValueError(f"no recorded state for year {year}")
        k = self._keyframes[bisect.bisect_right(self._keyframes, year) - 1]
        with np.load(os.path.join(self.dir, "keyframes", f"k_{k:07d}.npz")) as data:
            layers = {name: data[name].ravel().copy() for name in data.files
                      if names is None or name in names}
        for chunk in self._chunks:
            if chunk + self.meta["chunk_years"] <= k or chunk > year:
                continue
            data = self._chunk(chunk)
            for name, flat in layers.items():
                if f"{name}__years" not in data:
                    continue
                years, cells, values = data[f"{name}__years"], data[f"{name}__cells"], data[f"{name}__values"]
                use = (years > k) & (years <= year)   # rows are already in year order
                flat[cells[use]] = values[use]
        return {name: flat.reshape(self.shape) for name, flat in layers.items()}

    def _chunk(self, start):
        if start not in self._chunk_cache:
            with np.load(os.path.join(self.dir, "changes", f"c_{start:07d}.npz")) as data:
                self._chunk_cache = {start: {name: data[name] for name in data.files}}
        return self._chunk_cache[start]
