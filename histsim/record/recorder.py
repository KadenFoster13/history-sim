"""Saving a run so it can be replayed, explored and graphed afterwards.

A run is a folder:
    meta.json            seeds, versions, settings, years covered
    keyframes/k_<year>.npz   every per-cell layer in full, every `keyframe_every` years
    changes/c_<year>.npz     for each layer: (year, cell, new value) for every cell that
                             changed, one file per `chunk_years` years
    events.jsonl         the chronicle: one event per line
    stats.csv            one row of statistics per year

Storing only what changed keeps runs small (layers change slowly), while keyframes keep
jumping to any year fast.
"""
import csv
import json
import os
import time

import numpy as np

FORMAT_VERSION = 1


class RunRecorder:
    def __init__(self, run_dir, sim, keyframe_every=100, chunk_years=100, settings=None):
        self.dir = run_dir
        self.sim = sim
        self.keyframe_every = keyframe_every
        self.chunk_years = chunk_years
        self.settings = settings or {}
        os.makedirs(os.path.join(run_dir, "keyframes"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "changes"), exist_ok=True)
        self._previous = {}          # layer -> flat copy as last recorded
        self._changes = {}           # layer -> lists of (years, cells, values) arrays
        self._chunk_start = None
        self._stats_rows = []
        self._stat_keys = []
        self._events = open(os.path.join(run_dir, "events.jsonl"), "w", encoding="utf-8")
        self._t0 = time.time()
        self.first_year = None
        self.last_year = None
        sim.add_observer(self)

    # Called by the simulation after every year.
    def __call__(self, state):
        year = state.year
        if self.first_year is None:
            self.first_year = year
            self._chunk_start = year
            self._write_meta(state, finished=False)
        if year - self._chunk_start >= self.chunk_years:
            self._flush_changes()
            self._chunk_start = year
        if (year - self.first_year) % self.keyframe_every == 0:
            self._keyframe(state)
        else:
            self._record_changes(state)
        for event in state.events:
            self._events.write(json.dumps(event, default=_jsonable) + "\n")
        row = {"year": year, **state.stats}
        for key in row:
            if key not in self._stat_keys:
                self._stat_keys.append(key)
        self._stats_rows.append(row)
        self.last_year = year

    def _keyframe(self, state):
        layers = {name: arr for name, arr in state.cells.items()}
        np.savez_compressed(os.path.join(self.dir, "keyframes", f"k_{state.year:07d}.npz"), **layers)
        self._previous = {name: arr.ravel().copy() for name, arr in layers.items()}

    def _record_changes(self, state):
        for name, arr in state.cells.items():
            flat = arr.ravel()
            before = self._previous.get(name)
            if before is None:  # a layer that appeared after the last keyframe
                before = np.zeros_like(flat)
            changed = np.flatnonzero(flat != before)
            if len(changed):
                store = self._changes.setdefault(name, ([], [], []))
                store[0].append(np.full(len(changed), state.year, np.int32))
                store[1].append(changed.astype(np.int32))
                store[2].append(flat[changed].copy())
            self._previous[name] = flat.copy()

    def _flush_changes(self):
        if self._chunk_start is None:
            return
        data = {}
        for name, (years, cells, values) in self._changes.items():
            if years:
                data[f"{name}__years"] = np.concatenate(years)
                data[f"{name}__cells"] = np.concatenate(cells)
                data[f"{name}__values"] = np.concatenate(values)
        np.savez_compressed(os.path.join(self.dir, "changes", f"c_{self._chunk_start:07d}.npz"), **data)
        self._changes = {}

    def _write_stats(self):
        with open(os.path.join(self.dir, "stats.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._stat_keys)
            writer.writeheader()
            writer.writerows(self._stats_rows)

    def _write_meta(self, state, finished):
        import worldgen

        from .. import __version__
        world = state.world
        meta = {
            "format_version": FORMAT_VERSION,
            "histsim_version": __version__,
            "worldgen_version": worldgen.__version__,
            "world_data_version": worldgen.DATA_VERSION,
            "world_seed": world.cfg.seed,
            "world_width": world.cfg.width,
            "sim_seed": state.seed,
            "first_year": self.first_year,
            "last_year": self.last_year,
            "keyframe_every": self.keyframe_every,
            "chunk_years": self.chunk_years,
            "shape": list(state.shape),
            "layers": {name: str(arr.dtype) for name, arr in state.cells.items()},
            "finished": finished,
            "seconds": round(time.time() - self._t0, 1),
            "settings": self.settings,
        }
        with open(os.path.join(self.dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1, default=_jsonable)

    def checkpoint(self):
        """Write everything recorded so far (safe to call mid-run, e.g. every few centuries)."""
        self._events.flush()
        self._write_stats()
        self._write_meta(self.sim.state, finished=False)

    def close(self):
        """Finish the run: write the last changes, statistics and metadata."""
        self._flush_changes()
        self._events.close()
        self._write_stats()
        self._write_meta(self.sim.state, finished=True)


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
