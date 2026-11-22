"""Reproducible random numbers: one independent stream per system.

Each system asks for its own stream by name, so adding a new system (or new random
draws in one system) never changes the random numbers any other system gets. The same
simulation seed always replays the same history.
"""
import zlib

import numpy as np


class RngStreams:
    def __init__(self, seed):
        self.seed = int(seed)
        self._streams = {}

    def get(self, name):
        """The random generator for `name` (created on first use, then kept)."""
        if name not in self._streams:
            key = zlib.crc32(name.encode("utf-8"))  # stable across runs, unlike hash()
            self._streams[name] = np.random.default_rng((self.seed, key))
        return self._streams[name]

    def state(self):
        """Snapshot of every stream's position (for saving and resuming runs)."""
        return {name: g.bit_generator.state for name, g in self._streams.items()}

    def restore(self, state):
        for name, s in state.items():
            self.get(name).bit_generator.state = s
