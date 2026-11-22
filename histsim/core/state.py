"""The simulation's state: everything that changes over history.

- `cells`: per-cell layers, numpy arrays shaped like the world map (rows, cols).
- `tables`: entities (cultures, states, religions, people, ...), added in later phases.
- `stats` / `events`: what systems report for the current year; the recorder saves them.
"""
from dataclasses import dataclass, field

import numpy as np

from .rng import RngStreams


@dataclass
class SimState:
    world: object                       # the worldgen World this history happens on
    seed: int
    year: int
    rng: RngStreams
    cells: dict = field(default_factory=dict)
    tables: dict = field(default_factory=dict)
    climate: object = None              # worldgen ClimateState for the current year
    stats: dict = field(default_factory=dict)
    events: list = field(default_factory=list)

    @property
    def shape(self):
        return self.world.elevation_m.shape

    def add_layer(self, name, dtype, fill=0):
        """Create a per-cell layer (if it doesn't exist yet) and return it."""
        if name not in self.cells:
            self.cells[name] = np.full(self.shape, fill, dtype=dtype)
        return self.cells[name]

    def event(self, kind, text, **data):
        """Report something that happened this year (goes into the chronicle)."""
        self.events.append({"year": self.year, "kind": kind, "text": text, **data})
