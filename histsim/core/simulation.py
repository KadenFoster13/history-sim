"""The yearly loop.

A simulation is a list of systems run in a fixed order every year. Each system may run
only every few years (`every`), which keeps slow-changing early millennia cheap.
Observers (the recorder, the live viewer) are told after each year.
"""
import time

from .calendar import label
from .rng import RngStreams
from .state import SimState


class System:
    """One force of history. Subclasses set `name` and `every`, and implement step()."""

    name = "system"
    every = 1                 # run every this many years

    def setup(self, state):
        """Called once before the first year (create layers, tables...)."""

    def step(self, state):
        """Advance this system by `every` years at state.year."""

    def due(self, year, start_year):
        return (year - start_year) % self.every == 0


class Simulation:
    def __init__(self, world, seed=0, start_year=0, systems=()):
        self.start_year = start_year
        self.state = SimState(world=world, seed=seed, year=start_year, rng=RngStreams(seed))
        self.systems = list(systems)
        self.observers = []
        self._ready = False

    def add_observer(self, observer):
        """`observer(state)` is called after every simulated year."""
        self.observers.append(observer)

    def _setup(self):
        for system in self.systems:
            system.setup(self.state)
        self._ready = True

    def step(self):
        """Simulate one year."""
        if not self._ready:
            self._setup()
        state = self.state
        state.stats = {}
        state.events = []
        for system in self.systems:
            if system.due(state.year, self.start_year):
                system.step(state)
        for observer in self.observers:
            observer(state)
        state.year += 1

    def run(self, until, progress_every=None, log=print):
        """Simulate up to (not including) year `until`."""
        if not hasattr(self, "_t0"):
            self._t0 = time.perf_counter()   # total time across repeated run() calls
        while self.state.year < until:
            self.step()
            if progress_every and self.state.year % progress_every == 0:
                elapsed = time.perf_counter() - self._t0
                log(f"{label(self.state.year)}  ({elapsed:,.0f} s)")
        return self.state
