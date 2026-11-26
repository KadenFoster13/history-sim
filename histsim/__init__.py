"""histsim: a history simulator for worldgen worlds. See PLAN.md.

    from histsim import load_world, Simulation, default_systems, RunRecorder
    world = load_world(seed=31)
    sim = Simulation(world, seed=1, systems=default_systems())
    recorder = RunRecorder("runs/example", sim)
    sim.run(until=12_000)
    recorder.close()
"""
__version__ = "0.9.0"

import os as _os

# The compiled per-cell kernels run on several cores (numba, OpenMP). Idle OpenMP threads
# would otherwise spin between kernels and slow down the numpy work in between.
_os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
_os.environ.setdefault("KMP_BLOCKTIME", "0")

from .core import Simulation, System, label, real_label, real_year, sim_year
from .record import RunReader, RunRecorder
from .systems import default_systems
from .world import load_world
