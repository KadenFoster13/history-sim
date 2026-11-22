"""Command line:

    python -m histsim run --world 31 --seed 1 --until 12000 --out runs/w31
    python -m histsim run --world 31 --peopling all     (sparse / some / most / all)
    python -m histsim graphs runs/w31
    python -m histsim maps runs/w31 --kind people
    python -m histsim info runs/w31
"""
import argparse
import os
import time

from . import __version__
from .analysis import plot_run
from .analysis.maps import MAP_KINDS, plot_maps
from .config import PEOPLING_LEVELS, peopling
from .core import Simulation, label
from .record import RunReader, RunRecorder
from .systems import default_systems
from .world import load_world


def cmd_run(args):
    out = args.out or os.path.join("runs", f"world{args.world}_seed{args.seed}")
    print(f"histsim {__version__}: world {args.world} ({args.width} px), history seed {args.seed}")
    t0 = time.time()
    world = load_world(args.world, args.width)
    print(f"world ready in {time.time() - t0:.1f}s; simulating {label(args.start)} to {label(args.until)}")
    sim = Simulation(world, seed=args.seed, start_year=args.start, systems=default_systems(),
                     config=peopling(args.peopling))
    recorder = RunRecorder(out, sim, keyframe_every=args.keyframe_every,
                           settings={"until": args.until, "start": args.start, "peopling": args.peopling})
    try:
        while sim.state.year < args.until:
            sim.run(min(args.until, sim.state.year + args.checkpoint_every),
                    progress_every=args.progress_every)
            recorder.checkpoint()
    finally:
        recorder.close()
    print(f"saved to {out} ({time.time() - t0:.0f}s)")
    if not args.no_graphs:
        print(f"graphs: {plot_run(out)}")
        print(f"maps: {plot_maps(out, kind='people', world=world)}")


def cmd_graphs(args):
    print(plot_run(args.run, out=args.out, smooth_years=args.smooth))


def cmd_maps(args):
    years = [int(y) for y in args.years.split(",")] if args.years else None
    print(plot_maps(args.run, years=years, kind=args.kind, out=args.out))


def cmd_info(args):
    run = RunReader(args.run)
    m = run.meta
    print(f"world {m['world_seed']} ({m['world_width']} px), history seed {m['sim_seed']}")
    print(f"{label(m['first_year'])} to {label(m['last_year'])}, finished: {m['finished']}, "
          f"{m['seconds']}s, worldgen {m['worldgen_version']}, histsim {m['histsim_version']}")
    events = run.events()
    print(f"{len(events)} events; first few:")
    for e in events[:10]:
        print(f"  {label(e['year'])}: {e['text']}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="histsim", description="History simulator for worldgen worlds")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="simulate a history and save it")
    run.add_argument("--world", type=int, default=1, help="world seed")
    run.add_argument("--width", type=int, default=800, help="world map width (800 recommended)")
    run.add_argument("--seed", type=int, default=1, help="history seed (same seeds = same history)")
    run.add_argument("--start", type=int, default=0, help="first year (0 = 10,000 BCE)")
    run.add_argument("--until", type=int, default=12_000, help="stop before this year (12,000 = 2001 CE)")
    run.add_argument("--peopling", choices=list(PEOPLING_LEVELS), default="some",
                     help="how much of the world people reach by 10,000 BCE (default: some)")
    run.add_argument("--out", help="run folder (default: runs/world<W>_seed<S>)")
    run.add_argument("--keyframe-every", type=int, default=100, help="years between full snapshots")
    run.add_argument("--checkpoint-every", type=int, default=500, help="years between saves to disk")
    run.add_argument("--progress-every", type=int, default=1000, help="years between progress messages")
    run.add_argument("--no-graphs", action="store_true", help="don't draw graphs at the end")
    run.set_defaults(func=cmd_run)

    graphs = sub.add_parser("graphs", help="draw graphs of a saved run")
    graphs.add_argument("run")
    graphs.add_argument("--out")
    graphs.add_argument("--smooth", type=int, default=0, help="running average over this many years")
    graphs.set_defaults(func=cmd_graphs)

    maps = sub.add_parser("maps", help="draw maps of a saved run at several years")
    maps.add_argument("run")
    maps.add_argument("--kind", choices=MAP_KINDS, default="people",
                      help="people (how they live), density, or arrival (first settlement)")
    maps.add_argument("--years", help="comma-separated simulation years (default: six across the run)")
    maps.add_argument("--out")
    maps.set_defaults(func=cmd_maps)

    info = sub.add_parser("info", help="summarise a saved run")
    info.add_argument("run")
    info.set_defaults(func=cmd_info)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
