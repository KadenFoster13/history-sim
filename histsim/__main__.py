"""Command line:

    python -m histsim run --world 31 --seed 1 --until 12000 --out runs/w31
    python -m histsim run --world 31 --peopling all     (sparse / some / most / all)
    python -m histsim graphs runs/w31
    python -m histsim maps runs/w31 --kind people
    python -m histsim info runs/w31
    python -m histsim watch --world 31 --seed 1          watch a history live (add --save to keep it)
    python -m histsim replay runs/w31                    replay a saved run
    python -m histsim bench --world 31 --until 10001     time a run, system by system
"""
import argparse
import os
import time

from . import __version__
from .analysis import plot_run
from .analysis.maps import BACKGROUNDS, MAP_KINDS, plot_maps
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
        for kind in ("people", "cultures"):
            print(f"maps: {plot_maps(out, kind=kind, world=world)}")


def cmd_bench(args):
    """Time a run (one core, nothing saved unless --save): speed by millennium, time by
    system, and a fingerprint of the history (stats and events) so a change made only for
    speed can be checked to give exactly the same history."""
    import hashlib
    import numbers
    import platform

    print(f"histsim {__version__} bench: world {args.world} ({args.width} px), history seed {args.seed}, "
          f"{label(args.start)} to {label(args.until)}; {platform.processor() or platform.machine()}")
    t0 = time.perf_counter()
    world = load_world(args.world, args.width)
    t_world = time.perf_counter() - t0
    sim = Simulation(world, seed=args.seed, start_year=args.start, systems=default_systems(),
                     config=peopling(args.peopling))
    t0 = time.perf_counter()
    sim._setup()
    t_setup = time.perf_counter() - t0
    fingerprint = hashlib.sha256()

    def fingerprint_year(state):
        # Floats to 6 figures: float32 sums can differ in the last bit from run to run
        # (memory alignment), without changing the history.
        def rounded(v):
            return f"{v:.6g}" if isinstance(v, numbers.Real) and not isinstance(v, numbers.Integral) else v

        stats = sorted((k, rounded(v)) for k, v in state.stats.items() if not k.startswith("seconds_"))
        fingerprint.update(repr((state.year, stats, state.events)).encode())

    sim.add_observer(fingerprint_year)
    recorder = None
    if args.save:
        recorder = RunRecorder(args.save, sim, settings={"until": args.until, "start": args.start,
                                                         "peopling": args.peopling, "bench": True})
    print(f"world loaded in {t_world:.1f} s, set up (peopling) in {t_setup:.1f} s")
    def era(year):   # "year 0 (10000 BCE)" -> "10000 BCE"
        text = label(year)
        return text[text.find("(") + 1:-1] if "(" in text else text

    print(f"{'years':>21} {'seconds':>8} {'years/s':>8}")
    t0 = time.perf_counter()
    total = 0.0
    try:
        while sim.state.year < args.until:
            first = sim.state.year
            last = min(args.until, (first // args.every + 1) * args.every)
            t = time.perf_counter()
            sim.run(last)
            spent = time.perf_counter() - t
            total += spent
            print(f"{era(first):>10} - {era(last):<9}{spent:8.1f} {(last - first) / spent:8.1f}")
            if recorder:
                recorder.checkpoint()
    finally:
        if recorder:
            recorder.close()
    years = sim.state.year - args.start
    print(f"{'total':>21} {total:8.1f} {years / total:8.1f}")
    print("time by system:")
    for name, seconds in sorted(sim.timings.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<14}{seconds:8.1f} s  {100 * seconds / total:5.1f}%")
    print(f"history fingerprint: {fingerprint.hexdigest()[:16]}")


def cmd_graphs(args):
    print(plot_run(args.run, out=args.out, smooth_years=args.smooth, dark=args.dark))


def cmd_maps(args):
    years = [int(y) for y in args.years.split(",")] if args.years else None
    print(plot_maps(args.run, years=years, kind=args.kind, out=args.out, dark=args.dark,
                    background=args.background))


def cmd_watch(args):
    from .viewer import watch
    world = load_world(args.world, args.width)
    sim = Simulation(world, seed=args.seed, start_year=args.start, systems=default_systems(),
                     config=peopling(args.peopling))
    recorder = None
    if args.save:
        recorder = RunRecorder(args.save, sim, keyframe_every=100,
                               settings={"start": args.start, "peopling": args.peopling, "watched": True})
    print("preparing the world (peopling, cultures)...")
    watch(world, sim, title=f"World {args.world}, history seed {args.seed}", recorder=recorder)
    if args.save:
        print(f"saved to {args.save}")


def cmd_replay(args):
    from .viewer import replay
    run = RunReader(args.run)
    world = load_world(run.meta["world_seed"], run.meta["world_width"])
    replay(world, run, title=f"World {run.meta['world_seed']}, history seed {run.meta['sim_seed']}")


def _dark_flags(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dark", dest="dark", action="store_true", default=None, help="dark background")
    group.add_argument("--light", dest="dark", action="store_false", help="light background")


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
    run.add_argument("--width", type=int, default=800, help="world map width (800 is the size the rules were tuned on; other sizes are rescaled to match)")
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
    _dark_flags(graphs)
    graphs.set_defaults(func=cmd_graphs)

    maps = sub.add_parser("maps", help="draw maps of a saved run at several years")
    maps.add_argument("run")
    maps.add_argument("--kind", choices=MAP_KINDS, default="people",
                      help="people (how they live), density, cultures, families (language families) "
                           "or arrival (first settlement)")
    maps.add_argument("--years", help="comma-separated simulation years (default: six across the run)")
    maps.add_argument("--out")
    maps.add_argument("--background", choices=BACKGROUNDS,
                      help="what the maps are drawn on (default: the viewer's last choice, else terrain)")
    _dark_flags(maps)
    maps.set_defaults(func=cmd_maps)

    watch = sub.add_parser("watch", help="watch a history live as it is simulated")
    watch.add_argument("--world", type=int, default=1, help="world seed")
    watch.add_argument("--width", type=int, default=800, help="world map width")
    watch.add_argument("--seed", type=int, default=1, help="history seed")
    watch.add_argument("--start", type=int, default=0, help="first year (0 = 10,000 BCE)")
    watch.add_argument("--peopling", choices=list(PEOPLING_LEVELS), default="some",
                       help="how much of the world people reach by 10,000 BCE")
    watch.add_argument("--save", help="also save the run to this folder (for replay and graphs)")
    watch.set_defaults(func=cmd_watch)

    replay = sub.add_parser("replay", help="replay a saved run in the viewer")
    replay.add_argument("run")
    replay.set_defaults(func=cmd_replay)

    bench = sub.add_parser("bench", help="time a run, system by system (for optimization)")
    bench.add_argument("--world", type=int, default=31, help="world seed")
    bench.add_argument("--width", type=int, default=800, help="world map width")
    bench.add_argument("--seed", type=int, default=1, help="history seed")
    bench.add_argument("--start", type=int, default=0, help="first year")
    bench.add_argument("--until", type=int, default=10_001, help="stop before this year (10,001 = 2 CE)")
    bench.add_argument("--peopling", choices=list(PEOPLING_LEVELS), default="some")
    bench.add_argument("--every", type=int, default=1000, help="years per line of the speed table")
    bench.add_argument("--save", help="also save the run (adds the recorder's time)")
    bench.set_defaults(func=cmd_bench)

    info = sub.add_parser("info", help="summarise a saved run")
    info.add_argument("run")
    info.set_defaults(func=cmd_info)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
