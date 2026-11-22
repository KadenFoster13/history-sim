# history-sim

A historical simulator built on worlds from the separate
[`worldgen`](https://github.com/KadenFoster13/world_generator) project.

## How the two projects fit together

- **worldgen**: everything physical (terrain, climate, biomes, lakes, rivers).
- **history-sim**: everything human (people, settlements, borders, crossings, ports, history).

The simulator is written against **worldgen 1.2.0**: the world data it reads is listed
in worldgen's DATA.md, and `main.py` checks the data version when it starts.

## Setup

Needs Python 3.10 or newer and git.

```
python -m venv venv
venv\Scripts\activate            (Windows)   or   source venv/bin/activate   (macOS/Linux)
pip install -r requirements.txt
```

This installs worldgen 1.2.0 straight from GitHub.

### Working on both projects at once

If you also have the generator's folder next to this one (`../world-generator`), install
it as a link instead, so changes to the generator apply here immediately:

```
pip install -r requirements-dev.txt
```

## Run

```
python -m histsim run --world 31 --seed 1          simulate a history and save it to runs/
python -m histsim graphs runs/world31_seed1        draw its graphs (graphs.png in the run folder)
python -m histsim maps runs/world31_seed1          maps of where and how people live over time
python -m histsim info runs/world31_seed1          summarise it and show the first events
python -m histsim watch --world 31 --seed 1        watch a history live in the viewer
python -m histsim replay runs/world31_seed1        replay a saved run in the viewer
```

(`python main.py ...` does the same.) `--world` picks the world, `--seed` the history:
the same pair always gives the same history. Years count from year 0 (10,000 BCE), so
year 10,000 is 1 CE and `--until 12000` (the default) runs to 2000 CE.

`--peopling sparse|some|most|all` sets how much of the world people have reached by
10,000 BCE: `sparse` (short sea hops only, many empty lands), `some` (the default: rare
ancient voyages reach large lands within 1,500 km), `most` (voyages up to 2,500 km) or
`all` (every sizeable landmass). The levels are defined in `histsim/config.py`.

`maps --kind` draws `people` (how they live), `density`, `cultures`, `families`
(language families) or `arrival` (when people first got there); towns show as dots.
`graphs` and `maps` take `--dark` or `--light` (default: the viewers' remembered setting);
`maps --background terrain|biomes|white` picks what the maps are drawn over (default:
the history viewer's last choice). Farmers are crimson, herders orange, foragers gold.

### The viewer

`watch` simulates while you watch (add `--save runs/<name>` to keep the run for replay
and graphs); `replay` plays back a saved run with a year slider. Both show the map in
several modes (people, density, cultures, language families, background only, arrival), towns
and cities with their names, the largest peoples and towns, and recent events.

| Control | Does |
|---|---|
| Play / Pause, `Space` | Run history forward |
| Slower / Faster, `-` / `+` | Years per frame (1 to 500) |
| Step, `Right` | One jump forward |
| Year slider (replay) | Jump to any year |
| Map buttons | People, density, cultures, language families, background only, arrival |
| Background buttons | Draw the maps over terrain, biomes or plain white (remembered) |
| Scroll, drag, `R` | Zoom around the cursor, move (sideways turns the map; it wraps), reset |
| `C` | Cut the map where it crosses the least land (done automatically at the start) |
| Dark / Light, `D` | Dark mode (the same remembered setting as the world viewer) |
| `S` | Save the view as a PNG in `renders/` |
| Hover | People, way of life, culture and language family, town |

To browse worlds visually, use the generator's viewer: `python -m worldgen`.

## How it works

See [PLAN.md](PLAN.md) for the design and the phases. In short:

| Folder | Contents |
|---|---|
| `histsim/core` | The world state, the yearly loop, random streams, the calendar |
| `histsim/config.py` | Every tunable setting for the simulator, with comments |
| `histsim/systems` | One module per force of history: environment, population (and the peopling of the world before year 0), farming, cultures, settlements |
| `histsim/names` | Generated languages: sound systems, sound changes between related languages, names |
| `histsim/viewer` | The live and replay viewer |
| `histsim/record` | Saving runs (snapshots + changes, events, statistics) and reading them back |
| `histsim/analysis` | Graphs (with Earth's real population for comparison) and maps |
| `tests` | `python -m unittest discover tests` |

A saved run (`runs/<name>/`) holds `meta.json`, full snapshots every 100 years in
`keyframes/`, year-by-year changes in `changes/`, the chronicle in `events.jsonl` and
yearly statistics in `stats.csv` (which opens in a spreadsheet), and every culture,
language and town that ever existed (with founding and ending years) in `tables.json`.
