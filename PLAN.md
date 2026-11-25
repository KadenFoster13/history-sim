# History simulator: design plan

A simulation of a world's history from the first farmers to the near future, on worlds
made by [`worldgen`](https://github.com/KadenFoster13/world_generator). This document is
the plan: what it simulates, how it's built, how we check it's realistic, and the order
we build it in. It will change as we go; decisions are recorded in the log at the end.

---

## 1. Goals and principles

- **Realistic mechanisms, emergent history.** The world is not Earth, so the simulator
  can't replay Earth's events. Instead it uses the forces that shaped real history
  (geography, farming, disease, trade, states, religion, technology...) and lets history
  emerge. "Accurate" means the *patterns* match real history: how populations grow, how
  long empires last, how fast ideas spread, when cities and industry appear.
- **Real physics, invented culture.** Technologies are real (bronze needs copper and
  tin, industry needs coal), but discovered in flexible orders; religions, ideologies,
  languages, styles and names are generated for each world.
- **Watch it live, keep the results.** Runs can be watched as they happen and are saved
  so the whole history can be replayed, explored and graphed afterwards.
- **Same seed, same history.** Every run is reproducible from its world seed and
  simulation seed.
- **Observer first, playable later.** Every society decides through one shared decision
  step; a human player is later just another way of making those decisions.
- **Built in phases.** Every phase ends with something that runs and can be watched.

## 2. Decisions so far

| Question | Decision |
|---|---|
| How it's experienced | Watch live, save the run, replay and graph afterwards; player control later |
| Unit of simulation | Map cells of the 800 px world (~50 km); about 70,000 land cells |
| Movement | 4 directions between cells; rivers on cell edges (borders, crossings) |
| Level of detail | Societies in every cell, plus key figures (rulers, dynasties, prophets, generals, explorers) |
| Technology and culture | Blend: real technologies with flexible, resource-driven paths; culture-specific forms and names; invented religions, ideologies and governments; speculative future tech |
| Languages | Every culture has a generated language that drifts and splits; it names places, people and states |
| Start | Year 0 = 10,000 BCE; hunter-gatherers placed by a peopling pre-pass |
| Run time | Hours are acceptable while building; as a game it must be fast, so speed is planned work (section 4, "Performance and optimization") |
| End | No fixed end; past the present day into the near future |

## 3. Calendar

The simulation counts years from its start: **year 0 is 10,000 BCE**. As there is no
year 0 between BCE and CE, year 9,999 is 1 BCE, year 10,000 is 1 CE and 2025 CE is year
12,024. Each culture may also keep
its own era names in the chronicle (for example "year 312 of the Kessan kings").

When comparing against real history (astronomical years, 0 = 1 BCE), `real_year = sim_year - 9,999`.

## 4. Architecture

```
history-sim/
  histsim/
    core/        world state (arrays per cell, tables of entities), the yearly loop, RNG
    systems/     one module per force: demography, farming, culture, states, war, ...
    agents/      decision-making: the shared decision step and its AI (later: player)
    record/      saving runs: keyframes, yearly changes, events, statistics
    names/       generated languages and names
    viewer/      live view and replay (built on the worldgen viewer)
    analysis/    graphs, reports, calibration checks
  tests/
  PLAN.md
```

### State
- **Per cell** (numpy arrays, one value per land cell): population, subsistence (foraging,
  herding, farming), crops and animals in use, settlement size, culture, language,
  religion, owning state, local technology level, wealth, unrest, disease load...
- **Entities** (tables): cultures, languages, religions, states, settlements and cities,
  key figures, trade routes, technologies, wars, treaties.
- **Edges** (from worldgen 1.0): rivers with flow, fords, navigability and rapids, so
  rivers can be borders, barriers and highways.

### The yearly loop
Each simulated year runs the systems in a fixed order (environment, demography,
economy, culture, politics, conflict, events). Expensive systems can run less often
(every 5 or 10 years) when nothing needs yearly detail, especially in the early,
slow-moving millennia.

### Decisions
States (and later other actors) choose actions through one interface:
`decide(actor, situation) -> actions`. The built-in AI implements it; a human player
will later implement the same interface through the viewer.

### Determinism
One seeded random stream per system, so adding a feature to one system doesn't change
the random rolls of the others, and runs replay exactly.

### Performance and optimization (plan added 2026-11-24)
About 70,000 land cells x 12,000+ years. Systems work on whole arrays at once (numpy);
loops that can't be vectorised use numba, as the generator's river routing does. For
this to become a game it has to be as fast as it can reasonably be, so speed is planned
work with targets, not something left to the end.

**Where it stands (histsim 0.5.1, world 31, 800 px, seed 1, profiled 2026-11-24).** 475 s
to 1 CE on one core: ~21 years a second on average, ~45 s per millennium early on and
~63 s per millennium by 1 CE. 1600 px takes ~7-9x as long (4x the cells, plus extra
rounds). Where the time goes:

| Part | Share | Why |
|---|---|---|
| Farming | ~40% | domestication draws a full-map random array per species every step; spread runs per species per round on the whole grid; per-step `np.roll` copies |
| Environment (climate) | ~22% | worldgen's climate history rebuilds full-map fields every year, though it changes slowly |
| States | ~14% | `_summary` (7%) recomputes every state's totals each step; fights; decisions |
| Population, culture | ~8% each | migration; `_strongest_neighbour` |
| Settlements, recorder | ~6% | town boxes; saving changes and keyframes |

Every array is the full 800x400 grid, but only 22% of it is land, and many layers are
float64.

**Targets.** First for the simulation (800 px, one ordinary computer, measured by the
benchmark below), then for the game:

| Measure | Now | Target |
|---|---|---|
| Speed, 10,000 BCE to 1 CE | ~21 years/s (8 min) | 100+ years/s (under 2 min) |
| Speed with later phases (tech, trade, religions, modern world) | - | never below ~20 years/s at 800 px |
| 1600 px vs 800 px | ~7-9x | ~4x (in line with the cells) |
| New world to first simulated year | seconds (cached) | under ~10 s, worldgen included |
| Memory | not measured | under ~2 GB |
| Saved run to 1 CE | 40 MB | about the same or less as phases add layers |
| Viewer while simulating (game) | pauses while a year runs | smooth panning and zooming (~60 fps) whatever the simulation is doing |

**How.**
- **Measure first:** a benchmark command (`python -m histsim bench`: fixed world, seed and
  eras, one core) and per-system timings recorded in each run's stats, so every phase
  reports its cost and graphs show where time goes. Each phase gets a time budget; going
  over it is noted and fixed in the next optimization pass.
- **Same history, faster:** a pure speed change must give exactly the same run (same
  stats and events on a fixed seed), as with the 800 px checks so far; where it can't
  (float order, float32), the change is checked to stay within seed spread and logged.
- **Land-only cells:** keep per-cell layers as 1-D arrays of land cells with precomputed
  neighbour indices (and river edges), instead of the full grid with `np.roll` shifts:
  up to ~4x less work on every cell operation.
- **Smaller types:** float32 for layers, small integers for ids and modes.
- **Do less:** climate every few years or interpolated between keyframes (fixed once the
  Holocene settles); domestication and spread only on frontier cells and species that
  can move; state summaries updated by what changed (conquests, growth) rather than
  rebuilt; systems that go quiet run less often (the loop already allows it).
- **numba kernels** for per-cell loops and fused array maths (spread, migration, fronts,
  summaries), multi-core where it helps (`prange`), and the recorder writing in a
  background thread.
- **Game side:** the simulation runs in its own process or thread, the viewer draws from
  the latest finished year (and replays saved years), so the map never waits on a year;
  redraws only what changed; the game interface (phase 9) renders on the GPU. The
  simulation can run ahead of what the player is watching, and a game can start at a
  later era from a history simulated in the background.
- **If Python hits a ceiling** (likely only with the modern world, phase 8): move the
  hottest kernels, or the core loop, to a compiled language (e.g. Rust with Python
  bindings). Decided at the pass before phase 9, from the benchmark.

**When.**
- **Before phase 4 (O0, small):** the benchmark and per-system timings, plus the two easy
  wins (climate cadence, domestication only where it can happen), so phase 4's cost is
  measured from a clean baseline. **Done 2026-11-24 (histsim 0.5.2):** `python -m histsim
  bench` (years/s by millennium, time by system, a history fingerprint), `seconds_<system>`
  in every run's stats and a run-time graph; the climate's maps are rebuilt every
  `step_years` (when the systems that read them run) with global numbers kept yearly;
  domestication works only on cells where a species could be tamed. Same history as
  0.5.1 (world 31 seed 1 to 1 CE: every stat and event identical, apart from float32
  last-bit noise that 0.5.1 shows against itself). 800 px to 1 CE: ~475 s -> 297 s
  with saving (~34 years/s; 57 years/s early, 22 by 1 CE). Time now: farming 38%
  (spread), states 21%, population 13%, cultures 12%, environment 7%, settlements 6%,
  saving 3%: the next gains are O1's.
- **Every phase:** stays within its time budget; the benchmark is in its results.
- **After phase 4 (O1, main pass):** land-only cells and neighbour indices, float32,
  numba kernels for the hot systems, incremental state summaries; aim for the 100
  years/s target.
  **Part 1 done 2026-11-24 (histsim 0.7.3, worldgen 1.5.2):** only changes that give
  exactly the same history (world 31 seed 1 to 1 CE: same fingerprint, 4c4d4b5bf7db3cd1):
  the spread of crops and animals, crop maturing, domestication and food per cell worked
  out only on the cells involved (the random draws still cover the map); neighbour counts
  and the cultures' strongest-neighbour search compiled (numba); state summaries over
  states' own cells, with capitals and kinds kept in lookups rebuilt only when they
  change; drought fields cached in worldgen's climate history. 800 px to 1 CE: 320 s ->
  221 s (31 -> 45 years/s; 49 -> ~70 years/s early, 23 -> 34 by 1 CE). Time now: farming
  27%, population 20%, states 17%, settlements 14%, cultures 10%, technology 7%,
  environment 5%. A compiled `shift` was slower than numpy's and was dropped.
  Found on the way: a world freshly generated gave a slightly different history from the
  same world loaded from the cache (the cache keeps some fields at lower precision);
  worldgen 1.5.2's `load_or_generate` now returns the saved copy, so it no longer matters.
  **Part 2, first round done 2026-11-24 (histsim 0.7.4):**
  - exact: random numbers drawn only where they are used (the stream is skipped over the
    rest with the generator's `advance`, keeping its half-used 32-bit draw; same numbers);
  - compiled (numba), the same rules cell by cell: farming capacity and the
    intensification ceiling (one shared cell function), population growth and famine,
    migration, intensification, and the settlements' neighbourhood sums and maxima.
  These change floating-point details only: on worlds 31 and 22 (800 px, seeds 1-3, to
  1 CE) every event is the same as in 0.7.2 and no statistic differs by more than 0.12%.
  800 px to 1 CE: 221 s -> 176 s (45 -> 57 years/s; from 320 s at the start of O1).
  Time now: farming 22%, states 20%, population 15%, settlements 15%, cultures 12%,
  technology 9%, environment 7%.
  **Second round done 2026-11-24 (histsim 0.7.5):** compiled the states' per-state
  totals, cell worth and local strength (one pass over the map), farming's food per cell,
  crop maturing, frontier and pioneer cells (no per-species map scans), the settlements'
  feeding limits and ladder of towns, the cultures' strength and assimilation chance, and
  the technology needs check; capitals find their state's cells with one sort; worldgen's
  climate sums its regional patterns in one weighted pass. Again every event is the same as
  in 0.7.2 on worlds 31 and 22, seeds 1-3 at 800 px. 800 px to 1 CE: 176 s -> 135 s
  (74 years/s; ~100 years/s early on, ~51 by 1 CE). Time now: farming 23%, states 18%,
  settlements 18%, population 15%, cultures 11%, technology 9%, environment 4%.
  **Third round 2026-11-24 (histsim 0.7.6):** per-cell kernels run on all cores (numba
  `prange`; OpenMP threads set to sleep when idle, or they slow the numpy work between
  kernels); migration rewritten so each cell adds up its own four flows (parallel, and
  still the same numbers); every species' frontier and pioneer cells found in one pass;
  random-number skipping draws through short gaps. All of that kept the history
  identical. One change on purpose does not: domestication now decides with two random
  numbers per species (does any cell tame it this step, from the combined chance; which,
  in proportion to each cell's chance) instead of one per candidate cell. The odds are
  the same; checked on worlds 31 and 22, seeds 1-3 (800 px, 1 CE), everything stays
  within the seed spread of 0.7.2 (population ~215M against ~204M, people in states
  ~57% against ~54%, cities of 100,000+ 26 against 25, writing and bronze within a
  few centuries). Speed: the same history as 0.7.5 ran in 124 s (81 years/s); with the
  new domestication the six-run batch takes ~5% less time than before, and single runs
  vary with their history (world 31 seed 1: 134 s with 513 states at 1 CE; world 22
  seed 1: 153 s). The states system now takes the most time where there are many states
  (python loops over states in admin, decisions, capitals): the next target.
  **Fourth round 2026-11-24 (histsim 0.7.7):** compiled the states' absorption of land,
  war fronts and neighbour counts (`_surrounded`), the settlements' step (farming age,
  hinterland shares, sites and scores; the hinterland's land area computed once), forager
  capacity (the coastline recomputed only when the sea moves), and the technology spread's
  neighbour check over inhabited cells only; random-number skipping without per-block
  searches. Every event is again the same as 0.7.6 on all six runs. 800 px to 1 CE: world
  31 seed 1 134 s -> 102 s (98 years/s), world 22 seed 1 153 s -> 112 s (90 years/s); the
  six-run batch ~28% faster. Since O1 began: 320 s -> ~100-112 s, about 3x. Time now spread
  across all systems (states ~20%, farming ~20%, settlements ~16%, cultures ~13%,
  technology ~11%, population ~7%); the 100 years/s target is reached on one world and
  close on the other. What remains is many smaller pieces; O2 (after phase 6) re-profiles.
- **After phase 6 (O2):** re-profile with religions, disease and nomads; fix what grew.
- **Before phase 9 (O3, game pass):** simulation and viewer separated, rendering, start
  time, memory and save size, the compiled-core decision; then the 20 years/s floor is
  checked across the whole run to the present day.

## 5. Starting the world (year 0)

### The peopling pre-pass (before year 0)
A quick setup, not a year-by-year simulation of the Old Stone Age:

1. **Origin:** humans start in one warm, open, well-watered region on a large landmass.
2. **Ice-age world:** sea level is lowered by about 60-120 m during the pass, opening
   land bridges across shallow seas; ice sheets block the far north and south.
3. **Spread:** a travel-time calculation from the origin across land (easy in
   grassland and open woodland, slow in jungle, desert, mountains and tundra) and
   island-hopping across open water up to 300 km (long hops much later than short ones).
4. **Ancient voyages:** a large landmass still empty but within 1,500 km of settled land
   (`ancient_voyage_km`) gets founders on its nearest coast, later the longer the
   crossing; new lands can launch further voyages. Arrival dates are a schedule: lands
   reached after year 0 get their first bands when that year comes round.
5. **Result at year 0:** hunter-gatherer bands everywhere that was reachable, at
   densities set by plant growth. Truly remote lands, small islands and ice-covered land
   start empty, waiting for later seafarers (in a survey of 40 worlds, about 7 in 10
   still have at least one empty large landmass at year 0). For different tastes,
   `--peopling sparse|some|most|all` (PEOPLING_LEVELS in config.py) gives emptier or
   fuller worlds; later this becomes a choice in the viewer's new-run screen.
6. **Depth of settlement:** how long ago each region was reached sets its starting
   cultural and language diversity (older regions are more diverse).

### Cultures and languages (phase 2)
- **At year 0:** a few dozen language families (about one per 4 million km2 of settled
  land) and several hundred cultures inside them (about 4 per million km2), smaller and
  more numerous where people have lived longest and forage most. Borders follow travel
  effort (rough, wooded or mountain country) plus noise, not straight lines.
- **Languages:** each family has a proto-language (its own sounds, syllable shapes,
  word lengths and spelling); each culture speaks a daughter language. When a culture
  splits, the new language inherits the old one's words through 2-4 regular sound
  changes (p to f, final vowels lost, k before i to ch...) and replaces about 12% of its
  words, so related peoples have recognisably related names and unrelated ones don't.
- **Change:** farming arriving among foragers usually brings the farmers' culture (75%),
  so families spread with farming; denser, farming or herding, town-dwelling neighbours
  slowly absorb weaker ones; a culture spread wider than ~400 km from its centre (800
  for herders) splits along natural barriers; a piece cut off for long becomes its own
  people; cultures with nobody left end.

### Settlements (phase 2)
Towns grow where farming is old (from ~2,000 years after it arrives, fully by ~5,000),
the hinterland within ~160 km is dense, and the site is good (rivers, great rivers,
coasts, river mouths, lakes). The best site in each neighbourhood draws a share of its
hinterland's people and grows or shrinks gradually; established towns keep an edge.
Places of 2,000+ are named towns (in the local language, from features like river, ford,
hill or sea); 10,000+ are cities. Bigger capitals and urban shares come with states
(phase 3).

### States, war and breakup (phase 3)
- **Formation:** where farming is ~4,000 years old, a town may become a chiefdom's seat;
  herders with horses may form confederations. Chiefdoms of 150,000+ people with a city
  of 10,000 (or twice the people) become kingdoms; an empire needs a share of the world's
  people that shrinks as the world fills (3.5 million when it holds 100 million, rising
  with its square root; 80% of that if mostly of other peoples); a single city with its
  fields is a city-state.
- **Grip:** a state's strength at a cell is the square root of its people (herders with
  horses count extra) times its cohesion, fading with travel distance from the capital
  (open plains count as their real distance, rough country more). Its reach grows with
  its size. Strong states take in stateless villages, farmers far more easily than
  foragers or herders, and villages they half surround far more readily than ones at the
  tip of a thin finger, so borders fill in instead of reaching out in strings.
- **Decisions:** every state decides through one step (`histsim/agents`): attack a
  neighbour when clearly stronger along the border (more readily if it is busy elsewhere;
  equals fight now and then), seek allies against a giant. A player will later use the
  same step.
- **War:** front cells change hands by local strength against defence (hills, great
  rivers, walls, the capital); conquest kills 5% of a cell's people. Wars last ~18 years,
  then peace and a truce of 15-50 years; losing the capital means annexation for a small
  state, a new capital for a large one. Allies join defensive wars.
- **Cohesion** (Turchin's asabiya) grows on frontiers with peoples of other language
  families and fades in quiet cores, faster for very big states (overextension, from ~10
  million people). Distant regions of other cultures rebel when cohesion is low; when it
  runs out (below 0.12) a big state collapses at once into 2-8 successor states around
  its great towns, so empires rise fast and fall within centuries.
- **Submission:** a neighbour six times weaker may bow to a great power without war.
- **Effects:** capitals draw a share of their state's people (fed and peopled from across
  it); states speed up farming intensification; the rulers' culture slowly assimilates
  subjects.
- **Not yet:** vassals and tribute, marriages and named rulers (phase 5), armies moving
  over the map, sea power (phase 7). Technology (phase 4) should lengthen states' reach
  (writing, roads, horses, bureaucracy) so that a few empires can hold a fifth of the world.
- **Later: how wars are fought.** Today every cell along a front may change hands each
  step, so a war pushes the whole border outward at once and conquests grow as rounded
  blobs (or diamonds along the map grid; fronts now favour cells already half surrounded,
  which helps but doesn't change the idea). Real wars aim somewhere: at a rich province,
  a river valley, a city, a pass, a holy place, the enemy capital, along roads and rivers,
  and they end in treaties that hand over named regions, not a ring of land. When
  technology (roads, armies, siege), religion (holy war, sacred cities), trade (routes and
  ports worth taking) and named rulers arrive, war should become: pick war goals (regions,
  cities), move along routes toward them, take what is reached, and settle by treaty.
  Borders would then follow rivers, mountains and old provincial lines.
- **Later: empires as regional powers** (idea, 2026-11-23). An empire is defined today
  by how many people it rules against the whole world. Historically an empire was the
  dominant power of its own region, whatever the rest of the world held: the Inca ruled
  perhaps 10 million and the Aztecs 5-6 million, far fewer than Han China, yet each
  dominated its world; Akkad or Assyria likewise relative to their neighbours. A better
  rule: a lower absolute floor (around 1 million people) plus dominance where it stands,
  measured one of these ways (to be tried against each other):
  - its share of the people within ~1,500-2,000 km of its capital (or on its landmass);
  - its power against its strongest neighbour, or against the sum of its neighbours;
  - how many peoples (cultures, language families) and former states it rules.
  Gains: empires on small or isolated continents (an Andes or Mesoamerica) get counted,
  and a crowded region full of similar states doesn't make all of them empires. Unlike
  today's label, this could also feed back: dominance could slow cohesion loss
  (prestige, tribute) or draw neighbours to submit, which bears on empire lifespans and
  on the largest state's share (2-15% of the world here, against Rome's or Han's
  20-25%). Worth doing with the phase 3 re-check or phase 4's technology of rule; it
  needs a regional measure (people within a radius, already similar to the town
  ceiling's) and the empire-count and lifespan checks redone at 400 and 800 px.

### Rivers and the spread of farming (phase 3 follow-up)
Most early states arose on rivers (Mesopotamia, Egypt, the Indus, the Yellow River), and
farming spread unevenly, not as a smooth ring. The simulation now follows that:

- **River water** (worldgen 1.4 `floodplain`, `irrigable`): on floodplains crops grow on
  the receding flood whatever the rain, so desert rivers can be farmed (the Nile), and
  the yearly silt makes harvests up to twice as large. Canals water the wider irrigable
  land once known: `canal_irrigation` (0..1) is the hook for phase 4's technology.
- **Circumscription** (Carneiro): good farmland hemmed in by desert, mountains or sea
  within ~200 km forms chiefdoms up to 5x as often and up to 40% sooner; the losers of
  wars there have nowhere to flee.
- **Uneven spread:** farming moves fastest along great rivers (up to 7x) and coasts (4x)
  and over good soil (as the first farmers followed central Europe's loess valleys);
  pioneers settle good river or coast land 50-300 km ahead of the front, then the gaps
  fill in (Linear Pottery, Bantu farmers).
- **Forager frontiers:** foragers take up a crop readily only once it would feed 2.5x as
  many people as foraging; where they live well (rich coasts, lakes, rivers) farming
  creeps at a few percent of its pace until crops improve (northern Europe's ~1,000-year
  pause).
- **Later:** stored, taxable grain (cereals) favouring states over tuber farming;
  seafaring pioneers (the Mediterranean's coast-hopping farmers) with phase 7.

### Farming intensification (built 2026-11-23, histsim 0.5.0)
**Problem.** Farmland away from rivers is too even. World 31 at 1 CE (800 px, seed 1):
0.4-2.2 people/km2 (10th-99th percentile), a 4.5x spread with a hard ceiling near
`farmer_density` (2/km2); river farmland is no denser (median 1.0 against 1.1) and the
world holds 72-76 million people (target 100-500 million, Earth 200-300 million). On
Earth around 1 CE farmland ran from 2-5/km2 on frontiers (Germania) through 10-40
(Gaul, Italy) to 50-100 (Han China's core) and 100-300 on cultivated Nile land: tens of
times more in old cores than on frontiers.

**Causes on Earth, which the simulation should follow:**
1. Intensification (Boserup): crowded farmers work land harder over centuries: shorter
   fallow, ploughs, manure, terraces, irrigation, double cropping. The main cause.
2. The crop: wet rice paddies feed several times what rain-fed grain does.
3. States: order, storage, roads, markets and organised irrigation; old core provinces
   were far denser than borderlands.
4. Setbacks: war, famine and collapse abandon fields and let terraces and canals decay.
5. Frontier lag: new farmland takes generations to fill.

**Design.**
- A new cell layer `intensity` (0..1, recorded on change): how far farming there has been
  intensified. Farming capacity is multiplied by `1 + (ceiling - 1) x intensity`.
- **Growth from crowding:** intensity rises only while farmers press on their land
  (population over ~80% of capacity), over roughly 2,000-3,000 years to near full; faster
  next to more intensive land (methods spread) and inside a state (public works), and
  faster the longer that state has held the cell.
- **Ceiling by land:** about 6x on rain-fed farmland, up to ~15x on floodplains and
  irrigable land (irrigation agriculture), and higher where the best crop is the
  rice-like wet grain (paddies, x1.5); poor soil and steep land lower it (terraces later).
- **Decay:** intensity falls when people drop well below capacity (abandoned fields,
  broken canals: about half in a few centuries), when a cell is conquered (a little), or
  stops being farmed; so collapses lower capacity and dark ages last.
- **States:** the flat `state_food_bonus` (+25%) becomes part of this: states speed
  intensification rather than add food directly. A state that has held irrigable land
  for long waters it (a local version of `canal_irrigation`, which stays phase 4's
  technology hook for the wider spread of canals).
- Settlements, states and war need no change: bigger, denser populations make bigger
  towns, richer cores and stronger states by themselves. Check that the largest city
  and largest state move toward their targets (200,000-2 million; ~20-25% of the world)
  without overshooting.
- Per-cell and per-year, so it scales with map size like the rest (the neighbour spread
  uses the same rounds as other spreading). A graph of intensive farmland share and the
  spread of farm densities; the hover shows intensity; maybe a map mode later.

**Calibration targets (world 31 and 22, 800 px, three seeds; 400 px for iterating):**

| Measure, 1 CE | Now | Aim |
|---|---|---|
| World population | 72-76M | 150-300M (accept 100-500M) |
| Farm density, 90th / 10th percentile | 4.5x | 20x or more |
| Oldest cores (top 1% of farm cells) | ~2/km2 | 30-150/km2 |
| Frontier farmland (bottom 10%) | 0.4/km2 | 1-5/km2 |
| Largest city | 52-61k | 200,000+ (accept 100,000+ for now) |
| Urban share | 2.5-2.9% | 2-15% |
| Population 5000 BCE / 3000 BCE | in range | stay in range (intensification takes time) |

Plus the existing checks: first cities and kingdoms, wars and state counts, and 400 vs
800 px. Then re-check the phase 3 tuning (empire size and lifespan), which bigger
populations will move.

**Steps:** (1) intensity layer, ceiling and capacity multiplier, with unit tests (grows
only under pressure, never above the ceiling, decays when emptied); (2) spread to
neighbours and the state speed-up, folding in `state_food_bonus`; (3) decay on conquest
and depopulation; (4) stats, graph and hover; (5) calibrate on 400 px runs, confirm on
800 px for worlds 31 and 22; (6) 400 vs 800 px check; (7) docs, decision log, version
(histsim 0.5.0, since every history changes).

**Built.** As designed, with these changes from tuning:
- **Shifting and settled farming.** Farming begins as shifting (slash-and-burn) fields
  with long fallows, at a low base density (`farmer_density` 1.2/km2, was 2.0);
  intensity is how far a place has moved to settled, intensive farming (the viewer
  calls farmers below intensity 0.15 "shifting", above it "settled").
- **Density drives it (Boserup).** Intensification runs once the land is filling up
  (60% of what it feeds), at a pace that grows with farmers per km2 (full pace at 2/km2,
  up to 3x). Driving it by crowding alone stalled: each gain lowered crowding, so cores
  never pulled away. By density, dense places run ahead and thin land stays shifting.
- **Ceilings:** 6x on rain-fed fields, 30x on floodplains and on *dry* irrigable land a
  state holds (canals; 40% of farmland counts as irrigable, mostly in wet places where
  canals add little), x1.5 for paddies; states double the pace; conquest costs 10%.
- Tried and dropped: a market rule (faster near big towns): little gain.

Results at 1 CE, 800 px, seeds 1-3 of worlds 31 and 22:

| Measure, 1 CE | Before | Now | Aim |
|---|---|---|---|
| World population | 72-76M (w31) | 111-262M (w31), 225-328M (w22) | 150-300M |
| Population, 5000 BCE | 14-17M | 11-27M | 5-20M (accept 2-40M) |
| Densest 1% of farmland | ~3/km2 | 38-63/km2 | 30-150/km2 |
| Off-river: median / densest 1% | 1.1 / 2.2 per km2 | 0.7 / 4-12 per km2 | thin, with real cores |
| Off-river: densest tenth / thinnest tenth | 4.5x | 5-12x | ~10x (Italy against Germania) |
| Largest town | 52-61k | 0.43-0.84 million | 0.2-2 million |
| Urban share | 2.5-2.9% | 7.3-9.4% | 2-15% |
| Empires ever by 1 CE | 14-21 | 24-76 | (Earth roughly 20-30) |
| People in states | 51-73% | 82-93% | |

The map now shows dense cores along rivers and across whole old regions, with thin
shifting farmland between them, instead of an even field. Still open: (1) more people in
states (82-93%; the empire count is fixed in histsim 0.5.1, see the decision log); (2) the absolute densities are lower than Earth's (most farmland here is
thin), while the spread between cores and frontiers is about right; (3) 400 px runs
come out higher where the seeds already differ by 5000 BCE (where farming starts).

### Technology, economy and trade (phase 4 plan, 2026-11-23)
**Why now.** What is still off after phase 3 and intensification mostly comes from what
states and farmers can't yet do: the largest state holds 2-15% of the world (Rome or Han
20-25% each), time spent as an empire varies from 175 to 1,225 years by seed (Earth 2-4
centuries), 82-93% of people live in states by 1 CE, canals wait on the
`canal_irrigation` hook, and nothing spreads goods or ideas between distant places.
Phase 4 adds technology, then trade, and re-checks the phase 3 tuning on top of them.

**Technology: what it is.**
- A fixed web of real technologies (`histsim/tech.py`), each with prerequisites, needs
  (resources nearby or by trade, a tamed species, a minimum population, towns or
  specialists), a difficulty, and effects. Phase 4 covers about 25, roughly to the end
  of the ancient world; later phases add more to the same web.
- **Known per cell** as a bit mask (as tamed species are), so knowledge has a geography:
  it spreads, fades and survives in places, not by state or culture. States and
  cultures use what their cells know (a state uses what its core knows).
- **Discovery:** a small chance per step where the needs are met, rising with people,
  towns and contact (more minds and more exchange: Kremer, Henrich). Hard technologies
  need big, connected populations, so they are usually found once or twice and then
  spread; easy ones (pottery, the plough) arise in several places.
- **Spread** to neighbouring cells (faster along rivers, coasts and trade routes, and
  inside a state), only where it can be used (no bronze without copper and tin coming
  in, no chariots without horses).
- **Loss:** knowledge that needs specialists fades where towns and people fall below
  what keeps it alive (Tasmania; writing lost in Greece's dark age), so collapses set
  places back.
- **Names and chronicle:** each discovery gets a name in the discoverers' language and
  an event ("the Kessani learn to smelt bronze"); first discoveries and independent
  inventions are recorded.

**Technologies in phase 4** (Earth's first dates as a guide; prerequisites in brackets):

| Group | Technology | Earth | Main effect here |
|---|---|---|---|
| Farming | pottery | 14,000-9000 BCE | storage: famine milder |
| | plough [farming, cattle] | ~4000 BCE | rain-fed intensification ceiling up |
| | terraces [settled farming] | ~3000 BCE | steep land can be intensified |
| | canals [towns, irrigable land] | ~6000-4000 BCE | `canal_irrigation` per cell: dry irrigable land watered |
| | manuring and rotation [plough] | ~1000 BCE | rain-fed ceiling up again |
| Metals | copper working [copper] | ~5000 BCE | small war and tool bonus |
| | bronze [copper working; copper and tin] | ~3300 BCE | war bonus; drives the tin trade |
| | iron working [bronze or furnaces; iron] | ~1200 BCE spread | cheap metal for all: war bonus, plough and clearing |
| Transport | wheel and cart [cattle or horse] | ~3500 BCE | travel cost on open land down |
| | riding [horse] | ~3500-2000 BCE | herder and state reach up; raids |
| | chariot [wheel, horse, bronze] | ~2000 BCE | war bonus on open land |
| | boats and sailing [rivers or coast] | ~3500 BCE | river and coastal travel and trade; sea crossings (deep sea: phase 7) |
| | roads [state, wheel] | ~2000 BCE+ | travel inside a state cheaper: reach up |
| Rule | writing [towns, a state] | ~3200 BCE, 3-4 times | records: cohesion lasts, reach up |
| | counting and calendar [writing or towns] | ~3000 BCE | needed for bureaucracy |
| | bureaucracy [writing, counting, kingdom] | ~2500 BCE | overextension from a larger size: big states hold together |
| | law codes [writing] | ~1800 BCE | rebellions rarer |
| | money (coinage) [metals, trade] | ~600 BCE | trade volume up; taxes |
| Building and war | monuments [towns, surplus] | ~3000 BCE | prestige: cohesion |
| | city walls [towns] | ~7000 BCE+ | town defence up |
| | siegecraft [walls, bureaucracy] | ~900 BCE | town and capital defence down for attackers |
| | professional armies [bureaucracy, iron] | ~700 BCE | war bonus, wars longer |

**Technology of rule: how states change.** The aim is the plan's old note: a few empires
can hold a fifth of the world, and empires last 2-4 centuries, not 1,000+ years.
- Reach grows with writing, roads, riding and bureaucracy; overextension starts at a
  larger size with bureaucracy (today a fixed 10 million).
- Cohesion: law codes and monuments slow its loss; collapses stay possible (the empire
  lifespan target), so these lengthen the life of large states without making them
  permanent.
- War: metals, chariots, iron and professional armies add strength; walls and siegecraft
  move the balance between attacker and defender.
- Then the phase 3 re-check: empire lifespans, the largest state's share, people in
  states, and the empires-as-regional-powers idea (states section, "Later").

**Step 2 also reshapes how states grow (planned 2026-11-24).** Measured at 0.6.0 (worlds 31
seed 2 and 22 seed 1, 800 px, every century 5000 BCE-1 CE):
- *No pauses:* states take in 300-800 stateless cells every century, steadily; the big
  states grow in 60-90% of centuries and hold steady in only 0-30%. Earth's grew in bursts
  (Akkad, Qin, Rome's rise, the Inca, the Mongols: most conquest within 1-3 generations of
  a take-off, often a new dynasty), then held for long plateaus (Egypt, Han, Rome after
  Augustus).
- *No limits:* thinly peopled land is the easiest to absorb (resistance grows with
  density), so at 1 CE states hold 23% of tundra, 26% of boreal forest and 16% of alpine
  land. Early states stopped where taxable grain farming ended: desert, steppe,
  mountains, deep forest and swamp were not worth taking (Scott).
- *Nothing falls back:* no land ever returns to statelessness, and ~25-30% of all state
  land changes owner each century, so big states swing wildly (263 -> 871 -> 1,558 -> 202
  -> 514 cells). On Earth collapse often left land without states for centuries (Britain
  after Rome, the Maya lowlands, the Late Bronze Age collapse), and borders between
  settled kingdoms mostly moved a little at a time.
- *Chiefdoms too big:* states under 150,000 people have a median 12-19 cells (30,000-50,000
  km2) and up to 190 cells; Earth's chiefdoms were a few thousand km2, a day or two's walk
  across, and cycled: rose and broke up within a few generations (Anderson). On the map
  chiefdoms, kingdoms and empires all look the same.

Changes, built and calibrated with the technology of rule:
- **Expansion drive:** each state has a drive to expand, high while it is young and
  cohesive (a new dynasty, a rising people: Ibn Khaldun's asabiyyah) and fading with age;
  conquest and fast growth add a pause to digest new land (overextension felt sooner).
  Absorption and decisions to attack scale with it, so growth comes in pulses with
  plateaus between. Later (phase 5) rulers and dynasties drive it.
- **Absorb for what land is worth:** the chance of taking in stateless land rises with
  its farmers and surplus (people per km2 in farming, intensity) instead of falling with
  resistance; foragers' and herders' land, desert, tundra, forest and mountain stay
  outside states unless densely farmed, so states stop at natural frontiers.
- **Collapse can free land:** when a state breaks up or shrinks, poor outlying land
  (sparse, foraging or herding, far from towns) may fall back to statelessness instead of
  passing to a successor; chiefdoms often dissolve back into villages.
- **Chiefdoms as their own tier:** reach ~100 km, little conquest (raids rather than
  annexation), cohesion that fades fast, so they rise and break up within a few
  generations; becoming a kingdom needs a city and, to hold far, the technology of rule
  (writing, bureaucracy, roads: the reach and overextension effects above).
- **Calmer borders:** wars between established kingdoms end sooner with smaller gains;
  longer truces; attacking a state of similar strength is rarer (balance of power).
- **Viewer:** chiefdoms paler and hatched with thin borders, kingdoms solid, empires with
  bold borders; the kind in the hover and the legend; the same in the `states` maps.

Targets for this part (800 px, worlds 31 and 22; averages across seeds near Earth's, single
runs may stray):

| Measure | Now (0.6.0) | Aim |
|---|---|---|
| Big states: centuries steady (+/-5%) | 0-30% | about half or more |
| State land changing owner per century | ~25-30% | ~5-15% |
| Land falling back to statelessness | never | some, after collapses |
| Chiefdom size, median | 30,000-50,000 km2 | a few thousand km2 (1-3 cells at 800 px) |
| State share of tundra, boreal forest, alpine at 1 CE | 16-26% | under ~5% |
| People in states at 1 CE | 82-93% | 50-85% |

**Step 2 as built (histsim 0.6.1, 2026-11-24).** A state uses what its capital's people know:
- *Technology of rule:* reach x (1 + 0.2 writing + 0.5 roads + 0.2 riding + 0.5 bureaucracy);
  overextension starts x5 later with bureaucracy (x1.5 with writing); written law (-25%)
  and monuments (-15%) slow cohesion loss, and law calms provinces (restlessness -0.3).
- *War:* army strength + 10% copper, 25% bronze or 40% iron (the best counts), + 20%
  chariots, + 30% professional armies; walls add half to town and capital defence,
  siegecraft takes 60% of it away.
- *Growth in bursts:* each state has a drive to expand (0.5-1 when founded or when a new
  ruling house arises, about every 150 years; fading over ~150 years); gains being
  digested (~50 years) hold it back. Stateless land is taken in only while the drive is
  above 0.3 (fully from 0.55), at twice the old rate; the drive also scales the readiness
  to go to war.
- *Land by its worth:* the chance of taking in stateless land scales with taxable people
  per km2 (farmers fully, herders 0.3, foragers 0.05; full worth at 0.25/km2).
- *Chiefdoms:* found on their seat alone (at 800 px), also among stateless farmers at 1.5+
  per km2 without a town; reach 0.4x; war readiness 0.3x; dissolve back into villages when
  cohesion falls below 0.1.
- *Land falls back:* poor land (worth < 0.3) beyond a weak state's reach goes back to
  governing itself (typical wait 300 years at no cohesion), and is let go when a state
  collapses.
- *Calmer borders:* front cells change hands at 0.08 a year (was 0.12), truces 30-100 years
  (was 15-50), rivals of equal strength rarely attack (0.03, was 0.06).
- *Viewer and maps:* chiefdoms pale and hatched with thin borders, kingdoms solid, empires
  with bold borders; each state's kinds over time are kept (`kinds`), so replays show what
  a state was in that year; the hover names the state and its kind.

Worlds 31 and 22, 800 px, seeds 1-3, to 1 CE (0.6.0 -> 0.6.1):

| Measure | 0.6.0 | 0.6.1 | Aim |
|---|---|---|---|
| Big states: centuries steady (median per run) | 10-20% | 25-48% (avg ~33%) | about half or more |
| State land changing owner per century | 25-30% | 4-14% | 5-15% |
| Land falling back to statelessness per century | 0 | 2-9% of state land | some |
| Chiefdom size, median | 39,000-47,000 km2 | 2,200-4,900 km2 | a few thousand km2 |
| Tundra, boreal forest and alpine land in states | 3-22% | 0-4% | under ~5% |
| People in states, 1 CE | 82-93% | 29-74% (avg ~57%) | 50-85% |
| Farm cells in states, 1 CE | 50-88% | 9-41% | |
| Largest state, 1 CE (peak since 5000 BCE) | 2-7% (8-15%) | 3-11% (3-24%) | 8-30% |
| Empires ever; median years as one | 15-22; 175-1,225 | 1-17; 150-725 | 1-6 centuries |
| World population, 1 CE | 111-328M | 92-303M | 100-500M |

Better on almost everything; still short: states hold steady a third of the time, not half;
the largest state at 1 CE averages ~5% (peaks reach 15-24%, then break up); states hold
the dense cores but less farmland than before, so on the map they look small beside the
town dots; world 31 seed 1 develops late (first kingdom 2325 BCE). Run time about the
same. For step 5 (with trade) and the empires-as-regional-powers idea.

**Towns and cities (planned 2026-11-24, for steps 3-5).** Measured at 0.6.1 (worlds 31 and 22,
800 px, seeds 1-6; Earth's ancient city sizes are rough estimates, after Chandler and
Modelski):

| | 3000 BCE | 2000 BCE | 1000 BCE | 1 CE | Earth, roughly |
|---|---|---|---|---|---|
| Towns of 2,000+ | 26 | 54 | 91 | 136 | thousands by 1 CE (the Roman Empire alone had hundreds) |
| Cities of 10,000+ | 14 | 32 | 59 | 94 | |
| Cities of 100,000+ | 4 | 12 | 27 | 46 (17-70) | none before ~1000 BCE; perhaps 15-25 at 1 CE |
| Largest city | 285k | 462k | 587k | 715k (374k-1.1M) | Uruk ~40-80k (3000 BCE); Rome ~0.5-1M (1 CE) |
| Share in towns | 2% | 4% | 6% | 8% | 5-10% before industry |

The urban share and the largest city at 1 CE fit; cities grow too big too early, there are
too many very big ones and far too few small towns (Earth's places follow rank-size, about
ten times as many above each tenfold-smaller size; the sim has 46 / 94 / 136 places of
100k+ / 10k+ / 2k+ where rank-size from 46 would give ~46 / ~460 / ~2,000+). Why: capitals
draw a fixed share of their whole state's people from the start (`capital_share`), however
little the state can move food; each ~160 km neighbourhood gets one central place that
takes most of its townspeople; and the map keeps one place per ~50 km cell.

Map size changes little: at 400 / 800 / 1600 px (world 31, seeds 1-3, 1 CE) there are 0.81 /
0.91 / 0.89 towns of 2,000+ and 0.28 / 0.28 / 0.29 cities of 100,000+ per million people, because
the rules space places in km, not cells. A finer map draws more named small towns but does
not fix the counts, so the fixes below are in km and people too.

Changes:
- **Capitals fed by the means to feed them:** what a capital (and any city) can draw is
  capped by how far food and tribute can be moved: rivers and coasts, then canals, roads,
  writing and bureaucracy (step 2's knowledge), later trade (step 4) and sailing. Early
  capitals stay at tens of thousands; great capitals come with empires that can feed them.
- **A ladder of places (rank-size):** each region's townspeople are split among a main
  city, a few smaller cities and many market towns (rank-size), not mostly into one
  place. Each cell keeps its largest place as a named town (as now) plus a count of smaller
  towns (a number per cell, for stats and the hover), so small towns don't need a cell each.
- **Towns where goods move (step 4):** trade routes grow towns at crossroads, river ports,
  fords and coasts, the source of many of Earth's middling cities.
- **Farming technology (step 3):** the plough, canals and manuring feed denser hinterlands,
  so towns grow where farming intensifies.

Targets (averages across worlds and seeds): largest city ~30-100k at 3000 BCE, 0.2-1M at
1 CE; cities of 100,000+ none or a few before 1000 BCE, ~10-30 at 1 CE; towns of 2,000+ at
least ~10 per million people at 1 CE; urban share 5-10% (unchanged); about the same at any
map size.

**Step 3 as built (histsim 0.6.2, 2026-11-24).**
- *Farming technology, per cell:* canals water dry irrigable land where they are known
  (crops grow there and it can be intensified up to the irrigated ceiling; replaces "dry
  irrigable land a state holds"; `canal_irrigation` stays only as a test override); the
  plough and manuring each raise the rain-fed ceiling by 30% of its gain (5.5x with the
  hoe, ~7.3x with both); terraces halve the penalty of steep land.
- *Places are fed:* no place grows beyond `feed_limit`: 20,000 people, x2 on a river, lake
  or coast, x1.2 canals, x1.25 sailing, x1.3 writing, x1.8 each for roads, officials and
  coinage (what its cell knows). Capitals draw from their state only up to it.
- *Ladder of towns:* the townspeople of each area not in its main place (half) live in
  smaller towns, rank-size (Pareto 1) from 2,000 to half the feed limit, counted per cell in
  `small_towns`; stats `towns` and `cities` include them, plus `named_towns` and
  `cities_100k`; the hover shows them; urban share counts them (`urban_share_max` 0.15 -> 0.13).

Worlds 31 and 22, 800 px, seeds 1-3, mean (range) over runs:

| | 3000 BCE | 2000 BCE | 1000 BCE | 1 CE | Aim |
|---|---|---|---|---|---|
| Largest city (0.6.1) | 285k | 462k | 587k | 715k | |
| Largest city (0.6.2) | 56k (22-78k) | 106k | 148k | 196k (60-253k) | 30-100k; 0.2-1M at 1 CE |
| Cities of 100,000+ (0.6.1) | 4 | 12 | 27 | 46 | |
| Cities of 100,000+ (0.6.2) | 0 | 2 | 6 | 12 (0-26) | none or a few; 10-30 at 1 CE |
| Towns of 2,000+ (0.6.1 -> 0.6.2) | 26 -> 237 | 54 -> 562 | 91 -> 1,055 | 136 -> 1,752 | 10+ per million people at 1 CE |
| Share in towns | 3% | 5% | 7% | 8% | 5-10% |

At 1 CE there are 8.7 towns of 2,000+ per million people (aim 10+) and the largest city is at
the low end (Rome ~0.5-1M was fed by sea trade: step 4). States: people in states 32-73%,
largest state 5-8% at 1 CE, much as in 0.6.1; world population 90-290M (mean ~200M). Run
time the same.

**Economy and trade.**
- **Surplus and wealth:** each cell produces food beyond its people's needs (from
  intensity) plus goods from local resources (copper, tin, gold, silver, salt, timber,
  fish); a `wealth` layer gathers in towns and on routes.
- **Trade routes:** every few decades, links between towns and cities by least-cost
  paths over land, navigable rivers and (with sailing) coasts, chosen by a gravity
  rule (bigger, richer, closer places trade more) and by what each lacks (tin for
  bronze-makers, salt, metals). Routes are entities (path, volume, goods) with a
  per-cell trade-volume layer.
- **What trade does:** carries technology along routes (much faster than overland
  diffusion), lets bronze be made far from tin, feeds and enriches towns on routes
  (bigger cities at crossroads and ports), and gives states income (money, roads). It
  is also the road that religions (phase 5) and epidemics (phase 6) will travel.

**Step 4 as built (histsim 0.7.0, 2026-11-24).** `systems/trade.py`, every 50 years once towns
exist:
- *Routes:* every named town (2,000+) gets the land it reaches most cheaply (one multi-source
  shortest-path search over land and coastal waters, up to 600 km x travel cost); towns
  whose lands meet are linked along the cheapest crossing. Travel cost per km: the land's
  (biome, slope) x0.8 with the wheel, x0.6 on roads inside a state; 0.3 on big rivers and
  0.2 along coasts where people sail; 3 along coasts without sailing; no open-sea crossings
  (phase 7).
- *Flows:* every pair of towns within 3,000 (km x cost) along the links trades by gravity
  (size x size / cost^2), carried along the cheapest chain of links (numba), so busy
  routes and crossroads emerge; the `trade` layer is the volume through each cell.
- *Goods:* copper, tin and gold or silver reach towns within 1,500 of a town that has them,
  and those towns' lands: the technology needs `copper_near`, `tin_near`, `precious_near`
  are met by trade as well as by ore nearby (bronze far from tin).
- *Knowledge:* each link offers what one end knows to the other; the technology system
  adds that to neighbours' knowledge, more readily the busier the link.
- *Towns:* busy places (half "busy" at a volume of 200) are better town sites (x1.5) and
  can be fed more (x1.5): crossroads, river ports and coasts grow.
- *Viewer:* a Trade map mode (routes by volume; `maps --kind trade`), trade in the hover,
  graphs of links, volume and the longest distance goods travel.

Worlds 31 and 22, 800 px, seeds 1-3, to 1 CE, before (0.6.2) and after:

| | 0.6.2 | 0.7.0 | Aim |
|---|---|---|---|
| Longest distance goods travel, 1 CE | - | 3,200-4,700 km | 2,000+ km |
| Trade links, 1 CE | - | 84-297 | |
| People knowing bronze, 1 CE | 0-25% (none on some world-22 runs) | 21-71% | widespread where tin can be had |
| Bronze first found | 4,640-1,060 BCE (0.6.0) | 4,140-2,090 BCE (mean ~3,240) | ~3,300 BCE |
| Largest city, 3000 BCE / 1 CE | 56k / 196k | 130k / 332k | 30-100k / 0.2-1M |
| Cities of 100,000+, 1 CE | 12 | 33 | 10-30 |
| Share in towns, 1 CE | 8% | 9% | 5-10% |
| People in states / largest state, 1 CE | ~54% / ~6% | ~60% / ~8% (peaks to 18%) | 50-85% / 8-30% |
| Big states holding steady | ~28% | ~32% | about half |
| World population, 1 CE | ~200M | ~210M | 100-500M |

Trade lifts what it should: bronze spreads with tin, routes reach thousands of km, the
largest state and largest city at 1 CE move up into range. Early cities are too big again
(~130k at 3000 BCE): not from trade's own feeding but because trade spreads roads and
bureaucracy to the first great states sooner, and the technology of rule still comes too
soon after writing (the discovery rule, step 5). Iron is now rarer to find (rarity 500)
and coinage much rarer (250) since trade carries gold and silver. Runs take ~15% longer
at 800 px (O1).

**Step 5 as built (histsim 0.7.1, 2026-11-24): calibration.**
- *Discovery takes time:* each technology has a `lag` (tech.py `LAG`), the least time its
  methods take to mature once a place could find it; more people find it sooner, but never
  faster than that (chance per year h / (1 + h x lag)). Writing 400 years, bureaucracy 600,
  law 800, coinage and siegecraft 1,000, iron 1,500 and so on. So the technologies of rule
  follow writing over centuries, as on Earth, instead of within decades, and capitals no
  longer get the means to feed great cities too early.
- *Iron is first smelted by smiths:* finding it needs people who already work copper (the
  `smiths` need), so iron comes after copper and mostly after bronze.
- *Big states hold longer:* overextension from 20 million people (was 10), collapse below
  cohesion 0.10 (was 0.12).
- Rarities of the later technologies brought back down now that the lag spaces them out.

Worlds 31 and 22, 800 px, seeds 1-3, to 1 CE (averages across runs; single runs stray):

| Measure | Phase 4 start (0.6.0) | 0.7.1 | Earth / aim |
|---|---|---|---|
| Writing, first (median) | ~3,600 BCE | ~3,000 BCE | ~3,200 BCE |
| Bronze, first (mean) | ~3,030 BCE | ~2,970 BCE | ~3,300 BCE |
| Iron, first (mean) | ~1,780 BCE (often before bronze) | ~1,920 BCE (after bronze on average) | ~1,200 BCE |
| Bureaucracy / law / coinage (mean) | within ~300 years of writing | ~2,000 / ~2,570 / ~1,260 BCE | ~2,500 / ~1,800 / ~600 BCE |
| People knowing bronze, 1 CE | ~14% | ~20-70% (with trade, 0.7.0) | |
| Longest distance goods travel, 1 CE | - | 3,050-4,600 km | 2,000+ km |
| Largest city, 3000 BCE / 1 CE | 285k / 715k | 63k / 263k | 30-100k / 0.2-1M |
| Cities of 100,000+, 1 CE | 46 | 25 | 10-30 |
| Towns of 2,000+, 1 CE | 136 | 1,575 (~7.7 per million) | 10+ per million |
| Share in towns, 1 CE | 8% | 8% | 5-10% |
| People in states, 1 CE | 82-93% | 25-80% (mean ~54%) | 50-85% |
| State land changing owner per century | 25-30% | 4-15% | 5-15% |
| Big states holding steady | 10-20% | 22-37% (mean ~34%) | about half |
| Largest state, 1 CE (peak since 5000 BCE) | 2-7% (8-15%) | 2-10% (2-23%), mean ~5.5% | 8-30% |
| Median years as an empire | 175-1,225 | 425-1,075 | 100-600 (Earth 2-4 centuries) |
| World population, 1 CE | 111-328M | 90-298M (mean ~204M) | 100-500M |

Phase 4 leaves short: the largest state at 1 CE (big states reach 10-23% at their peak but
break up again; trade and the technology of rule have not been enough, so phases 5-6:
rulers, dynasties and faiths binding peoples, nomad conquests), how often states hold
steady (a third, not half: phase 5's dynasties), and empires lasting a little long. Riding
and chariots almost never appear: horses are seldom tamed (a farming question for later).

**400 vs 800 px.** The 400 px runs (worlds 31 and 22, seeds 1-3) come out about half as
populous at 1 CE (~112M against ~204M) and everything happens later: writing ~1,250 BCE
(median) against ~3,000, the largest city at 1 CE ~224k against ~263k. The gap was already
there at 0.6.0 (~127M against ~205M), so it comes from phase 3-4 rules (intensification,
towns or states on coarse cells), not from this step. 400 px is the quick-iteration size,
not the size runs are judged on; finding where the gap comes from is left as a follow-up.

*Looked into (histsim 0.7.2):* most of it was world 22's peopling. Its stepping-stone
islands vanish on the 400 px map, so only ~35% of its land was reached by 10,000 BCE (94% at
800 px) and runs started with ~1.8M people instead of ~5.7M. On coarser maps the peopling
pre-pass now runs on the reference-width (800 px) map of the same world and carries the
earliest arrival in each coarse cell over (`peopling_on_reference_map`; ~1 s; 800 px runs
unchanged). World 22 at 400 px now starts as at 800 px. What remains (1 CE, seeds 1-3): ~145M
against ~204M at 800 px (was ~112M); world 22 -23% with the same share of land farmed, world
31 -39% (noisy: one seed moved from 200M to 80M from the small change of path alone). The
rest grows from farming starting and spreading more slowly on coarse cells (~40% fewer
farmers by 4000 BCE): a cell-by-cell front moves at a different speed on bigger cells.
Left for later; judge runs at 800 px.

**Viewer and graphs.** A Technology map mode (the age each cell is in: stone, copper,
bronze, iron, with writing and other marks), a trade overlay (routes by volume), hover
lines for known technologies and trade; graphs of each technology's share of people,
first discoveries, trade volume and wealth. Chronicle entries for discoveries.

**Calibration targets (worlds 31 and 22, 800 px; 400 px to iterate):**

| Measure | Earth | Accept about |
|---|---|---|
| First writing | ~3200 BCE (3-4 independent inventions) | 5000-1000 BCE, 1-4 inventions |
| Bronze age begins | ~3300 BCE | 4500-1500 BCE |
| Iron working spreads | ~1200 BCE | 2500 BCE-500 CE |
| Wheel, riding, chariot | 3500 / 3000 / 2000 BCE | in that order, within ±1500 years |
| Coinage | ~600 BCE | 1500 BCE-500 CE |
| Largest state's share, 1 CE | ~20-25% | 8-30% |
| Time spent as an empire | 2-4 centuries | 1-6 centuries, and steadier across seeds |
| People in states, 1 CE | perhaps 60-75% | 50-85% |
| Longest trade routes, 1 CE | ~6,000 km (Silk Road) | 2,000+ km on big landmasses |
| Places without metals or writing by 1 CE | the Americas, Australia, much of Africa | some lands left behind |

Plus the existing checks (population, cities, farming, wars) and 400 vs 800 px.

**Steps** (each ends with something to run and watch):
1. Technology core: the web, per-cell knowledge, discovery, spread and loss, names and
   chronicle, stats and a Technology map mode; no effects yet. Check discovery dates.
   **Built 2026-11-24 (histsim 0.6.0)**, see "Step 1 as built" below.
2. Technology of rule and war: effects on reach, overextension, cohesion and war; how
   states grow (expansion drive, absorbing land by its worth, land falling back after
   collapse, chiefdoms as their own tier, calmer borders) and the viewer showing state
   kinds; the phase 3 re-check (lifespans, largest state, people in states, regional
   empires).
   **Built 2026-11-24 (histsim 0.6.1)**, see "Step 2 as built" above.
3. Farming technology: plough, terraces, canals (per cell, replacing the global
   `canal_irrigation`), manuring; re-check population and densities. Towns and cities:
   capitals fed only by the means to feed them, and a rank-size ladder of places with small
   towns counted per cell (see "Towns and cities").
   **Built 2026-11-24 (histsim 0.6.2)**, see "Step 3 as built" above.
4. Economy and trade: surplus, wealth, trade routes and goods, trade spreading
   technology, towns on routes (crossroads, river ports, fords, coasts) and trade feeding
   cities; trade overlay.
   **Built 2026-11-24 (histsim 0.7.0)**, see "Step 4 as built" above.
5. Calibration on worlds 31 and 22 at 800 px, 400 vs 800 px, docs and decision log.
   Versions: histsim 0.6 (technology), 0.7 (trade).
   **Built 2026-11-24 (histsim 0.7.1)**, see "Step 5 as built" above. Phase 4 is done.

**Step 1 as built (histsim 0.6.0).** `histsim/tech.py` holds 22 technologies, each with
prerequisites, `needs` (to use it, so to take it up from neighbours), `finds` (needed as
well to find it first: iron is found near towns, then smelted by villagers wherever there
is ore), `keeps` (what keeps it alive), a rarity and a `crowd` exponent. The `tech` layer
(64-bit mask per cell) is updated every 10 years by `systems/technology.py`:
- needs per cell: farming, settled farming, cattle or horses kept, a town within 150 km,
  a state or kingdom, coast or big river, hills, and ore within reach (copper 500 km,
  tin 2,000 km and gold or silver 1,000 km until trade carries them; iron 300 km);
- discovery: per technology, a hazard of (people / 1M) x (1 + 3 near towns) x (people
  within 300 km / 1M) ^ crowd / (rarity x 1,000 years), summed over the cells that could
  find it; a find within 1,500 km of anyone who knows it doesn't count (it would spread);
- spread ~2 km/year to neighbours that can use it (x3 along coasts and big rivers, x2
  inside states); loss with a 150-year half-life where its keepers are gone, at once
  where nobody is left;
- each invention named in the inventors' language, in the chronicle and the `inventions`
  table; stats `know_<tech>` (share of people), graphs by group, a Technology map mode
  (stone, copper, bronze, iron; hatched = writing) and the hover lists what is known.
No effects: every other system's history is exactly as in 0.5.2 (checked on world 31
seed 1 to 1 CE: all stats and events the same). Cost ~5% of run time.

Worlds 31 and 22 at 800 px, seeds 1-3, first finds (and how many independent inventions):

| Technology | Earth | World 31 | World 22 | Accept |
|---|---|---|---|---|
| Writing | 3200 BCE, 3-4 | 4040-1620 BCE, 4-5 | 4260-3590 BCE, 3-7 | 5000-1000 BCE, 1-4 |
| Bronze | 3300 BCE | 4640-1060 BCE | 4600-2850 BCE | 4500-1500 BCE |
| Iron | 1200 BCE (spreads) | 2080-660 BCE | 2970-1270 BCE | 2500 BCE-500 CE |
| Coinage | 600 BCE | 2930 BCE or never | 3090-420 BCE | 1500 BCE-500 CE |
| Plough, wheel | 4000, 3500 BCE | never (seeds 2-3) | 4300-2360 BCE | |
| Riding, chariot | 3500, 2000 BCE | 1260 BCE once, never | never | in that order |

By 1 CE 18-61% of people live where writing is known, 12-31% where iron is, and lands
without metals or writing remain on every run (target met). Still off, for step 5:
- later technologies of rule (bureaucracy, law, siegecraft, coinage) follow writing within
  a few centuries where Earth took 1-2 millennia: once a region qualifies its hazard is
  large, so rarity has to change tenfold to matter; a slower, steadier discovery rule may
  be better than more tuning;
- writing is invented 5-7 times on world 22 (Earth 3-4);
- horses are tamed so rarely that riding and chariots almost never appear, and world 31
  has few cattle (no plough or wheel on two seeds, like the Americas): a farming question.

**Risks.** Run time (per-cell knowledge and route-finding; budget about 25% more);
everything interacts (technology raises populations and state sizes, which feed
discovery), so each step is calibrated before the next; a fixed web can make every
world's history look alike, so needs, chance and loss should leave room for different
orders and for some lands to go without.

### Wild plants and animals (worldgen 1.1)
Each world gets domesticable species with climate ranges and native homelands, blocked
by oceans and mountains, so not every continent gets every species:

- **Crops:** big-seeded wild grasses (wheat/barley-like), wetland grains (rice-like),
  a hard-to-domesticate grain (maize-like, slow to improve), tubers and roots, legumes,
  fibre and oil crops, tree crops.
- **Animals:** herd animals for meat, milk and traction (cattle-like), riding and war
  animals (horse-like), pack animals for deserts and mountains (camel- and llama-like),
  small livestock, dogs.

Farming then *emerges* where good wild species, climate and population pressure meet,
and spreads along similar latitudes more easily than across them.

## 6. What it simulates

Each system lists its main mechanisms and what it produces. Phases (section 10) say
when each arrives.

1. **Population.** Births and deaths by subsistence, food, disease, war and wealth;
   growth limited by how much food the land can provide (a Malthusian ceiling) until
   industrial farming and medicine lift it. Migration toward opportunity and away from
   famine, war and persecution.
2. **Subsistence and farming.** Foraging, herding, shifting and settled farming,
   irrigation along rivers, later improved agriculture. Domestication emerges from wild
   species; crops and animals spread along trade and migration routes. Soil fertility,
   growing season, fresh water and floodplains (from worldgen) set yields.
3. **Settlements and cities.** Villages, towns and cities from food surplus, trade,
   rivers, harbours and political centres; urbanisation and city sizes; capitals.
4. **Cultures and languages.** Cultures carry customs, values and a language. Languages
   drift with time and distance and split when communities separate; cultures merge,
   assimilate and split. Generated names for everything.
5. **Technology.** Real technologies on a flexible web of prerequisites (metallurgy,
   writing, sailing, mathematics, gunpowder, printing, steam, electricity...), discovered
   with probabilities driven by population, cities, trade contact, resources and
   institutions; spread by contact and lost after collapses. Culture-specific forms and
   names. Speculative technologies beyond the present day.
6. **Economy and trade.** Production by resource and land use; trade along rivers,
   coasts, caravan and sea routes; luxury and bulk goods; money, markets, merchants.
   Trade carries goods, ideas, technology, religion and disease.
7. **States and government.** Chiefdoms, city-states, kingdoms, empires, confederations,
   republics, nation states; generated forms of government and law. Territory on cells,
   borders following rivers and ridges; administration costs that grow with distance and
   diversity; tax, armies, legitimacy.
8. **Diplomacy and war.** Alliances, vassals, marriages, treaties, tribute; wars over
   land, resources, succession, religion and trade. Armies move over the cell grid with
   terrain, supply and river-crossing costs; battles, sieges, conquest, occupation.
   Military technology (chariots, cavalry, iron, gunpowder, ships) changes the balance.
9. **Rise and fall.** Cycles of growth, elite overcrowding, unrest and collapse (as in
   historical "secular cycles"); civil wars, secession, dynastic crises, dark ages.
10. **Religion and ideology.** Generated faiths built from deities, values, rituals and
    ideas of authority; founders and prophets; conversion along trade and conquest;
    schisms, reform movements, state religions, persecution, tolerance; later secular
    ideologies (nationalism, liberalism, socialism-like movements) also generated.
11. **Key figures.** Rulers and dynasties with traits and succession, founders of
    religions, generals, explorers, inventors; they nudge events where single people
    mattered historically.
12. **Disease.** Endemic disease by climate (malaria zones), crowd diseases that arise
    with dense farming and livestock, epidemics along trade routes, devastating "first
    contact" epidemics between long-separated populations; later medicine.
13. **Climate and disasters.** Droughts, cold spells and good centuries varying over
    time; floods, famines, volcanic winters, earthquakes.
14. **Nomads and migration.** Herders of the steppe and desert; raids, conquests and
    great migrations; mass movements from famine, war or opportunity.
15. **Exploration, navigation and colonisation.** Ships and navigation set how far
    people can sail (winds and currents from worldgen); settlement of empty islands;
    long-distance trade; colonial empires shaped by distance, disease and technology
    gaps; independence movements.
16. **Industrialisation and the modern era.** Coal, iron and institutions enable
    industry; railways, steamships, telegraph; population boom and urbanisation;
    world-spanning wars and alliances; decolonisation; globalisation.
17. **Into the future.** No fixed end: speculative technologies, resource limits,
    climate pressure, new ideologies.

## 7. Generator additions (worldgen 1.1)

Physical facts belong in worldgen (see the two-project rule), so before or during the
early phases worldgen gains:

- **Wild crops and animals:** species with climate ranges and native homelands (section 5).
- **Minerals:** separate deposits of copper, tin, iron, gold, silver, coal, oil,
  saltpeter and stone, replacing the single "ore" score.
- **Disease environment:** malaria-type zones from warmth, wetness and wetlands.
- **Winds and currents for sailing:** usable sailing directions per sea cell.
- **Ice-age sea level:** land that was exposed at lower sea levels (for the peopling pass).
- **Changing climate:** slow and sudden climate variation over the centuries.

These raise worldgen's data version, and the simulator's check is updated with them.

## 8. Saving, replay and graphs

- **Saved run:** keyframes (the full state every ~25 years), yearly changes in between,
  the event log, and yearly statistics. Expected size: around 100s of MB per full run.
- **Replay:** scrub to any year; every map mode works as in the live view.
- **Dark mode:** the live viewer and replay share the worldgen viewer's dark/light switch
  (same colours, same remembered choice); graphs and saved maps follow it too (or take
  `--dark` / `--light`).
- **Chronicle:** a readable history from the event log: foundings, wars, conquests,
  discoveries, schisms, plagues, explorations, collapses, with generated names.
- **Graphs after a run** (also used for testing):
  - world population, and by continent and largest states
  - number of states, cultures, languages and religions over time
  - followers of each religion (stacked), share of the largest religions
  - urban share, number of cities, largest city
  - technology frontier and spread of key technologies
  - wars per century, battle deaths, largest empire size, empire lifespans
  - trade volume, famine and epidemic deaths
- **Comparison view:** several runs or seeds side by side.
- **Map seam:** the viewer opens with the map cut where it crosses the least land (like
  world maps cut between Alaska and Siberia); drag sideways to turn it, `C` to re-cut.

### Later: a new-run screen (noted 2026-11-22)
Open the simulator and set up a history before starting, instead of command-line flags:
- **Choose a world:** browse world seeds with a very low-resolution preview of each
  (fast, so many can be looked through), type a seed, or pick at random; only the chosen
  world is generated at full detail.
- **History seed:** type one or randomise it (same world + same history seed = same
  history; a new history seed plays out a different history on the same map).
- **Other settings:** how much of the world is peopled at the start (`--peopling`),
  start and end years, whether to save the run, and later anything else worth exposing
  from config.py.

### Later: a more game-like interface (noted 2026-11-22)
The current viewer is a working tool built from plain matplotlib widgets. When the
player phase (phase 9) comes, the whole interface should be redesigned to feel like a
game: proper menus and screens, styled panels, smoother map interaction, tooltips and
event pop-ups. Until then, function over looks.

## 9. Checking it against real history

Each phase adds targets; results outside the ranges mean something needs tuning.
Ranges are wide on purpose: a different world should not hit Earth's exact numbers.

| Measure (real year) | Earth, roughly | Accept about |
|---|---|---|
| World population, 10,000 BCE (year 0) | 4 million | 1-10 million |
| World population, 5000 BCE | 5-20 million | 2-40 million |
| World population, 1 CE | 200-300 million | 100-500 million |
| World population, 1500 CE | 450 million | 200-800 million |
| World population, 1800 CE | 1 billion | 0.4-2 billion |
| World population, 2000 CE | 6 billion | 2-12 billion |
| First farming | ~9500 BCE | 9000 BCE ± 3000 years |
| Independent farming origins | 7-11 regions | 3-15 |
| First cities (10,000+ people) | ~3500 BCE | 5000-2000 BCE |
| First writing | ~3200 BCE | 5000-1000 BCE |
| Iron working spreads | ~1200 BCE | 2500 BCE-500 CE |
| Largest city, 1 CE | ~1 million | 200,000-2 million |
| Urban share before industry | 5-10% | 2-15% |
| Typical empire lifespan | 2-4 centuries | 1-6 centuries |
| Industrialisation begins | ~1760 CE | allow wide variation (or not at all) |
| Urban share, 2000 CE | ~50% | 25-75% |

Plus sanity checks: no growth faster than possible, conservation of people (births,
deaths and migration add up), no empire controlling everything forever, history
differs across seeds.

## 10. Phases

Each phase ends with something to run, watch and graph.

| Phase | Builds | You can watch |
|---|---|---|
| 0 | worldgen 1.1 (species, minerals, disease zones, winds, ice-age coasts); simulator skeleton: state, yearly loop, RNG, save/replay format, stats recorder, basic graphs | An empty world ticking, with graphs |
| 1 | Peopling pre-pass; foraging populations; emergence and spread of farming and herding; villages | Humans spreading out, farming appearing in a few hearths and spreading |
| 2 | Cultures and generated languages; towns and cities; the live viewer with map modes | Cultures splitting, named places, the first cities |
| 3 | States: chiefdoms to empires, territory, borders, administration; diplomacy and war | Borders forming, wars and conquests, empires rising and falling |
| O0 (built) | Benchmark and per-system timings; easy speed wins (climate cadence, domestication) | Run time and its breakdown in each run's graphs |
| 4 (built) | Technology web and its spread (including canals); technology of rule and war; states growing in bursts; farming technology; fed cities and a ladder of towns; economy, resources and trade routes | Bronze and iron ages emerging, trade networks |
| O1 | Optimization pass: land-only cells, float32, numba kernels, incremental summaries (section 4) | The same histories, ~5x faster |
| 5 | Religions and ideologies; key figures and dynasties; the chronicle | Faiths founded and spreading, named rulers, a written history |
| 6 | Disease, epidemics, climate shocks, disasters; nomads and migrations; collapse cycles | Plagues, droughts, steppe invasions, dark ages |
| O2 | Optimization pass: re-profile and fix what phases 5-6 added | |
| 7 | Navigation, exploration, island settlement, colonisation | Explorers crossing oceans, colonial empires |
| 8 | Industrialisation, modern states and ideologies, world wars, globalisation, the near future | The modern world taking shape |
| O3 | Game pass: simulation and viewer separated, rendering, start time, memory, compiled-core decision | A smooth map while history runs |
| 9 | Player control of one society | Playing a society through history |

## 11. Risks and open questions

- **Scope:** "everything that affected history" can't all be modelled; each system
  starts simple and grows. The calibration table decides what needs more detail.
- **Emergent weirdness:** interacting systems will produce odd histories at first;
  expect tuning rounds after each phase.
- **Speed:** whole-array code and numba from the start; profile each phase.
- **Run size on disk:** keyframe spacing and compression are tunable.
- **Readability:** generated names, the chronicle and good map modes matter as much as
  the models; they are built in from phase 2.
- **Open:** how future eras (past ~2100 CE) behave; how much player control changes the
  AI; whether to show the globe view in replays.
- **Map size (mostly done):** the rules are scaled to cell size (`geo.cell_scale`): extra
  rounds for fronts; river and coast bands the same width in km (widened on finer maps,
  part of a cell on coarser ones); frontier bands for cohesion likewise; counts in cells
  scaled; new states founded on ~150 km of land; breakaways bigger than a one-cell
  statelet; towns capped by the people within ~2,500 km2. Worldgen 1.5 picks wild
  species' homelands on a fixed grid, so ranges match at any width. Border strength is
  measured as on the 800-px map (cell centres lie half a cell inside a border, so on
  other maps distances to the capital are shifted by the difference), and a coarse war
  front crosses open country and then the defended core of a cell, with a capital falling
  when the front reaches its town. 800 px gives exactly the same runs as before.
  World 31 at 1 CE, all with the 800 px wildlife (worldgen 1.5), before intensification (histsim 0.4.2):

  | Width | Seeds | People | Farmed | States | In states | Biggest town | Empires ever | Wars/century |
  |---|---|---|---|---|---|---|---|---|
  | 400 px | 1-3, 6 | 66-91M | 34-40% | 250-390 | 67-85% | 74-96k | 12-37 | 31-83 |
  | 800 px | 1-3 | 72-76M | 37-39% | 250-285 | 51-73% | 52-61k | 14-21 | 33-47 |
  | 1600 px (before the last two fixes; worldgen 1.4 wildlife) | 1-3 | 59-85M | 35-45% | 175-273 | 52-66% | 43-66k | 13-19 | 31-51 |

  The 400 px state count was ~70% above 800 px before these fixes, and is now ~20%
  above it on seeds 1-3 (seed 6 has states unusually early), about the spread between
  seeds. States per million people in them are the same at both widths (5-7); what is
  left is that 400 px puts a little more of the world in states after ~1000 BCE, and its
  biggest towns are ~30% larger. 1600 px has not been rerun since (slow); expect a
  little more state turnover there now. The peopling homeland is drawn on a fixed grid of
  ~400 km blocks and sea crossings are measured to the shore's edge, so on world 31 all
  eight seeds tried start the same way at 400 and 800 px (seeds 4 and 5 used to be
  peopled late at 400 px). On world 22 a Pacific island chain is still crossable at
  800 px but not at 400 px (its islands are smaller on the coarser map), so people spread
  more slowly there at 400 px.

  **1600 px check with intensification (histsim 0.5.1, 2026-11-23).** World 31, seeds 1-3, to 1 CE:

  | Width | People | Farmed | States | In states | Largest state | Biggest town | Urban | Empires ever | As empire (median) | Wars/century | First kingdom |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | 800 px | 111-262M | 29-38% | 239-387 | 89-93% | 2.7-6.5% | 431-840k | 7.3-9.4% | 15-19 | 175-1,225 y | 48-88 | 5000-3075 BCE |
  | 1600 px | 177-287M | 41-46% | 362-442 | 89-93% | 1.8-4.7% | 481-1,032k | 7.9-9.9% | 15-24 | 275-425 y | 64-106 | 4075-3550 BCE |

  1600 px sits at the top of the 800 px range or a little above it: about a quarter more
  people, more land farmed (+7 points), ~20% more states and wars; urban share, people in
  states, empire counts and the densest farmland (43-53 vs 38-63 per km2) match. Farming
  reaching more land is the main difference (and the rest follows from it); worth a look
  if 1600 px becomes a common width, not needed for phase 4.

## 12. Decision log

| Date | Decision |
|---|---|
| 2026-11-24 | World 0 is Earth (worldgen 1.6): ETOPO 2022 elevation, wild species from real homelands, ores in real ore regions, the generator's modelled climate. For playing on Earth and calibrating against Earth's geography; the simulator models forces, it does not replay Earth's events. First run (histsim 0.7.7, seed 1, 800 px, to 1 CE, 136 s): farming begins in many places 9400-8500 BCE (the Fertile Crescent 9330 BCE, Earth ~9500), but the lead civilisation is on the Niger and Benue (first town 5475, first kingdom 3950, bronze 3680 next to the Jos tin, writing 3100 BCE), with others in the Sahel and Nubia and in Manchuria. The Nile gets a town 4025 and a kingdom 2050 BCE (Earth ~3100); the Fertile Crescent a town 4480, bronze 3100 (Earth ~3300) and a kingdom only 800 BCE; the Yellow River a town 2820 and kingdom 650 BCE; no Indus towns, no Andean domestication, Mesoamerican towns only 575 BCE. Main causes are worldgen's modelled climate on real ground (Sahara and Sahel too wet, Ganges and Yellow River too dry, Andes too cold), not the simulator: real climate normals are the next step before calibrating on Earth |
| 2026-11-23 | Empires relative to their time (histsim 0.5.1): the bar rises with the square root of world population (3.5 million people when the world holds 100 million; ~1.9 million at 30 million, ~6 million at 300 million), mostly-foreign states count from 80% of it (was 50%), and an empire stays one down to half the bar (was a quarter); only titles and counts change, not the history. 800 px, worlds 31 and 22 to 1 CE: 24-76 -> 15-22 empires (before intensification 14-21; Earth roughly 20-30), first empire 4450-2625 BCE. Time spent as an empire varies widely by seed (median 175-1,225 years; Earth 2-4 centuries): a collapse question, left for the phase 3 re-check |
| 2026-11-23 | Farming intensification (histsim 0.5.0): farming begins as thin shifting fields and becomes settled, intensive farming where farmers grow dense (Boserup: pace grows with people per km2 once the land fills up), up to 6x on rain-fed fields and 30x on floodplains and dry irrigated land, x1.5 for paddies; faster in states, lost when land empties or is fought over; replaces the flat state food bonus. 800 px, 1 CE: world population 72-76M -> 111-328M, densest farmland ~3 -> 38-63/km2, off-river farmland thin (median 0.7/km2) with cores of 4-12/km2, largest town ~60k -> 0.43-0.84M, urban share ~2.7% -> 7-9%. Empires doubled (phase 3 re-check) |
| 2026-11-23 | Peopling the same at any map width (histsim 0.4.2): the homeland is drawn on a fixed grid of ~400 km blocks, sea crossings are measured to the shore's edge and coarse travel blocks are land only when mostly land. This moves the homeland on most seeds, so every history differs from 0.4.1, at 800 px too. Worldgen 1.5.1: Enhance keeps the world's river courses (big rivers used to vanish when the view's edge fell near them) |
| 2026-11-25 | Earth is world 0 (worldgen 1.6.0): real ETOPO terrain, real 1981-2010 climate on land, real wild species homelands and ore regions. First runs (800 px, seeds 1-3, 1 CE): 267-296M people, writing 3740-1460 BCE; but the first towns and kingdoms rise in the Sahel and West Africa, around the Aral Sea and Oxus, in the Amur valley and Amazonia, while Mesopotamia, the Nile, the Indus and the Yellow River mostly come later, and many species are tamed far from their real centres (cattle in Siberia, pigs in Russia). A calibration lead for the farming and town rules (crowded rich foragers and river sites count for more than early farming's real homelands) |
| 2026-11-24 | O1 part 2, fourth round (histsim 0.7.7): states' absorption and war fronts, settlements' step, forager capacity compiled; events identical to 0.7.6 on six runs. 800 px to 1 CE ~100-112 s (90-98 years/s), about 3x faster than before O1 |
| 2026-11-24 | O1 part 2, third round (histsim 0.7.6): kernels on all cores, parallel migration, all species' frontiers in one pass (all exact); domestication drawn with two random numbers per species instead of one per cell (same odds; six runs within the seed spread of 0.7.2). ~75-81 years/s at 800 px depending on the history; the states system is next |
| 2026-11-24 | O1 part 2, second round (histsim 0.7.5): states' totals, farming per-species work, settlements, cultures and technology needs compiled; climate patterns summed in one pass (worldgen 1.5.2); events identical to 0.7.2 on six runs. 800 px to 1 CE 176 s -> 135 s (74 years/s; 320 s before O1) |
| 2026-11-24 | O1 part 2, first round (histsim 0.7.4): random numbers drawn only where used (exact), and farming capacity, growth, migration, intensification and box sums compiled; events identical to 0.7.2 on six runs, stats within 0.12%. 800 px to 1 CE 221 s -> 176 s (57 years/s) |
| 2026-11-24 | O1 part 1 (histsim 0.7.3, worldgen 1.5.2): speed changes that keep the history exactly the same; 800 px to 1 CE 320 s -> 221 s (45 years/s). Part 2 (land-only arrays, compiled growth and migration) will change float details, so it is checked against seed spread |
| 2026-11-24 | histsim 0.7.2: on coarser maps the peopling pre-pass runs on the 800 px map of the same world (small stepping-stone islands survive), fixing world 22's slow peopling at 400 px; 400 px runs now ~145M at 1 CE against ~204M at 800 px (was ~112M); the rest is slower farming spread on coarse cells, left for later |
| 2026-11-24 | Phase 4 done (histsim 0.7.1, step 5 calibration): technologies take a least time to mature once a place could find them (later technologies of rule now follow writing over centuries), iron needs copper smiths, big states hold longer (overextension from 20M, collapse below 0.10). 800 px averages near Earth for first dates, cities and trade; largest state at 1 CE still ~5.5% and states hold steady a third of the time (phases 5-6). 400 px runs come out about half as populous and later (a gap since phase 3-4, to investigate) |
| 2026-11-24 | Phase 4 step 4 built (histsim 0.7.0): trade routes between towns over land, rivers and coasts (gravity flows along the cheapest chains of links), copper, tin and gold or silver carried up to 1,500 (km x cost), knowledge carried along routes, busy places grow and are fed more; Trade map mode. 800 px: goods travel 3,200-4,700 km by 1 CE, bronze known by 21-71% of people, largest state ~8% and largest city ~330k at 1 CE; early cities too big again (~130k at 3000 BCE) through early technology of rule (step 5); runs ~15% longer. Added REFERENCES.md |
| 2026-11-24 | Phase 4 step 3 built (histsim 0.6.2): canals, plough, manuring and terraces per cell; places capped by what can be brought to them to eat (transport and the technology of rule); a rank-size ladder of smaller towns counted per cell. 800 px: largest city 285k -> 56k at 3000 BCE, 715k -> 196k at 1 CE; cities of 100,000+ at 1 CE 46 -> 12; towns of 2,000+ 136 -> 1,752 |
| 2026-11-24 | Towns and cities to be reworked in phase 4 steps 3-4: early cities far too big (largest ~285k at 3000 BCE vs Uruk ~40-80k), too many of 100,000+, far too few small towns (top-heavy, not rank-size); urban share and the largest city at 1 CE fit. Plan: capitals capped by the means to feed them (transport and the technology of rule), a rank-size ladder of places with small towns counted per cell, towns on trade routes. Counts per million people barely change with map size |
| 2026-11-24 | Phase 4 step 2 built (histsim 0.6.1): technology of rule and war (reach, holding together, law, arms, walls, siegecraft), growth in bursts (an expansion drive renewed by new dynasties), land taken in for its worth, chiefdoms that dissolve, poor land falling back, calmer borders; viewer shows chiefdoms, kingdoms and empires. 800 px: border churn 25-30% -> 4-14% a century, chiefdoms ~3,000 km2, cold lands in states 0-4%, people in states 29-74%; largest state still ~5% at 1 CE |
| 2026-11-24 | State growth to be reshaped in phase 4 step 2: states grew steadily with no pauses or limits, never gave land back, churned ~25-30% of their land a century, and chiefdoms were kingdom-sized. Plan: an expansion drive (pulses and plateaus), absorbing land by its worth, land falling back after collapse, chiefdoms as their own tier, calmer borders, state kinds shown in the viewer. Calibration judged on averages across worlds and seeds, not single runs |
| 2026-11-24 | Phase 4 step 1 built (histsim 0.6.0): 22 technologies known per cell, found where needs are met and many connected people live, spread to neighbours that can use them, lost where their keepers are gone, named in the inventors' languages; Technology map mode and graphs; no effects yet (rest of history unchanged). 800 px: writing first 4260-1620 BCE with 3-7 inventions, bronze 4640-1060 BCE, iron 2970-660 BCE; later technologies of rule come too soon after writing (step 5) |
| 2026-11-24 | O0 built (histsim 0.5.2): `bench` command with a history fingerprint, per-system run time in stats and graphs, climate maps every `step_years`, domestication only where possible. Same history as 0.5.1; 800 px to 1 CE ~475 s -> ~297 s (with saving) |
| 2026-11-24 | Optimization planned (it is to become a game): targets (800 px: 100+ years/s to 1 CE, never below ~20 years/s later; a smooth viewer while simulating), a benchmark and per-system timings, a time budget per phase, and passes O0 (before phase 4), O1 (after 4), O2 (after 6) and O3 (before 9). Profile at 0.5.1: 475 s to 1 CE, farming ~40%, climate ~22%, states ~14% |
| 2026-11-23 | Map size, states (histsim 0.4.1): border strength measured as on the 800-px map, coarse fronts cross open land then the defended core, coarse capitals fall when the front reaches the town. 400 px state count ~70% -> ~20% above 800 px (within seed spread); 800 px runs unchanged |
| 2026-11-23 | Rules scaled to map size so a different `--width` gives a comparable history: most 1 CE measures at 400 and 1600 px now fall near the 800 px range; coarser maps still have more states and wars (see Risks, "Map size"). Worldgen 1.5: wild species homelands picked on a fixed grid, so ranges are the same at any width (this changes every world's wildlife, so histories differ from 1.4) |
| 2026-11-21 | Simulate on worldgen map cells (800 px), 4-direction moves, rivers on edges |
| 2026-11-21 | Watch live, save and replay, graphs after a run; player control in a later phase |
| 2026-11-21 | Societies plus key figures |
| 2026-11-21 | Technology blend: real physics, flexible paths; invented culture, religion, ideology; generated names |
| 2026-11-21 | Year 0 ~ 10,000 BCE; peopling pre-pass places humans, wild species set by worldgen 1.1 |
| 2026-11-21 | Generated languages per culture, drifting and splitting |
| 2026-11-21 | Full runs may take hours |
| 2026-11-21 | Calendar: year 0 is exactly 10,000 BCE; year 10,000 = 1 CE; 2025 CE = year 12,024 |
| 2026-11-21 | Peopling: island-hopping up to 300 km, plus rare ancient voyages to large empty lands within 1,500 km; remote lands stay empty for later seafarers |
| 2026-11-22 | Phase 3 empire tuning: sudden collapse into successor states, submission of small neighbours, size counts more in war (x1.5), overextension from 10 million, reach grows faster with size, rulers assimilate subjects faster. Worlds 31 and 22 to 1 CE: empires last a median 325-375 years as empires (Earth 2-4 centuries), first empire ~2850-4100 BCE, the largest state holds 4-12% of the world at its peak and ~4% at 1 CE (Earth ~20-25%: still short; for phase 4's technology of rule) |
| 2026-11-22 | Rivers and farming spread (worldgen 1.4: `floodplain`, `irrigable`): crops grow on floodplains without rain and yield up to 2x; canals left to phase 4 (`canal_irrigation`); hemmed-in farmland forms states up to 5x as often and 40% sooner (Carneiro); farming spreads up to 7x faster along great rivers, 4x along coasts, toward good soil, by pioneers 50-300 km ahead, and stalls where foragers live well. Worlds 31 and 22: farmers on rivers 21-23% -> ~30%, on floodplains (5-6% of land) 8-10% -> 14-17%; newly formed states' capitals on rivers 70-86% -> 90-100%, on floodplains 30-40% -> 50-70%; first kingdom 4400 / 3450 BCE; world 22 farms more slowly (35% of land by 1 CE, was 42%); runs ~8% slower |
| 2026-11-22 | Phase 3 shapes and detail: stateless villages join a state more readily the more of their neighbours it holds, and war fronts advance where the attacker half surrounds a cell (fewer stringy borders and grid diamonds); goal-driven war noted for later. The viewer gets Enhance: the world generator's finer map of the view, with history's cells drawn on it with wandering (noise-warped) borders |
| 2026-11-22 | Phase 3 built: states from chiefdoms to empires, territory by strength and travel distance, a shared decision step (agents/) with an AI for war and alliances, fronts, peace and truces, conquest, capitals falling, cohesion and overextension, rebellions; capitals draw people, states feed more, rulers' culture assimilates. World 31 to 1 CE: first kingdom ~4000 BCE (Earth ~3100), first empire ~3750 BCE (Earth ~2300), ~40-50 wars per century, ~120 war and ~90 state events in the chronicle by 1 CE. Off target: empires last too long (median ~700 years as an empire; Earth 2-4 centuries), the largest state holds only 3-7% of the world (Earth 1 CE ~20-25%) and the largest city ~50,000 (Earth 200,000+); left for phase 4's technology of rule and a later tuning round |
| 2026-11-21 | Phase 2 built: cultures and language families from depth of settlement; generated languages with sound changes and cognates; farming spreads cultures, assimilation, splits; named towns and cities; the viewer (`watch` live, `replay` saved runs) with map modes and dark mode; dark graphs and maps; entity tables saved with runs. World 31: first town 5475 BCE, first city 3900 BCE (Earth ~3500), ~1,600 cultures in 50 families by 2000 CE. Still low until states arrive (phase 3): largest city at 1 CE ~25,000 (Earth 200,000-2 million), urban share ~2%, world population at 1 CE 88 million |
| 2026-11-21 | Peopling level is a run option (sparse / some / most / all); dark mode in the worldgen viewer now, in the history viewer and graphs when they're built (phase 2) |
| 2026-11-21 | Phase 1 built: peopling pre-pass, foragers from plant growth, domestication and spread of farming and herding, growth, migration, famine, drowned coasts; settings in histsim/config.py; maps command. Calibrated on worlds 31 and 22 to Earth's population (10,000-3000 BCE) and number of farming origins |
| 2026-11-21 | Phase 0 built: worldgen 1.1 (species, minerals, disease, seas, ice-age land, climate history; data version 2) and the simulator skeleton (histsim 0.1: loop, random streams, calendar, environment system, run recording and replay, graphs, CLI) |
