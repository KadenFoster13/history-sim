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
| Run time | Hours are acceptable for a full run |
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

### Performance
About 70,000 land cells x 12,000+ years. Systems work on whole arrays at once (numpy);
loops that can't be vectorised use numba, as the generator's river routing does.
Rough budget: a full run in 1-3 hours; early millennia much faster than recent ones.

## 5. Starting the world (year 0)

### The peopling pre-pass (before year 0)
A quick setup, not a year-by-year simulation of the Old Stone Age:

1. **Origin:** humans start in one warm, open, well-watered region on a large landmass.
2. **Ice-age world:** sea level is lowered by about 60-120 m during the pass, opening
   land bridges across shallow seas; ice sheets block the far north and south.
3. **Spread:** a travel-time calculation from the origin across land (easy in
   grassland and open woodland, slow in jungle, desert, mountains and tundra) and across
   short sea crossings of up to ~100-150 km (enough for an Australia-like crossing).
4. **Result at year 0:** hunter-gatherer bands everywhere that was reachable, at
   densities set by biome (roughly 0.01-0.1 people per km2; ~4 million people in total
   on an Earth-like world). Remote islands and ice-covered land start empty, waiting for
   later seafarers.
5. **Depth of settlement:** how long ago each region was reached sets its starting
   cultural and language diversity (older regions are more diverse).

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
| 4 | Technology web and its spread; economy, resources and trade routes | Bronze and iron ages emerging, trade networks |
| 5 | Religions and ideologies; key figures and dynasties; the chronicle | Faiths founded and spreading, named rulers, a written history |
| 6 | Disease, epidemics, climate shocks, disasters; nomads and migrations; collapse cycles | Plagues, droughts, steppe invasions, dark ages |
| 7 | Navigation, exploration, island settlement, colonisation | Explorers crossing oceans, colonial empires |
| 8 | Industrialisation, modern states and ideologies, world wars, globalisation, the near future | The modern world taking shape |
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

## 12. Decision log

| Date | Decision |
|---|---|
| 2026-11-21 | Simulate on worldgen map cells (800 px), 4-direction moves, rivers on edges |
| 2026-11-21 | Watch live, save and replay, graphs after a run; player control in a later phase |
| 2026-11-21 | Societies plus key figures |
| 2026-11-21 | Technology blend: real physics, flexible paths; invented culture, religion, ideology; generated names |
| 2026-11-21 | Year 0 ~ 10,000 BCE; peopling pre-pass places humans, wild species set by worldgen 1.1 |
| 2026-11-21 | Generated languages per culture, drifting and splitting |
| 2026-11-21 | Full runs may take hours |
| 2026-11-21 | Calendar: year 0 is exactly 10,000 BCE; year 10,000 = 1 CE; 2025 CE = year 12,024 |
| 2026-11-21 | Phase 0 built: worldgen 1.1 (species, minerals, disease, seas, ice-age land, climate history; data version 2) and the simulator skeleton (histsim 0.1: loop, random streams, calendar, environment system, run recording and replay, graphs, CLI) |
