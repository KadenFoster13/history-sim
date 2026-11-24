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
  of 10,000 (or twice the people) become kingdoms; 3 million people (or 1.5 million mostly
  of other peoples) make an empire; a single city with its fields is a city-state.
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
  it); farmland in a state feeds 25% more; the rulers' culture slowly assimilates subjects.
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
| 4 | Technology web and its spread (including canals: `canal_irrigation`); economy, resources and trade routes | Bronze and iron ages emerging, trade networks |
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
  World 31 at 1 CE, all with the 800 px wildlife (worldgen 1.5):

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
  little more state turnover there now. Seed 4 is peopled late at 400 px but not at
  800 px (the peopling homeland and crossings depend on the map); seed 5 is peopled late
  at both.

## 12. Decision log

| Date | Decision |
|---|---|
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
