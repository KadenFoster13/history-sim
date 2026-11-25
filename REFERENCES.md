# References

The simulator doesn't replay Earth's history; it models the forces that shaped it. This
list gathers the scholarship those forces are taken from, and where the numbers the runs
are checked against come from, for anyone who wants to read further.

Two kinds of use:
- **Idea:** a rule in the simulator follows this work's argument.
- **Check:** runs are compared with figures from this work, or typical of it.

Earth's ancient figures (populations, city sizes, first dates) are estimates, often with
wide ranges and disagreement between scholars. That is one reason the calibration ranges
in [PLAN.md](PLAN.md) are wide, and why runs are judged on their average across worlds and
seeds, not one by one.

## Farming and population

| Work | Use | Where |
|---|---|---|
| Ester Boserup, *The Conditions of Agricultural Growth: The Economics of Agrarian Change under Population Pressure* (1965) | Idea: farmers work land harder (shorter fallow, ploughs, manure, irrigation) as they grow denser; population drives intensification, not the other way round | `systems/farming.py` (`intensity`), `config.py`, PLAN "Farming intensification" |
| Albert J. Ammerman and Luigi Luca Cavalli-Sforza, *The Neolithic Transition and the Genetics of Populations in Europe* (1984) | Idea and check: farming spread as a "wave of advance" of about 1 km a year, by farmers moving (demic diffusion) as well as by neighbours adopting it | `systems/farming.py` (spread), `systems/culture.py` (cultures spreading with farming) |
| Jared Diamond, *Guns, Germs, and Steel* (1997) | Idea: crops and animals spread more easily along latitudes than across them; which species a land has shapes its history (no plough without draught animals) | `systems/farming.py`, `tech.py` (plough, wheel, riding need herds) |
| World population reconstructions, for example Colin McEvedy and Richard Jones, *Atlas of World Population History* (1978), and the HYDE database (Kees Klein Goldewijk et al., "Anthropogenic land use estimates for the Holocene - HYDE 3.2", *Earth System Science Data*, 2017) | Check: world population at 10,000 BCE, 5000 BCE, 3000 BCE, 1 CE and later. The targets in `config.py` (`POPULATION_TARGETS`) are rounded ranges typical of such estimates, not taken from one source | `config.py`, the population graph's grey bars |

## States, war and collapse

| Work | Use | Where |
|---|---|---|
| Robert L. Carneiro, "A Theory of the Origin of the State", *Science* 169 (1970) | Idea: farmland hemmed in by desert, mountains or sea forms states sooner, because the defeated cannot move away and become subjects | `systems/states.py` (circumscription) |
| Ibn Khaldun, *The Muqaddimah* (1377) | Idea: *asabiyyah*, the solidarity that lets a new ruling group conquer and that fades within a few generations; dynasties rise and decline in cycles | `systems/states.py` (cohesion; the expansion drive renewed by new ruling houses) |
| Peter Turchin, *Historical Dynamics: Why States Rise and Fall* (2003) and *War and Peace and War* (2006) | Idea: cohesion grows on frontiers facing other peoples and fades in secure cores; large states overextend; rise and fall in long cycles | `systems/states.py` (cohesion, overextension) |
| Kenneth E. Boulding, *Conflict and Defense: A General Theory* (1962) | Idea: the "loss-of-strength gradient": a state's power fades with distance from its centre | `systems/states.py` (strength fading with travel distance; reach) |
| James C. Scott, *The Art of Not Being Governed* (2009) and *Against the Grain* (2017) | Idea: early states grew where taxable grain farming was; hills, forests, swamps and steppe stayed outside them; collapse often left land self-governing | `systems/states.py` (land taken in for its worth, land falling back) |
| David G. Anderson, *The Savannah River Chiefdoms: Political Change in the Late Prehistoric Southeast* (1994) | Idea: "cycling" chiefdoms that rise and fragment within a few generations | `systems/states.py` (chiefdoms that dissolve) |

## Technology

| Work | Use | Where |
|---|---|---|
| Michael Kremer, "Population Growth and Technological Change: One Million B.C. to 1990", *Quarterly Journal of Economics* 108 (1993) | Idea: more people means more inventors, so discovery rises with (connected) population | `systems/technology.py` (discovery weighted by people within reach) |
| Joseph Henrich, "Demography and Cultural Evolution: How Adaptive Cultural Processes Can Produce Maladaptive Losses: The Tasmanian Case", *American Antiquity* 69 (2004) | Idea: skills need enough people to keep them; small or cut-off populations lose them | `systems/technology.py` (loss where keepers are gone) |
| First dates of technologies (writing, bronze, iron, the wheel and so on) | Check: the "Earth" column of `tech.py` and PLAN's technology table uses widely cited approximate dates from the archaeological consensus, not one source | `tech.py`, PLAN "Technology, economy and trade" |

## Towns and cities

| Work | Use | Where |
|---|---|---|
| Tertius Chandler, *Four Thousand Years of Urban Growth: An Historical Census* (1987) | Check: sizes of the largest ancient cities (Uruk, Rome and others), as rough estimates | PLAN "Towns and cities", `systems/settlements.py` |
| George Modelski, *World Cities: -3000 to 2000* (2003) | Check: the number and size of ancient cities over time | PLAN "Towns and cities" |
| George Kingsley Zipf, *Human Behavior and the Principle of Least Effort* (1949), after Felix Auerbach (1913) | Idea: the rank-size rule: a region's places follow a ladder, about ten times as many above each tenfold-smaller size | `systems/settlements.py` (`small_towns`) |

## Examples

The plan and code also name historical cases as examples of a rule (the Nile and
Mesopotamia for floodplain farming, Linear Pottery and Bantu farmers for pioneer spread,
Akkad, Qin, Rome, the Inca and the Mongols for bursts of conquest, Greece's dark age and
Tasmania for lost knowledge). These are standard textbook cases rather than references to a
particular work.

Citations are given as far as they are known for certain; corrections are welcome.
