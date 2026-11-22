"""Every tunable setting for the history simulator lives here.

Like worldgen's config.py: change a default here and it applies everywhere. Values are
in real units where possible (people per km2, years, km). Comments say what each does.
"""
from dataclasses import dataclass, replace


@dataclass
class SimConfig:
    # ---------------------------------------------------------------- timing
    step_years: int = 5                 # population and farming update every this many years

    # ------------------------------------------------- peopling (before year 0)
    out_of_homeland_year: int = -60_000 # when people first leave their homeland (simulation years)
    settled_by_year: int = -1_000       # by then nearly everywhere reachable has been reached
    max_sea_crossing_km: float = 300.0  # longest open-water crossing managed before year 0
    sea_crossing_difficulty_km: float = 50.0  # crossings get much slower beyond this distance from shore
    ancient_voyage_km: float = 1500.0   # rare ancient voyages settle large empty lands this close to settled land
    voyage_min_area: float = 0.003      # ... if at least this share of the globe (~1.5 million km2)
    max_voyages: int = 20               # most ancient voyages in one world
    glacial_sea_level_m: float = -80.0  # sea level during the spread (land bridges open)

    # ---------------------------------------------------------- foragers
    # Hunter-gatherer density follows plant growth (from temperature and rainfall), less in
    # dense forest where little is edible, more where fish and shellfish are plentiful.
    forager_density_per_npp: float = 0.00004   # people per km2 per g/m2/yr of plant growth
    coast_bonus: float = 1.0            # extra foragers on sea coasts (x1 + this)
    river_bonus: float = 0.6            # extra foragers by big rivers and lakes
    forager_growth: float = 0.002       # yearly growth rate where there's room (falls to 0 at capacity)

    # ----------------------------------------------------------- farming
    farmer_density: float = 2.0         # people per km2 on good land with a mature early crop package
    herder_density: float = 0.6         # people per km2 living off herds on open grazing land
    farmer_growth: float = 0.004        # yearly growth rate where there's room
    crop_maturity_years: float = 2500.0 # years for a newly tamed crop to reach full yield (x difficulty)

    # Domestication: chance per year for each suitable, crowded forager cell, before
    # adjusting for how hard the species is to tame and how crowded the cell is.
    domestication_rate: float = 2.5e-7
    farmer_domestication_bonus: float = 4.0   # people who already farm tame new species more readily
    holocene_start_year: int = 300      # before this the climate is too unstable for farming
    # Spread of farming and herding to neighbouring cells (about 1 km per year historically).
    spread_rate: float = 0.012          # yearly chance a suitable neighbouring cell takes it up
    tsetse_block: float = 0.5           # tsetse above this keeps cattle and horses out

    # ---------------------------------------------------------- cultures
    # At year 0: language families and cultures per million km2 of settled land (more
    # of them, and smaller, where people have lived longest and are most numerous).
    families_per_mkm2: float = 0.25
    cultures_per_mkm2: float = 4.0
    family_min_km: float = 600.0        # family homelands at least this far apart
    culture_seed_min_km: float = 120.0  # culture homelands at least this far apart
    # How strongly a culture holds and spreads (x people per km2), by way of life.
    forager_prestige: float = 1.0
    herder_prestige: float = 3.0        # mobile, organised, often warlike
    farmer_prestige: float = 2.0
    farmers_replace: float = 0.75       # chance farming arriving among foragers brings the farmers' culture
    assimilation_rate: float = 0.0005   # yearly pull of a much stronger neighbour (x log of strength ratio)
    cohesion_rate: float = 0.01         # yearly pull on a cell surrounded (3+ sides) by another culture
    culture_check_years: int = 50       # how often cultures are checked for splits and endings
    culture_radius_km: float = 400.0    # a culture reaching further than this from its centre may split
                                        # (twice as far for herders)
    culture_split_years: float = 500.0  # typical wait before an over-wide culture splits
    isolation_split_years: float = 1500.0 # typical wait before a cut-off piece becomes its own culture
    min_culture_cells: int = 6          # smallest cut-off piece that can become a culture

    # ------------------------------------------------------- settlements
    town_min: float = 2000.0            # people in a place to count as a (named) town
    town_delay_years: float = 2000.0    # years of farming before towns can start to appear
    town_ramp_years: float = 3000.0     # ... and before they reach their full potential
    hinterland_km: float = 160.0        # a central place draws on the land about this far around it
    urban_share_max: float = 0.15       # most of a dense hinterland's people that live in its central places
    urban_density_ref: float = 3.0      # hinterland people per km2 for the full urban share
    central_place_share: float = 0.6    # share of those townspeople in the main place
    river_site: float = 0.5             # site bonus on a river (counted twice for great rivers)
    coast_site: float = 0.4             # site bonus on a sea coast
    incumbency: float = 0.5             # an existing town's advantage over a new site
    settlement_years: float = 150.0     # how quickly places grow or shrink toward their potential
    largest_city_event_years: int = 300  # least years between "largest city" events

    # --------------------------------------------------------- migration
    migration_rate: float = 0.08        # yearly share of crowding pressure that moves to neighbours
    famine_halving_years: float = 10.0  # how fast people beyond what the land can feed die off

    def copy(self, **changes) -> "SimConfig":
        return replace(self, **changes)


DEFAULT = SimConfig()

# How much of the world people have reached by year 0 (10,000 BCE), for emptier or fuller
# worlds:  python -m histsim run --peopling sparse
# Each level changes a few peopling settings above. In a survey of 40 worlds, "some" left
# at least one large landmass empty in about 7 worlds in 10.
PEOPLING_LEVELS = {
    "sparse": {"max_sea_crossing_km": 150.0, "ancient_voyage_km": 0.0},  # short hops only, no voyages
    "some": {},                                                           # the defaults above
    "most": {"ancient_voyage_km": 2500.0},                                # longer voyages
    "all": {"ancient_voyage_km": float("inf"), "voyage_min_area": 0.0002,  # every landmass of
            "max_voyages": 200},                                           # ~100,000 km2 or more
}


def peopling(level: str, cfg: SimConfig = DEFAULT) -> SimConfig:
    """`cfg` with one of the PEOPLING_LEVELS applied."""
    if level not in PEOPLING_LEVELS:
        raise ValueError(f"peopling level must be one of {list(PEOPLING_LEVELS)}, not {level!r}")
    return cfg.copy(**PEOPLING_LEVELS[level])

# Earth's history, for checking runs: (simulation year, low, typical, high) world population.
# Simulation year 0 = 10,000 BCE; 10,000 = 1 CE.
POPULATION_TARGETS = [
    (0, 1e6, 4e6, 10e6),          # 10,000 BCE
    (5_000, 2e6, 10e6, 40e6),     # 5000 BCE
    (7_000, 7e6, 25e6, 60e6),     # 3000 BCE
    (10_000, 100e6, 250e6, 500e6),  # 1 CE (needs later phases: cities, iron, empires)
    (11_500, 200e6, 450e6, 800e6),  # 1500 CE
    (11_800, 400e6, 1e9, 2e9),     # 1800 CE
    (12_000, 2e9, 6e9, 12e9),      # 2000 CE
]
