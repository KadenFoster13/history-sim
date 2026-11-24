"""Every tunable setting for the history simulator lives here.

Like worldgen's config.py: change a default here and it applies everywhere. Values are
in real units where possible (people per km2, years, km). Comments say what each does.
"""
from dataclasses import dataclass, replace


@dataclass
class SimConfig:
    # ---------------------------------------------------------------- timing
    step_years: int = 5                 # population and farming update every this many years
    # The rules were tuned on 800-px worlds (cells ~50 km across). On other map sizes every
    # rule that works cell by cell is rescaled to give the same speeds in km per year and
    # the same sizes in km2 (see systems/geo.py: cell_scale).
    reference_width: int = 800          # map width (px) the rules were tuned on

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
    farmer_density: float = 1.2         # people per km2 on good land with a mature early crop package, farmed
                                        # the first way: shifting (slash-and-burn) fields with long fallows
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
    river_corridor: float = 6.0         # ... this much more (x) along great rivers
    coast_corridor: float = 3.0         # ... and along coasts
    pioneer_rate: float = 0.001         # yearly chance per frontier cell on a river or coast that
    pioneer_km: tuple = (50, 300)       #   pioneers settle good land this far ahead
    farming_advantage: float = 2.5      # foragers take up farming readily once it would feed this many
    frontier_floor: float = 0.04        #   times more people than foraging (less: this share of the pace)
    flood_yield: float = 1.0            # floodplain harvests up to (1 + this) times others (yearly silt)
    canal_irrigation: float = 0.0       # how far canals water irrigable land, 0..1 (technology, phase 4)
    tsetse_block: float = 0.5           # tsetse above this keeps cattle and horses out
    # Intensification (Boserup): crowded farmers work their land harder over the centuries
    # (shorter fallow, ploughs, manure, terraces, irrigation), so old farmland feeds many
    # times more people than new. `intensity` (0..1) per cell multiplies what farmland
    # feeds, up to its ceiling.
    rainfed_ceiling: float = 6.0        # most a rain-fed field can be intensified (x)
    irrigated_ceiling: float = 30.0     # ... a floodplain, or dry irrigable land under a state (canals)
    paddy_ceiling: float = 1.5          # ... and this much more where the rice-like grain grows well
    intensify_years: float = 2500.0     # time scale of intensification under full pressure
    intensify_pressure: float = 0.6     # it starts once people reach this share of what the land feeds
    intensify_density: float = 2.0      # ... at full pace with this many people per km2 (faster with more, up to 3x)
    intensity_start: float = 0.1        # how slowly it begins (growth ~ this + intensity: compounding)
    intensity_neighbour: float = 1.0    # faster next to more intensive land (methods spread)
    state_intensify: float = 1.0        # faster inside a state (order, works, markets): x(1 + this)
    intensity_decay_years: float = 400.0  # half-life of intensity on emptied or abandoned land
    conquest_intensity_loss: float = 0.1  # share lost when a cell is taken in war

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

    # ------------------------------------------------------------ states
    state_town_min: float = 2000.0      # a town this big may become a chiefdom's seat
    state_formation_rate: float = 0.008 # yearly chance per stateless town (more for bigger towns)
    state_delay_years: float = 4000.0   # ... once farming there is this old (chiefs need surplus, time)
    chief_density: float = 1.5          # stateless farmers this dense (per km2) may also form a chiefdom
    chief_rate: float = 0.0005          # ... with this yearly chance per ~50 km cell
    circumscription_states: float = 4.0 # hemmed-in farmland (by desert, mountains, sea) forms states (1 + this)x as often
    circumscription_sooner: float = 0.4 # ... and up to this share sooner
    circumscription_km: float = 200.0   # how far around a cell counts as its surroundings
    circumscription_arable: float = 0.35  # farmland at least this good counts as good
    herder_state_density: float = 1.0   # herders per km2 for a confederation to form among them
    herder_state_rate: float = 0.01     # yearly chance one forms somewhere, where dense enough
    herder_military: float = 1.5        # herders with horses count this much extra in war
    reach_km: float = 400.0             # how far a state's grip reaches (km of easy travel; grows
                                        # with its size; rough country counts as longer)
    reach_growth: float = 2.0           # reach x (1 + this x log10(1 + people / 100,000))
    chiefdom_reach: float = 0.4        # chiefdoms reach only this share as far (~100 km)
    chiefdom_seed_cells: float = 0.5    # a new chiefdom: its seat and this many ~50 km cells around it
    chiefdom_fade: float = 1.0          # chiefs' hold fades this much faster than a kingdom's
    chiefdom_dissolve: float = 0.1     # a chiefdom whose cohesion falls below this dissolves
    chiefdom_war: float = 0.3           # chiefs go to war this much less readily (they raid)
    absorb_rate: float = 0.1           # yearly chance a strong state takes in a stateless village
    absorb_resistance: float = 1.6      # how hard stateless people resist (x square root of people per
                                        # 2,500 km2 at their density)
    worth_density: float = 0.25          # land is worth taking in fully with this many taxable people
                                        # per km2: farmers count fully,
    worth_foragers: float = 0.05        # ... foragers hardly,
    worth_herders: float = 0.3          # ... herders in part
    fallback_worth: float = 0.3         # land worth less than this beyond a weak state's reach may
    fallback_years: float = 300.0       # ... go back to governing itself (typical wait), and is let
                                        # go when a state collapses
    drive_years: float = 150.0          # a state's drive to expand fades over about this long
    dynasty_years: float = 150.0        # typical wait for a new ruling house to revive it
    digest_years: float = 50.0          # recent gains are digested over about this long ...
    digest_share: float = 0.3           # ... and gaining this share of the state halves the drive
    drive_floor: float = 0.2            # eagerness of an old or sated state (a young one: up to 2 - this)
    expand_from: float = 0.3           # a state takes in new land only while its drive is above this,
    expand_ramp: float = 0.25           # ... fully from this much above it
    cohesion_gain: float = 0.02         # cohesion growth on frontiers with other peoples (per year)
    cohesion_loss: float = 0.004        # cohesion fading in a state's quiet core (per year)
    overextension_people: float = 10_000_000.0  # ... twice as fast for a state this big, 3x at twice that
    decision_years: int = 10            # rulers weigh war and alliances this often
    war_interval_years: float = 50.0    # typical wait before a clearly stronger state attacks
    war_margin: float = 0.2             # how much stronger (log ratio) a state must feel to attack
    rivalry: float = 0.03               # ... though neighbours of similar strength fight now and then
    max_wars_per_state: int = 2
    min_war_people: float = 20_000.0    # smaller states don't start wars
    war_edge: float = 1.5               # how much strength counts along a front (higher: big states win more)
    war_rate: float = 0.08              # yearly chance a front cell changes hands (even fight: half)
    war_years: float = 18.0             # typical length of a war
    truce_years: tuple = (30, 100)      # truce after a war, years (random in this range)
    conquest_loss: float = 0.05         # share of a cell's people lost when it is conquered
    hill_defence: float = 0.6           # defence bonuses (log strength) for rough ground,
    river_defence: float = 0.4          # ... a great river,
    town_defence: float = 0.3           # ... walls (per doubling of town size),
    capital_defence: float = 1.0        # ... and the capital
    submit_ratio: float = 6.0           # a neighbour this much weaker (along the border and overall) ...
    submit_chance: float = 0.15         # ... may submit without war (chance each time rulers decide)
    alliance_threat: float = 3.0        # a neighbour this many times stronger makes states seek allies
    alliance_chance: float = 0.3        # chance a threatened state forms an alliance when it decides
    enclave_absorb: float = 0.5         # chance per check an enclave joins the state around it
    statelet_absorb: float = 0.3        # ... and a one- or two-cell statelet its strongest neighbour
    admin_years: int = 25               # how often distances, rebellions and kinds are checked
    related_gap: float = 0.4            # restlessness of subjects of a related culture
    foreign_gap: float = 0.6            # ... and of an unrelated one
    secession_threshold: float = 1.9    # restlessness (distance / reach + culture gap, x weakness)
                                        # above which a region may break away
    rebellion_years: float = 500.0
    collapse_cohesion: float = 0.12     # a big state whose cohesion falls below this falls apart
    collapse_min_cells: int = 30        # ... if it has at least this many cells
    collapse_cells_per_successor: float = 30.0   # roughly one successor state per this many cells (2-8)     # typical wait before a restless region breaks away
    min_rebel_cells: int = 4            # smallest region that can break away
    kingdom_people: float = 150_000.0   # people for a state to count as a kingdom
    kingdom_capital_min: float = 10_000.0  # ... with a capital city at least this big (or twice the people)
    empire_people: float = 3_500_000.0  # ... and as an empire
    empire_mixed: float = 0.8           # ... or this share of that, if mostly of other peoples
    empire_keep: float = 0.5            # an empire is one until it falls below this share of the bar
    empire_world: float = 100e6         # ... when the world holds this many people; the bar moves with
    empire_growth: float = 0.5          # (world people / empire_world) to this power
    largest_state_event_years: int = 300
    notable_share: float = 0.04         # chronicle wars and rebellions of states with this share of the
                                        # world's people (and at least 300,000)
    capital_share: float = 0.015        # share of a state's people drawn to its capital
    state_food_bonus: float = 0.0       # farmland in a state feeds this much more (now through
                                        # intensification: states speed it up, see state_intensify)
    ruling_prestige: float = 3.0        # the rulers' culture is this much stronger in its state

    # ------------------------------------------- technology of rule and war (phase 4 step 2)
    # A state uses what its capital's people know.
    copper_war: float = 0.1             # army strength + this with copper weapons,
    bronze_war: float = 0.25            # ... bronze (the best metal counts),
    iron_war: float = 0.4               # ... iron;
    chariot_war: float = 0.2            # ... and + this with chariots,
    armies_war: float = 0.3             # ... professional armies
    writing_reach: float = 0.2          # reach x (1 + these): records and messengers,
    roads_reach: float = 0.5            # ... roads,
    riding_reach: float = 0.2           # ... riders,
    bureaucracy_reach: float = 0.5      # ... officials
    bureaucracy_hold: float = 4.0       # overextension begins this much later (x (1 + this)) with
    writing_hold: float = 0.5           # ... bureaucracy and writing
    law_cohesion: float = 0.25          # written law slows cohesion loss by this share,
    monument_cohesion: float = 0.15     # ... monuments by this share
    law_loyalty: float = 0.3            # written law lowers provinces' restlessness by this much
    walls_defence: float = 0.5          # walls add this share to town and capital defence
    siege_breach: float = 0.6           # siegecraft takes away this share of town and capital defence

    # --------------------------------------------------------- technology (histsim/tech.py)
    tech_every: int = 10                # technology is found, spread and lost every this many years
    tech_years: float = 1000.0          # a rarity-1 technology is found in about this many years by a
                                        # million people (with a million more within reach)
    tech_contact_km: float = 300.0      # people within this distance count as in contact
    tech_town_boost: float = 3.0        # people near towns find things this much more readily (+)
    tech_independent_km: float = 1500.0 # a find this far from anyone who knows it is a new invention
    tech_spread_km_year: float = 2.0    # how fast knowledge spreads overland
    tech_corridor: float = 2.0          # ... this much faster (+) along coasts and big rivers
    tech_state_spread: float = 1.0      # ... and inside a state
    tech_loss_half_life: float = 150.0  # knowledge whose keepers are gone halves in this many years
    tech_town_km: float = 150.0         # "towns" need: a town within this distance
    copper_reach_km: float = 500.0      # ore within reach (before trade carries it further)
    tin_reach_km: float = 2000.0
    iron_reach_km: float = 300.0
    precious_reach_km: float = 1000.0   # gold or silver, for coinage
    irrigable_reach_km: float = 50.0

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
