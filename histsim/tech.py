"""The web of technologies (phase 4: roughly to the end of the ancient world).

Each technology has prerequisites (other technologies), needs (what a place must have to
find, take up or keep it) and a rarity (how hard it is to find). Knowledge is kept per
cell as a bit mask (bit i = TECHS[i] known), so it has a geography: it is found where
the needs are met and many connected people live, spreads to neighbours that can use it,
and fades where what keeps it alive is gone. See systems/technology.py.

Needs (per cell, recomputed every step; `_near` ones are fixed facts of the land):
  people        anyone living there
  farming       farmers                      settled     settled (not shifting) farming
  herds         cattle or horses kept        cattle      cattle kept
  horse         horses kept
  towns         a town within ~150 km        state       part of a state
  kingdom       part of a kingdom or empire
  water         on a coast or a big river    hills       steep land nearby
  copper_near / tin_near / iron_near / precious_near (gold or silver)   ore within reach
  irrigable_near                             river land that canals could water
  dry_irrigable_near                         ... in dry country, where canals are worth digging
                                             (first dug by villagers on the Mesopotamian plain,
                                             before towns: Samarra, Choga Mami ~6000 BCE)
  smiths        people who already work copper (iron is first smelted by smiths)

`needs` are what a place must have to use it (and so to take it up from neighbours),
`finds` what it must have as well to find it first (iron is found in towns, then smelted
by villagers wherever there is ore), `keeps` what keeps it alive (default: the needs);
knowledge whose keepers are gone fades within a few centuries (writing lost in Greece's
dark age, Tasmania).

`earth` is the first date on Earth (real year, negative = BCE), for calibration; `effect`
says what it will do (from phase 4 step 2 on; step 1 adds no effects).
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Tech:
    key: str
    name: str                 # English name, for graphs and the chronicle
    verb: str                 # "learn to smelt bronze": the chronicle's phrase
    group: str                # farming, metals, transport, rule, building
    prereqs: tuple = ()
    needs: tuple = ("people",)   # to use it (and so to take it up from neighbours)
    keeps: tuple = None       # to keep it (None = same as needs)
    finds: tuple = ()         # needed as well to find it in the first place
    rarity: float = 1.0       # higher = rarer (waiting time for a reference population)
    crowd: float = 0.0        # how much finding it needs many connected people (0 = not at all)
    lag: float = 0.0          # years it takes at the least, however many could find it (see LAG)
    earth: int = 0
    effect: str = ""

    @property
    def keep_needs(self):
        return self.needs if self.keeps is None else self.keeps

    @property
    def find_needs(self):
        return self.needs + self.finds


T = Tech
TECHS = (
    # ---- farming
    T("pottery", "pottery", "learn to fire pottery", "farming", needs=("people",), rarity=8,
      earth=-14000, effect="storage: famine milder"),
    T("plough", "the plough", "learn to plough with oxen", "farming", needs=("farming", "herds"),
      keeps=("farming",), rarity=4, crowd=0.3, earth=-4000, effect="rain-fed intensification ceiling up"),
    T("terraces", "terracing", "learn to terrace their hillsides", "farming", needs=("farming", "hills"),
      keeps=("farming",), rarity=4, crowd=0.3, earth=-3000, effect="steep land can be intensified"),
    T("canals", "irrigation canals", "dig irrigation canals", "farming", needs=("farming", "irrigable_near"),
      finds=("dry_irrigable_near",), rarity=2, crowd=0.3, earth=-5000,
      effect="dry irrigable land watered (canal_irrigation per cell)"),
    T("manuring", "manuring and crop rotation", "learn to manure and rotate their fields", "farming",
      ("plough",), needs=("farming",), finds=("settled",), rarity=12, crowd=0.5, earth=-1000,
      effect="rain-fed ceiling up again"),
    # ---- metals
    T("copper", "copper working", "learn to work copper", "metals", needs=("farming", "copper_near"),
      keeps=("farming",), rarity=3, crowd=0.3, earth=-5000, effect="small war and tool bonus"),
    T("bronze", "bronze", "learn to smelt bronze", "metals", ("copper",),
      needs=("farming", "copper_near", "tin_near"), finds=("towns",), rarity=4, crowd=0.8, earth=-3300,
      effect="war bonus; drives the tin trade"),
    T("iron", "iron working", "learn to smelt iron", "metals", needs=("farming", "iron_near"),
      finds=("towns", "smiths"), rarity=150, crowd=1.0, earth=-1200, effect="cheap metal: war, ploughs, clearing"),
    # ---- transport
    T("wheel", "the wheel and cart", "build the first wheeled carts", "transport", needs=("herds",),
      finds=("farming",), rarity=10, crowd=0.5, earth=-3500, effect="travel on open land cheaper"),
    T("riding", "riding", "learn to ride horses", "transport", needs=("horse",), rarity=4,
      earth=-3500, effect="herder and state reach up; raids"),
    T("chariot", "the war chariot", "build war chariots", "transport", ("wheel", "bronze"), needs=("horse",),
      finds=("towns",), rarity=6, crowd=0.8, earth=-2000, effect="war bonus on open land"),
    T("sailing", "boats and sailing", "learn to sail", "transport", needs=("water",), finds=("farming",),
      rarity=8, crowd=0.3, earth=-3500, effect="river and coastal travel and trade"),
    T("roads", "built roads", "build roads", "transport", ("wheel",), needs=("state",), finds=("kingdom",),
      rarity=10, crowd=0.8, earth=-2000, effect="travel inside a state cheaper: reach up"),
    # ---- rule
    T("counting", "counting and the calendar", "keep counts and a calendar", "rule", needs=("towns",),
      rarity=3, crowd=0.5, earth=-3500, effect="needed for bureaucracy"),
    T("writing", "writing", "invent writing", "rule", ("counting",), needs=("towns",), finds=("state",),
      rarity=40, crowd=1.8, earth=-3200, effect="records: cohesion lasts, reach up"),
    T("bureaucracy", "bureaucracy", "rule through scribes and officials", "rule", ("writing",),
      needs=("state",), finds=("kingdom",), rarity=10, crowd=1.0, earth=-2500,
      effect="overextension from a larger size"),
    T("law", "written law", "write down their laws", "rule", ("writing",), needs=("state",),
      finds=("kingdom",), rarity=15, crowd=0.8, earth=-1800, effect="rebellions rarer"),
    T("coinage", "coinage", "strike coins", "rule", ("bronze", "counting"), needs=("towns",),
      finds=("kingdom", "precious_near"), rarity=60, crowd=1.0, earth=-600, effect="trade volume up; taxes"),
    # ---- building and war
    T("walls", "city walls", "wall their towns", "building", needs=("towns",), rarity=2, crowd=0.3,
      earth=-8000, effect="town defence up"),
    T("monuments", "monumental building", "raise great monuments", "building", needs=("towns",),
      finds=("state",), rarity=5, crowd=0.8, earth=-3000, effect="prestige: cohesion"),
    T("siegecraft", "siegecraft", "master siegecraft", "building", ("walls", "bureaucracy"), needs=("state",),
      finds=("kingdom",), rarity=20, crowd=1.0, earth=-900, effect="town defence down for attackers"),
    T("armies", "professional armies", "raise professional armies", "building", ("bureaucracy", "iron"),
      needs=("state",), finds=("kingdom",), rarity=15, crowd=1.0, earth=-700, effect="war bonus, longer wars"),
)

# Years a technology takes at the least once somewhere could find it: the methods behind it
# have to mature (on Earth writing ~3200 BCE, officials ~2500, written law ~1800, coinage
# ~600 BCE). More people find it sooner, but never faster than this: the chance per year
# is h / (1 + h x lag) for a hazard h.
LAG = {"plough": 300, "terraces": 300, "canals": 300, "manuring": 1000, "copper": 300, "bronze": 300,
       "iron": 1500, "wheel": 300, "riding": 200, "chariot": 400, "sailing": 300, "roads": 700,
       "counting": 200, "writing": 400, "bureaucracy": 600, "law": 800, "coinage": 1000, "walls": 200,
       "monuments": 300, "siegecraft": 1000, "armies": 800}
TECHS = tuple(replace(t, lag=float(LAG.get(t.key, 0))) for t in TECHS)

INDEX = {t.key: i for i, t in enumerate(TECHS)}
GROUPS = ("farming", "metals", "transport", "rule", "building")
NEEDS = ("people", "farming", "settled", "herds", "cattle", "horse", "towns", "state", "kingdom",
         "water", "hills", "copper_near", "tin_near", "iron_near", "precious_near", "irrigable_near",
         "dry_irrigable_near", "smiths")
assert len(TECHS) <= 64
for _t in TECHS:
    assert all(p in INDEX and INDEX[p] < INDEX[_t.key] for p in _t.prereqs), _t.key
    assert all(n in NEEDS for n in _t.find_needs + _t.keep_needs), _t.key


def bit(key):
    return 1 << INDEX[key]


def mask_of(keys):
    m = 0
    for k in keys:
        m |= bit(k)
    return m


# Ages of metal, for maps and graphs: the best metal a place works.
STONE, COPPER, BRONZE, IRON = 0, 1, 2, 3
AGE_NAMES = {STONE: "stone", COPPER: "copper", BRONZE: "bronze", IRON: "iron"}


def age_of(known):
    """The metal age of each cell (numpy array of bit masks, or one int)."""
    import numpy as np
    known = np.asarray(known, np.uint64)
    age = np.zeros(known.shape, np.int8)
    for value, key in ((COPPER, "copper"), (BRONZE, "bronze"), (IRON, "iron")):
        age[(known & np.uint64(bit(key))) != 0] = value
    return age


def known_list(mask):
    """The technologies in one cell's bit mask, in web order."""
    mask = int(mask)
    return [t for i, t in enumerate(TECHS) if mask >> i & 1]
