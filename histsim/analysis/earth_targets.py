"""Earth's population at 1300 CE by region, blended from the standard estimates.

Calibration targets for world 0 (Earth). 1300 is used, not 1400, because the simulator
has no Black Death. Each region is a latitude/longitude box on the map (the first region
listed owns cells where boxes overlap). Its target is the median of the sources below and
its range their spread; where a source covers a different area than the box, the adjustment
is in its name. HYDE 3.3 values are the sums of its 1300 CE grid over each box.

Sources (millions), as read from Wikipedia's comparison tables (Estimates of historical
world population, Demographics of India, Medieval demography, Population history of China,
Demographic history of Japan before the Meiji Restoration, Population history of
Indigenous peoples of the Americas), November 2026:
  McEvedy & Jones 1978, Biraben 1979/1980/2005, Durand 1974, Clark 1967, Maddison 2001,
  Urlanis 1941 (Europe), Farris 2006 (Japan), Denevan 1992 and Alchon 2003 (the Americas,
  for 1492), Snow 2001, Thornton 2005, Peros 2009, Milner 2010 (North America), HYDE 3.3
  (Klein Goldewijk, Utrecht University, CC-BY).
Maddison gives only 1000 and 1500 for most regions: those are interpolated geometrically
(60% of the way, for 1300). Confidence says how many independent sources a region has.
The Americas' figures are for 1492 and lack the low counts (Kroeber, Rosenblat).
"""


def _interp(a, b, f=0.6):
    """Geometric interpolation between Maddison's 1000 and 1500 figures (f = 0.6 for 1300)."""
    return a * (b / a) ** f


_KOREA = _interp(3.9, 8.0)

# (lat min, lat max, lon min, lon max)
REGIONS = {
    "Europe (W+C)": (36, 60, -10, 30), "Russia/Ukraine": (45, 62, 30, 60), "China": (20, 42, 98, 123),
    "India": (7, 33, 66, 92), "SE Asia mainland": (8, 23, 95, 110), "Indonesia": (-9, 6, 95, 141),
    "Japan+Korea": (31, 43, 124, 146), "Middle East": (25, 42, 30, 62), "N Africa+Egypt": (20, 37, -17, 35),
    "Sub-Saharan Africa": (-35, 18, -18, 52), "Mesoamerica": (10, 23, -106, -82),
    "Andes+S America": (-56, 12, -82, -34), "N America (N of Mexico)": (23, 60, -130, -60),
}

SOURCES_1300 = {
    "Europe (W+C)": ("good", {"Urlanis 1941 (Europe 78.7 less ~7 of NE Europe outside the box)": 72.0,
                              "HYDE 3.3": 78.0}),
    "Russia/Ukraine": ("low", {"Urlanis 1941 (NE Europe 11 less Poland and the Baltics)": 6.5, "HYDE 3.3": 5.8}),
    "China": ("medium", {"modern estimate for the 1290 Yuan census": 75.3, "HYDE 3.3": 73.2,
                         "modern estimate for 1351 (upper)": 120.4}),
    "India": ("medium", {"Maddison 2001": 94.5, "Clark 1967": 75.0, "Biraben 1979": 83.0, "Durand 1974": 95.7,
                         "McEvedy & Jones 1978": 88.0, "HYDE 3.3": 137.3}),
    "SE Asia mainland": ("low", {"HYDE 3.3": 8.2,
                                 "Maddison Indochina (interp.) x2.5 for Burma and Siam (rough)": _interp(2.2, 4.5) * 2.5}),
    "Indonesia": ("low", {"Maddison (interp.)": _interp(5.2, 10.7), "HYDE 3.3": 5.3}),
    "Japan+Korea": ("medium", {"McEvedy & Jones 1978 + Korea (Maddison interp.)": 9.75 + _KOREA,
                               "Biraben 2005 + Korea": 7.0 + _KOREA, "Farris 2006 (1280) + Korea": 6.0 + _KOREA,
                               "HYDE 3.3": 11.0}),
    "Middle East": ("low", {"HYDE 3.3": 18.9,
                            "Maddison MENA less North Africa, plus Turkey (interp.)": _interp(24.0, 19.6) - 9.2 + 7.0}),
    "N Africa+Egypt": ("medium", {"Maddison (Egypt, Maghreb, Libya interp.)":
                                  sum(_interp(a, b) for a, b in ((5, 4), (2, 1.5), (2, 1.5), (1, .8), (.5, .5))),
                                  "HYDE 3.3": 11.8}),
    "Sub-Saharan Africa": ("low", {"Maddison (interp.)": _interp(21.8, 38.3), "HYDE 3.3": 43.0}),
    "Mesoamerica": ("low", {"Denevan 1992 (1492: Mexico + Central America)": 22.8, "Alchon 2003 (1492)": 22.5,
                            "HYDE 3.3": 19.3}),
    "Andes+S America": ("low", {"Denevan 1992 (1492: Andes, Amazonia, Patagonia)": 24.3, "Alchon 2003 (1492)": 21.5,
                                "HYDE 3.3": 24.8}),
    "N America (N of Mexico)": ("medium", {"Denevan 1992": 3.79, "Alchon 2003": 3.5, "Snow 2001": 3.44,
                                           "Thornton 2005": 7.0, "Peros 2009": 2.5, "Milner 2010": 3.8, "HYDE 3.3": 4.8}),
}

WORLD_1300 = {"HYDE 3.3": 456.2, "HYDE 2010": 392.0, "Biraben 1980": 432.0, "McEvedy & Jones 1978": 360.0,
              "Thomlinson 1975": 400.0}


def _median(values):
    v = sorted(values)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def targets_1300():
    """{region: {"target", "lo", "hi", "confidence", "sources"}} in millions."""
    out = {}
    for name, (conf, src) in SOURCES_1300.items():
        vals = list(src.values())
        out[name] = {"target": _median(vals), "lo": min(vals), "hi": max(vals), "confidence": conf, "sources": src}
    return out


def world_1300():
    vals = list(WORLD_1300.values())
    return {"target": _median(vals), "lo": min(vals), "hi": max(vals), "sources": WORLD_1300}


def region_owner(shape):
    """Name of the region each map cell belongs to ("" = none), for a map `shape` (rows, cols)."""
    import numpy as np
    h, w = shape
    d = 360.0 / w
    owner = np.full(shape, "", object)
    for name, (a0, a1, o0, o1) in REGIONS.items():
        sl = (slice(int((90 - a1) / d), int((90 - a0) / d)), slice(int((o0 + 180) / d), int((o1 + 180) / d)))
        sub = owner[sl]
        sub[sub == ""] = name
        owner[sl] = sub
    return owner
