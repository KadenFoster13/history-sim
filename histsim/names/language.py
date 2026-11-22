"""Generated languages: sound systems, words, sound changes and names.

A language family starts with a proto-language: an inventory of consonants and vowels,
the syllable shapes it allows, how long its words are, and how it is written. Its words
are generated from those sounds (the same meaning always gives the same word).

When a community splits, the daughter language inherits its parent's words through a
few regular sound changes (p becomes f, final vowels drop, k before i becomes ch...)
and replaces a share of its vocabulary. So related languages share recognisable
cognates (Tamar, Tamor, Thamr) while unrelated ones look nothing alike.

Names of places and peoples are built from a language's words (big + river, new +
town) or from fresh roots in its sounds, and written in the family's spelling.
"""
import zlib

import numpy as np

# Every sound a language can have, with its default spelling.
CONSONANTS = {
    "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "g": "g", "q": "q",
    "f": "f", "v": "v", "s": "s", "z": "z", "sh": "sh", "zh": "zh", "th": "th", "dh": "dh",
    "x": "kh", "gh": "gh", "h": "h", "ch": "ch", "j": "j", "ts": "ts",
    "m": "m", "n": "n", "ng": "ng", "ny": "ny", "l": "l", "r": "r", "w": "w", "y": "y",
}
VOWELS = {"a": "a", "e": "e", "i": "i", "o": "o", "u": "u",
          "aa": "aa", "ee": "ee", "ii": "ii", "oo": "oo", "uu": "uu",
          "ai": "ai", "au": "au", "ei": "ei", "ou": "ou"}
FRONT = {"i", "e", "ii", "ee", "ei"}
STOPS = {"p", "b", "t", "d", "k", "g", "q"}
LIQUIDS = {"l", "r"}
NASALS = {"m", "n", "ng", "ny"}

# Sounds a proto-language may have beyond the basic m n t k, with how often.
OPTIONAL_CONSONANTS = [
    ("p", .8), ("s", .85), ("l", .75), ("r", .75), ("d", .6), ("g", .55), ("b", .55),
    ("h", .5), ("y", .5), ("w", .45), ("sh", .4), ("f", .3), ("v", .3), ("z", .25),
    ("ch", .25), ("j", .2), ("x", .2), ("ng", .25), ("ts", .12), ("th", .15), ("q", .08),
    ("ny", .1), ("zh", .06), ("gh", .08), ("dh", .06),
]
EXTRA_VOWELS = [("e", .8), ("o", .8), ("aa", .2), ("ii", .12), ("uu", .12),
                ("ai", .25), ("au", .2), ("ei", .12), ("ou", .1)]

# The meanings place and people names are built from.
MEANINGS = ["people", "land", "home", "town", "fort", "place", "river", "water", "ford",
            "spring", "lake", "sea", "bay", "island", "hill", "mountain", "stone", "forest",
            "field", "valley", "marsh", "desert", "plain", "big", "small", "new", "old",
            "high", "white", "black", "red", "green", "golden", "north", "south", "east",
            "west", "sun", "moon", "sky", "fire", "salt", "grain", "cattle", "horse", "deer",
            "fish", "bird", "wolf", "oak", "god", "king", "mother", "father", "first"]

# Regular sound changes a daughter language may undergo:
# (from, to, where); where = any | between (two vowels) | final | initial | front (before i/e).
SOUND_CHANGES = [
    ("p", "f", "any"), ("p", "b", "between"), ("t", "d", "between"), ("k", "g", "between"),
    ("k", "x", "any"), ("x", "h", "any"), ("h", "", "any"), ("t", "th", "any"),
    ("th", "t", "any"), ("th", "s", "any"), ("b", "v", "between"), ("d", "dh", "between"),
    ("dh", "d", "any"), ("dh", "z", "any"), ("g", "gh", "between"), ("gh", "", "any"),
    ("g", "y", "between"), ("s", "h", "initial"), ("s", "sh", "front"), ("k", "ch", "front"),
    ("t", "ts", "front"), ("g", "j", "front"), ("ts", "s", "any"), ("ch", "sh", "any"),
    ("j", "zh", "any"), ("v", "w", "any"), ("w", "v", "any"), ("w", "b", "initial"),
    ("y", "j", "initial"), ("z", "r", "between"), ("l", "r", "any"), ("r", "l", "any"),
    ("ny", "n", "any"), ("ng", "n", "final"), ("m", "n", "final"), ("q", "k", "any"),
    ("f", "h", "initial"), ("sh", "s", "any"), ("zh", "j", "any"), ("b", "p", "final"),
    ("d", "t", "final"), ("g", "k", "final"),
    ("a", "o", "any"), ("a", "e", "any"), ("o", "u", "any"), ("e", "i", "any"),
    ("u", "o", "any"), ("i", "e", "any"), ("e", "a", "any"), ("o", "a", "any"),
    ("aa", "a", "any"), ("ee", "i", "any"), ("ii", "i", "any"), ("oo", "u", "any"),
    ("uu", "u", "any"), ("ai", "e", "any"), ("au", "o", "any"), ("ei", "i", "any"),
    ("ou", "u", "any"), ("a", "aa", "final"), ("o", "oo", "any"), ("e", "ei", "any"),
]
# Changes to the shape of words.
SHAPE_CHANGES = ["lose_final_vowel", "lose_final_consonant", "add_final_vowel",
                 "simplify_clusters", "lose_initial_vowel"]

# Spelling habits a family may have (each replaces the default spelling of some sounds).
ORTHOGRAPHIES = [
    {},
    {"k": "c", "x": "h", "sh": "sh"},
    {"sh": "sch", "ch": "tsch", "y": "j", "j": "dj", "x": "ch"},
    {"sh": "x", "ch": "c", "ts": "tz", "x": "j", "j": "dj"},
    {"aa": "á", "ee": "é", "ii": "í", "oo": "ó", "uu": "ú", "sh": "š", "ch": "č", "zh": "ž"},
    {"aa": "ā", "ee": "ē", "ii": "ī", "oo": "ō", "uu": "ū", "y": "y", "x": "kh"},
    {"k": "c", "w": "u", "y": "i", "th": "th", "sh": "ss"},
    {"aa": "â", "ee": "ê", "oo": "ô", "ny": "ñ", "ng": "ng", "q": "qu"},
]


def _rng(*parts):
    """A random generator fixed by its parts (the same parts always give the same draws)."""
    key = zlib.crc32(":".join(str(p) for p in parts).encode("utf-8"))
    return np.random.default_rng(key)


class Language:
    """One language: a proto-language (no parent) or a daughter with sound changes."""

    def __init__(self, lang_id, seed, parent=None, rng=None, name=""):
        self.id = lang_id
        self.seed = int(seed)
        self.parent = parent
        self.name = name
        self.family = parent.family if parent else lang_id
        self.depth = parent.depth + 1 if parent else 0
        self._words = {}
        if parent is None:
            self._make_phonology(_rng("phonology", seed))
            self.changes = []
            self.replace_rate = 0.0
        else:
            self.consonants = parent.consonants
            self.vowels = parent.vowels
            self.onsets, self.codas = parent.onsets, parent.codas
            self.syllables, self.length = parent.syllables, parent.length
            self.spelling = dict(parent.spelling)
            self.patterns = parent.patterns
            self.sounds = set(parent.sounds)
            self.changes = self._pick_changes(rng or _rng("changes", seed))
            self.replace_rate = 0.12
            # The daughter's sounds: the parent's after the changes that apply everywhere.
            for frm, to, where in self.changes:
                if where == "any" and frm in self.sounds:
                    self.sounds.discard(frm)
                    if to:
                        self.sounds.add(to)

    # ----------------------------------------------------------- proto-language
    def _make_phonology(self, rng):
        cons = ["m", "n", "t", "k"] + [c for c, p in OPTIONAL_CONSONANTS if rng.random() < p]
        vows = ["a", "i", "u"] + [v for v, p in EXTRA_VOWELS if rng.random() < p]
        self.consonants, self.vowels = cons, vows
        # Some sounds are much more common than others (Zipf-like weights).
        self.c_weights = _zipf(len(cons), rng)
        self.v_weights = _zipf(len(vows), rng)
        # Syllable shapes: (C)V(C) with optional clusters.
        self.onsets = rng.random() < 0.35          # stop + l/r clusters at the start (tra, kle)
        allowed = [c for c in cons if c in NASALS | LIQUIDS | {"s", "t", "k", "sh"}]
        self.codas = list(rng.choice(allowed, size=min(len(allowed), int(rng.integers(0, 5))),
                                     replace=False)) if allowed else []
        self.syllables = [("CV", 1.0), ("V", 0.25 * rng.random()), ("CVC", 0.6 * rng.random() if self.codas else 0)]
        self.length = [0.15 + 0.4 * rng.random(), 0.5, 0.15 + 0.35 * rng.random()]   # 1, 2, 3 syllables
        self.spelling = {**CONSONANTS, **VOWELS, **ORTHOGRAPHIES[int(rng.integers(len(ORTHOGRAPHIES)))]}
        self.sounds = set(cons) | set(vows)
        # How this family likes to build place names (weights of the patterns in place_name()).
        self.patterns = rng.dirichlet([1.5, 1.0, 1.0, 1.2, 0.8])

    def _syllable(self, rng, first):
        shapes, w = zip(*self.syllables)
        w = np.array(w) / sum(w)
        shape = shapes[rng.choice(len(shapes), p=w)]
        if shape == "V" and not first:
            shape = "CV"            # no vowel pile-ups inside words
        out = []
        if "C" in shape[:1]:
            out.append(self.consonants[rng.choice(len(self.consonants), p=self.c_weights)])
            if self.onsets and out[0] in STOPS and rng.random() < 0.2:
                liquid = [c for c in self.consonants if c in LIQUIDS]
                if liquid:
                    out.append(liquid[rng.integers(len(liquid))])
        out.append(self.vowels[rng.choice(len(self.vowels), p=self.v_weights)])
        if shape.endswith("VC") and self.codas:
            out.append(self.codas[rng.integers(len(self.codas))])
        return out

    def _fresh_root(self, rng, syllables=None):
        """A new word in the proto-language's sounds."""
        root = self
        while root.parent is not None:
            root = root.parent
        if syllables is None:
            p = np.array(root.length) / sum(root.length)
            syllables = 1 + rng.choice(3, p=p)
        word = []
        for i in range(syllables):
            word += root._syllable(rng, i == 0)
        # At most one long vowel or diphthong per word.
        fancy = [i for i, s in enumerate(word) if len(s) == 2 and s in VOWELS]
        for i in fancy[1:]:
            word[i] = word[i][0]
        return word

    # -------------------------------------------------------------- daughters
    def _pick_changes(self, rng):
        usable = [c for c in SOUND_CHANGES if c[0] in self.sounds]
        n = int(rng.integers(2, 5))
        picks = [usable[i] for i in rng.choice(len(usable), size=min(n, len(usable)), replace=False)]
        if rng.random() < 0.45:
            picks.append(("shape", SHAPE_CHANGES[int(rng.integers(len(SHAPE_CHANGES)))], "any"))
        return picks

    def evolve(self, word):
        """Apply this language's own sound changes to a word inherited from its parent."""
        for frm, to, where in self.changes:
            if frm == "shape":
                word = _shape_change(word, to, self)
            else:
                word = _sound_change(word, frm, to, where)
        return word or ["a"]

    def _inherit(self, word):
        """A proto-language word as it sounds in this language (all changes since the root)."""
        if self.parent is None:
            return word
        return self.evolve(self.parent._inherit(word))

    # ------------------------------------------------------------------ words
    def word(self, meaning):
        """The word for `meaning` (a list of sounds); the same every time."""
        if meaning not in self._words:
            if self.parent is None:
                w = self._fresh_root(_rng("word", self.seed, meaning))
            elif _rng("replace", self.seed, meaning).random() < self.replace_rate:
                w = self._inherit(self._fresh_root(_rng("word", self.seed, meaning)))   # a new word
            else:
                w = self.evolve(self.parent.word(meaning))
            self._words[meaning] = w
        return self._words[meaning]

    def root(self, rng, syllables=None):
        """A new name-like root in this language's sounds (no particular meaning)."""
        return self._inherit(self._fresh_root(rng, syllables))

    def spell(self, sounds, capital=True):
        text = "".join(self.spelling.get(s, s) for s in sounds)
        return text[:1].upper() + text[1:] if capital else text

    # ------------------------------------------------------------------ names
    def people_name(self, rng):
        """A name for a people (often their word for 'people' or a root)."""
        if rng.random() < 0.3:
            name = _tidy(self.spell(self.word("people")))
            if len(name) >= 3:
                return name
        for _ in range(20):
            name = _tidy(self.spell(self.root(rng, syllables=int(rng.integers(1, 3)) + 1)))
            if len(name) >= 3:
                break
        return name

    def place_name(self, rng, features=()):
        """A place name from the land around it (river, hill, sea...) and this language."""
        for _ in range(20):
            name = self._place_name(rng, features)
            if len(name) >= 4:
                break
        return name

    def _place_name(self, rng, features):
        feature = features[int(rng.integers(len(features)))] if features else "field"
        town = self.word("town" if rng.random() < 0.6 else "place")
        pattern = rng.choice(5, p=self.patterns)
        if pattern == 0:      # root + town word: Kessador
            parts = self.root(rng) + _short(town)
        elif pattern == 1:    # adjective + feature: Whitewater-like
            adj = ["big", "new", "old", "high", "white", "black", "red", "green", "golden", "first"]
            parts = self.word(adj[int(rng.integers(len(adj)))]) + _short(self.word(feature))
        elif pattern == 2:    # feature + town: Riverton-like
            parts = self.word(feature) + _short(town)
        elif pattern == 3:    # a bare root: Ulmo
            parts = self.root(rng, syllables=int(rng.integers(2, 4)))
        else:                 # root + feature: Kessa-river
            parts = self.root(rng, syllables=1) + self.word(feature)
        return _tidy(self.spell(parts))

    def to_dict(self):
        return {"id": self.id, "name": self.name, "parent": self.parent.id if self.parent else None,
                "family": self.family, "seed": self.seed,
                "changes": [list(c) for c in self.changes]}


# ------------------------------------------------------------------ helpers
def _zipf(n, rng):
    w = 1 / np.arange(1, n + 1) ** 0.9
    w = w[rng.permutation(n)] if n > 4 else w
    # keep the basic sounds (first four) common
    w[:min(4, n)] = np.maximum(w[:min(4, n)], 0.5)
    return w / w.sum()


def _is_vowel(s):
    return s in VOWELS


def _sound_change(word, frm, to, where):
    out = []
    n = len(word)
    for i, s in enumerate(word):
        if s != frm:
            out.append(s)
            continue
        prev_v = i > 0 and _is_vowel(word[i - 1])
        next_v = i + 1 < n and _is_vowel(word[i + 1])
        ok = (where == "any" or (where == "between" and prev_v and next_v)
              or (where == "final" and i == n - 1) or (where == "initial" and i == 0)
              or (where == "front" and i + 1 < n and word[i + 1] in FRONT))
        if not ok:
            out.append(s)
        elif to:
            out.append(to)
    return out


def _shape_change(word, kind, lang):
    if kind == "lose_final_vowel" and len(word) > 3 and _is_vowel(word[-1]) and not _is_vowel(word[-2]):
        return word[:-1]
    if kind == "lose_final_consonant" and len(word) > 2 and not _is_vowel(word[-1]):
        return word[:-1]
    if kind == "add_final_vowel" and word and not _is_vowel(word[-1]):
        return word + ["a" if "a" in lang.sounds else "e"]
    if kind == "simplify_clusters":
        out = []
        for s in word:
            if out and not _is_vowel(s) and not _is_vowel(out[-1]):
                continue
            out.append(s)
        return out
    if kind == "lose_initial_vowel" and len(word) > 3 and _is_vowel(word[0]):
        return word[1:]
    return word


def _short(word):
    """A word cut to its first syllable or so, as when it ends a compound (-ton, -by)."""
    for i in range(1, len(word)):
        if _is_vowel(word[i - 1]) and not _is_vowel(word[i]):
            return word[:i + 1]
    return word


def _tidy(name):
    """Readable names: no triple letters, not too long."""
    out = []
    for ch in name:
        if len(out) >= 2 and out[-1].lower() == ch.lower() and out[-2].lower() == ch.lower():
            continue
        out.append(ch)
    name = "".join(out)
    vowels = "aeiouáéíóúāēīōūâêô"
    out, run = [], 0
    for ch in name:                      # at most two vowel letters in a row
        run = run + 1 if ch.lower() in vowels else 0
        if run <= 2:
            out.append(ch)
    name = "".join(out)
    if len(name) > 13:
        name = name[:13]
        while len(name) > 4 and name[-1].lower() in "aeiouáéíóúāēīōū" and name[-2].lower() in "aeiouáéíóúāēīōū":
            name = name[:-1]
    return name[:1].upper() + name[1:]
