"""Reproducible random numbers: one independent stream per system.

Each system asks for its own stream by name, so adding a new system (or new random
draws in one system) never changes the random numbers any other system gets. The same
simulation seed always replays the same history.
"""
import zlib

import numpy as np


class RngStreams:
    def __init__(self, seed):
        self.seed = int(seed)
        self._streams = {}

    def get(self, name):
        """The random generator for `name` (created on first use, then kept)."""
        if name not in self._streams:
            key = zlib.crc32(name.encode("utf-8"))  # stable across runs, unlike hash()
            self._streams[name] = np.random.default_rng((self.seed, key))
        return self._streams[name]

    def state(self):
        """Snapshot of every stream's position (for saving and resuming runs)."""
        return {name: g.bit_generator.state for name, g in self._streams.items()}

    def restore(self, state):
        for name, s in state.items():
            self.get(name).bit_generator.state = s


def draws_at(rng, n, idx):
    """`rng.random(n)[idx]` for sorted indices `idx`, leaving `rng` exactly where
    `rng.random(n)` would, but drawing only the numbers needed when `idx` is sparse (the
    stream is skipped forward over the rest). Same numbers, same history, less work."""
    idx = np.asarray(idx, np.int64)
    if len(idx) == 0:
        if n:
            _skip(rng, [int(n)])
        return np.zeros(0)
    # Runs of wanted numbers; gaps shorter than GAP are drawn through rather than skipped
    # (drawing a few unneeded numbers is cheaper than one more skip).
    breaks = np.flatnonzero(np.diff(idx) > GAP) + 1
    starts = idx[np.concatenate([[0], breaks])]
    ends = idx[np.concatenate([breaks - 1, [len(idx) - 1]])] + 1
    if len(starts) * 600 > n:                 # dense: one draw over everything is cheaper
        return rng.random(n)[idx]
    bg = rng.bit_generator
    kept = _buffer(bg)
    out = np.empty(len(idx))
    cuts = np.searchsorted(idx, ends, side="left")            # where each block's wanted numbers end
    pos = k = 0
    for s, e, cut in zip(starts.tolist(), ends.tolist(), cuts.tolist()):
        if s > pos:
            bg.advance(s - pos)
        block = rng.random(e - s)
        out[k:cut] = block[idx[k:cut] - s]
        k = cut
        pos = e
    if n > pos:
        bg.advance(int(n - pos))
    _restore_buffer(bg, kept)
    return out


GAP = 64


def _buffer(bg):
    """The half-used 32-bit draw a generator may hold (`advance` drops it; drawing doubles
    doesn't touch it), so it can be put back."""
    st = bg.state
    return st.get("has_uint32"), st.get("uinteger")


def _restore_buffer(bg, kept):
    st = bg.state
    if (st.get("has_uint32"), st.get("uinteger")) != kept:
        st["has_uint32"], st["uinteger"] = kept
        bg.state = st


def _skip(rng, steps):
    bg = rng.bit_generator
    kept = _buffer(bg)
    for d in steps:
        bg.advance(d)
    _restore_buffer(bg, kept)


def hits(rng, where, chance):
    """`where & (rng.random(where.shape) < chance)`, drawing only where `where` is True."""
    idx = np.flatnonzero(where)
    u = draws_at(rng, where.size, idx)
    p = chance if np.ndim(chance) == 0 else np.asarray(chance).ravel()[idx]
    out = np.zeros(where.shape, bool)
    out.ravel()[idx[u < p]] = True
    return out
