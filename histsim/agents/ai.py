"""Decisions: every state chooses its actions through one step, `decide(situation)`.

The built-in AI below is the only decider for now; a human player will later be
another implementation of the same step for one state, through the viewer.

A Situation is everything a ruler can see when deciding: its own strength and wars,
and each neighbour's relative strength along the shared border, whether they are at war,
under a truce or allied. Actions are tuples:
    ("war", target_state_id)
    ("annex", target_state_id)   a much weaker neighbour submits without war
    ("ally", other_state_id)
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Neighbour:
    id: int
    border: float               # shared border length, in ~50 km cells (800-px map cells)
    ratio: float                # our strength / theirs along that border
    at_war: bool                # we are at war with them
    truce: bool                 # a truce forbids war for now
    allied: bool
    their_wars: int             # wars they are fighting (with anyone)
    same_family: bool           # their rulers' language is related to ours
    threat: float               # how much stronger they are than us overall (their power / ours)


@dataclass
class Situation:
    id: int
    year: int
    power: float
    cohesion: float
    aggression: float           # this state's temper, 0.2 (cautious) .. 2 (warlike)
    wars: int                   # wars we are fighting now
    neighbours: list = field(default_factory=list)


class StateAI:
    """Rulers who go to war when they are clearly stronger along a border, more readily
    when the enemy is busy elsewhere, and who seek allies against a much stronger
    neighbour."""

    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng

    def decide(self, s: Situation, years: float):
        cfg = self.cfg
        actions = []
        if s.wars < cfg.max_wars_per_state:
            best, best_desire = None, 0.0
            for nb in s.neighbours:
                if nb.at_war or nb.truce or nb.allied:
                    continue
                desire = max(0.0, np.log(max(nb.ratio, 1e-9)) - cfg.war_margin)
                if nb.ratio > 0.8:                         # rivals of similar strength fight too
                    desire += cfg.rivalry
                desire *= s.aggression * (1.6 if nb.their_wars else 1.0) * (0.7 if nb.same_family else 1.0)
                desire *= min(1.0, nb.border / 4)          # a few border cells: little to gain
                if desire > best_desire:
                    best, best_desire = nb, desire
            if best is not None:
                chance = 1 - np.exp(-best_desire * years / cfg.war_interval_years)
                if self.rng.random() < chance:
                    actions.append(("war", best.id))
        for nb in s.neighbours:                           # small neighbours may submit
            if (not nb.at_war and nb.ratio >= cfg.submit_ratio and nb.threat <= 1 / cfg.submit_ratio
                    and self.rng.random() < cfg.submit_chance):
                actions.append(("annex", nb.id))
                break
        for nb in s.neighbours:                           # seek protection against a giant
            if nb.threat > cfg.alliance_threat and not nb.at_war:
                for friend in s.neighbours:
                    if (friend.id != nb.id and not friend.at_war and not friend.allied
                            and friend.ratio > 0.3 and self.rng.random() < cfg.alliance_chance):
                        actions.append(("ally", friend.id))
                        break
                break
        return actions
