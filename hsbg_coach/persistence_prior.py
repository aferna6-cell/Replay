"""Frozen board-slot persistence prior for Phase 2J opportunity cost.

Fitted from raw-greedy DEV traces only. Features available at decision time:
tavern-tier band, raw-stat tertile on board, target-core vs non-core.
No card-name memorization.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

METHODOLOGY_VERSION = "2j_v1"
SURVIVAL_HORIZON = 2
WEIGHT_1 = 0.5
WEIGHT_2 = 0.5
MIN_CELL_N = 25
DEFAULT_P1 = 0.55
DEFAULT_P2 = 0.40

TierBand = str  # "le4" | "5" | "6plus"
RankBand = str  # "weak" | "mid" | "strong"


def tier_band(tier: int) -> TierBand:
    if tier <= 4:
        return "le4"
    if tier == 5:
        return "5"
    return "6plus"


def report_tier_band(tier: int) -> str:
    """Reporting bands for Phase 2J outcomes (≤4 / 5 / 6)."""
    if tier <= 4:
        return "tier_le4"
    if tier == 5:
        return "tier_5"
    return "tier_6"


def raw_stats(m: Dict) -> float:
    return float((m.get("attack") or 0) + (m.get("health") or 0))


def minion_key(m: Dict) -> Tuple:
    """Identity for survival matching across turns (stats scale; names do not)."""
    return (m.get("name"), bool(m.get("golden")))


def minion_feature_key_parts(m: Dict) -> Tuple:
    """Optional richer identity; survival matching uses ``minion_key`` only."""
    return (
        m.get("name"),
        int(m.get("attack") or 0),
        int(m.get("health") or 0),
        bool(m.get("golden")),
    )


def rank_band_for_index(raws: List[float], index: int) -> RankBand:
    """Tertile of ``index`` among board raw stats (ascending = weakest first)."""
    n = len(raws)
    if n <= 0:
        return "mid"
    order = sorted(range(n), key=lambda i: (raws[i], i))
    pos = order.index(index)
    if pos < n / 3:
        return "weak"
    if pos < 2 * n / 3:
        return "mid"
    return "strong"


def feature_key(tier: int, rank: RankBand, is_core: bool) -> str:
    return f"{tier_band(tier)}|{rank}|{'core' if is_core else 'noncore'}"


@dataclass
class PersistenceCell:
    tier_band: str
    rank_band: str
    is_core: bool
    n: int
    p_survive_1: float
    p_survive_2: float

    @property
    def persistence_weight(self) -> float:
        return WEIGHT_1 * self.p_survive_1 + WEIGHT_2 * self.p_survive_2

    @property
    def key(self) -> str:
        return feature_key_from_bands(self.tier_band, self.rank_band, self.is_core)


def feature_key_from_bands(tb: str, rb: str, is_core: bool) -> str:
    return f"{tb}|{rb}|{'core' if is_core else 'noncore'}"


@dataclass
class PersistencePrior:
    methodology_version: str
    survival_horizon: int
    weight_1: float
    weight_2: float
    fit_seed_base: int
    fit_lobbies: int
    cells: Dict[str, PersistenceCell]
    global_p_survive_1: float
    global_p_survive_2: float
    collapsed_from: List[str]

    def persistence_weight(self, *, tier: int, rank: RankBand,
                           is_core: bool) -> float:
        key = feature_key(tier, rank, is_core)
        cell = self.cells.get(key)
        if cell is not None:
            return cell.persistence_weight
        for rb in (rank, "mid"):
            for core in (is_core, False):
                k = feature_key(tier, rb, core)
                if k in self.cells:
                    return self.cells[k].persistence_weight
        return (WEIGHT_1 * self.global_p_survive_1
                + WEIGHT_2 * self.global_p_survive_2)

    def to_dict(self) -> Dict:
        return {
            "methodology_version": self.methodology_version,
            "survival_horizon": self.survival_horizon,
            "weight_1": self.weight_1,
            "weight_2": self.weight_2,
            "fit_seed_base": self.fit_seed_base,
            "fit_lobbies": self.fit_lobbies,
            "global_p_survive_1": self.global_p_survive_1,
            "global_p_survive_2": self.global_p_survive_2,
            "collapsed_from": list(self.collapsed_from),
            "cells": {
                k: asdict(v) for k, v in sorted(self.cells.items())
            },
        }

    def canonical_dict(self) -> Dict:
        """Behavioral prior contents only (stable key order) for hashing."""
        return {
            "methodology_version": self.methodology_version,
            "survival_horizon": self.survival_horizon,
            "weight_1": self.weight_1,
            "weight_2": self.weight_2,
            "global_p_survive_1": self.global_p_survive_1,
            "global_p_survive_2": self.global_p_survive_2,
            "cells": {
                k: {
                    "tier_band": v.tier_band,
                    "rank_band": v.rank_band,
                    "is_core": v.is_core,
                    "n": v.n,
                    "p_survive_1": v.p_survive_1,
                    "p_survive_2": v.p_survive_2,
                }
                for k, v in sorted(self.cells.items())
            },
        }

    def content_hash_sha256(self) -> str:
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_dict(cls, d: Dict) -> "PersistencePrior":
        cells = {
            k: PersistenceCell(**{kk: vv for kk, vv in v.items()
                                  if kk in ("tier_band", "rank_band", "is_core",
                                            "n", "p_survive_1", "p_survive_2")})
            for k, v in (d.get("cells") or {}).items()
        }
        return cls(
            methodology_version=d.get("methodology_version", METHODOLOGY_VERSION),
            survival_horizon=int(d.get("survival_horizon", SURVIVAL_HORIZON)),
            weight_1=float(d.get("weight_1", WEIGHT_1)),
            weight_2=float(d.get("weight_2", WEIGHT_2)),
            fit_seed_base=int(d.get("fit_seed_base", 0)),
            fit_lobbies=int(d.get("fit_lobbies", 0)),
            cells=cells,
            global_p_survive_1=float(d.get("global_p_survive_1", DEFAULT_P1)),
            global_p_survive_2=float(d.get("global_p_survive_2", DEFAULT_P2)),
            collapsed_from=list(d.get("collapsed_from") or []),
        )

    def save(self, path: str) -> None:
        d = self.to_dict()
        d["prior_hash_sha256"] = self.content_hash_sha256()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PersistencePrior":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def empty_prior(*, fit_seed_base: int = 0, fit_lobbies: int = 0
                ) -> PersistencePrior:
    return PersistencePrior(
        methodology_version=METHODOLOGY_VERSION,
        survival_horizon=SURVIVAL_HORIZON,
        weight_1=WEIGHT_1,
        weight_2=WEIGHT_2,
        fit_seed_base=fit_seed_base,
        fit_lobbies=fit_lobbies,
        cells={},
        global_p_survive_1=DEFAULT_P1,
        global_p_survive_2=DEFAULT_P2,
        collapsed_from=[],
    )
