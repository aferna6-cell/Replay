"""Fit Phase 2J slot-persistence prior from raw-greedy DEV traces."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from hsbg_coach.build_path import infer_target
from hsbg_coach.persistence_prior import (
    DEFAULT_P1,
    DEFAULT_P2,
    MIN_CELL_N,
    METHODOLOGY_VERSION,
    PersistenceCell,
    PersistencePrior,
    feature_key_from_bands,
    minion_key,
    rank_band_for_index,
    raw_stats,
    tier_band,
)


def _multiset(board: List[Dict]) -> Dict[Tuple, int]:
    counts: Dict[Tuple, int] = defaultdict(int)
    for m in board:
        counts[minion_key(m)] += 1
    return counts


def _still_present(key: Tuple, counts: Dict[Tuple, int]) -> bool:
    return counts.get(key, 0) > 0


def fit_persistence_prior_from_traces(
        traces: Dict, *,
        fit_seed_base: int,
        fit_lobbies: int,
        min_cell_n: int = MIN_CELL_N) -> PersistencePrior:
    """Estimate P(survive 1/2 recruit turns) from greedy turn summaries."""
    # Index: (lobby, seat, turn) -> board_before_recruit + tier
    by_lst: Dict[Tuple[int, int, int], Dict] = {}
    for ts in traces.get("turn_summaries") or []:
        key = (ts["lobby"], ts["seat"], ts["turn"])
        by_lst[key] = ts

    # Accumulators: feature_key -> [n, survive1, survive2]
    acc: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    global_n = global_s1 = global_s2 = 0.0

    for (lobby, seat, turn), ts in by_lst.items():
        board = ts.get("board_before_recruit") or []
        if not board:
            continue
        tier = int(ts.get("tavern_tier") or 1)
        fit = infer_target(board)
        core_names = set((fit.arch.core if fit is not None else {}) or {})
        raws = [raw_stats(m) for m in board]

        next1 = by_lst.get((lobby, seat, turn + 1))
        next2 = by_lst.get((lobby, seat, turn + 2))
        if next1 is None:
            continue
        counts1 = _multiset(next1.get("board_before_recruit") or [])
        counts2 = (_multiset(next2.get("board_before_recruit") or [])
                   if next2 is not None else None)

        # Consume matches without double-counting identical keys.
        remaining1 = dict(counts1)
        remaining2 = dict(counts2) if counts2 is not None else None

        for i, m in enumerate(board):
            key = minion_key(m)
            rb = rank_band_for_index(raws, i)
            is_core = m.get("name") in core_names
            fk = feature_key_from_bands(tier_band(tier), rb, is_core)

            s1 = 1.0 if remaining1.get(key, 0) > 0 else 0.0
            if s1:
                remaining1[key] -= 1
            if remaining2 is None:
                # Require both horizons observed.
                continue
            s2 = 1.0 if remaining2.get(key, 0) > 0 else 0.0
            if s2:
                remaining2[key] -= 1

            acc[fk][0] += 1
            acc[fk][1] += s1
            acc[fk][2] += s2
            global_n += 1
            global_s1 += s1
            global_s2 += s2

    g_p1 = (global_s1 / global_n) if global_n else DEFAULT_P1
    g_p2 = (global_s2 / global_n) if global_n else DEFAULT_P2

    cells: Dict[str, PersistenceCell] = {}
    collapsed: List[str] = []
    for fk, (n, s1, s2) in sorted(acc.items()):
        tb, rb, core_s = fk.split("|")
        is_core = core_s == "core"
        if n < min_cell_n:
            collapsed.append(fk)
            continue
        cells[fk] = PersistenceCell(
            tier_band=tb, rank_band=rb, is_core=is_core,
            n=int(n),
            p_survive_1=s1 / n,
            p_survive_2=s2 / n,
        )

    # Fill sparse cells by pooling within tier_band + rank (ignore core),
    # then within tier_band only.
    for fk, (n, s1, s2) in sorted(acc.items()):
        if fk in cells:
            continue
        tb, rb, core_s = fk.split("|")
        is_core = core_s == "core"
        pool_n = pool_s1 = pool_s2 = 0.0
        for other, (on, os1, os2) in acc.items():
            otb, orb, _ = other.split("|")
            if otb == tb and orb == rb:
                pool_n += on
                pool_s1 += os1
                pool_s2 += os2
        if pool_n < min_cell_n:
            pool_n = pool_s1 = pool_s2 = 0.0
            for other, (on, os1, os2) in acc.items():
                otb, _, _ = other.split("|")
                if otb == tb:
                    pool_n += on
                    pool_s1 += os1
                    pool_s2 += os2
        if pool_n >= min_cell_n:
            cells[fk] = PersistenceCell(
                tier_band=tb, rank_band=rb, is_core=is_core,
                n=int(pool_n),
                p_survive_1=pool_s1 / pool_n,
                p_survive_2=pool_s2 / pool_n,
            )
            collapsed.append(f"{fk}->pooled")
        else:
            collapsed.append(f"{fk}->global")

    return PersistencePrior(
        methodology_version=METHODOLOGY_VERSION,
        survival_horizon=2,
        weight_1=0.5,
        weight_2=0.5,
        fit_seed_base=fit_seed_base,
        fit_lobbies=fit_lobbies,
        cells=cells,
        global_p_survive_1=g_p1,
        global_p_survive_2=g_p2,
        collapsed_from=collapsed,
    )
