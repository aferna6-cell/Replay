"""Restricted comparison against a checkpoint that fails the DEV protocol.

``ml.benchmark`` refuses to score an episode that has not terminated within
``MAX_DECISIONS``, so a policy that stalls on even one of the fixed DEV
lobbies has **no defined DEV score**. Experiment 3 hit exactly that: training
seed 1's 5,120-episode policy loops on 5 of the 1000 greedy lobbies and 2 of
the 500 mixed ones, and
``scripts/ppo_multiseed_protocol_failure.py`` records the failure rather than
inventing a placement. That treatment is the primary one and is not relaxed
here: the unscoreable checkpoint stays out of the headline tables.

This module supplies the *supplementary* reading. Question B asks for
iter320 − iter80 on every training seed, and for seed 1 the only honest
answer short of "undefined" is the paired difference over the lobbies both
checkpoints actually finished. That number exists, so it is computed — with
its bias stated every time it is produced:

    the excluded lobbies are precisely the ones where the policy degenerated,
    so a restricted estimate flatters the failing checkpoint.

DEV split only; nothing here re-runs games or reads TEST.
"""

import json
from typing import Dict, List, Mapping, Optional, Sequence

from .analyze_benchmark import paired_diff

BIAS_NOTE = (
    "restricted to the DEV lobbies both checkpoints finished. The excluded "
    "lobbies are the ones the failing checkpoint could not terminate, so "
    "this estimate flatters that checkpoint and is NOT a benchmark result.")


def load_protocol_failure(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    if "n_non_terminating" not in blob:
        raise ValueError(f"{path}: not a protocol-failure diagnostic")
    return blob


def completed_indices(diagnostic: Mapping) -> List[int]:
    """Game indices the failing checkpoint did finish, in evaluation order."""
    base = diagnostic["seed_range"][0]
    attempted = int(diagnostic["games_attempted"])
    stalled = {int(s) - base for s in diagnostic["non_terminating_game_seeds"]}
    idx = [i for i in range(attempted) if i not in stalled]
    placements = diagnostic["completed_games_diagnostic"]["placements"]
    if len(idx) != len(placements):
        raise ValueError(
            f"diagnostic is inconsistent: {len(idx)} lobbies implied "
            f"complete but {len(placements)} placements recorded")
    return idx


def restricted_pair(diagnostic: Mapping, scored: Mapping,
                    seed: int = 0) -> Dict:
    """Paired difference (failing checkpoint − scored checkpoint) over the
    lobbies the failing checkpoint finished.

    ``scored`` is an ordinary DEV result JSON for the same field and seed
    block, so its placement list is indexed by game order.
    """
    base = diagnostic["seed_range"][0]
    if scored["base_seed"] != base:
        raise ValueError("the two runs use different DEV base seeds")
    if scored["field"] != diagnostic["field"]:
        raise ValueError("the two runs use different opponent fields")
    attempted = int(diagnostic["games_attempted"])
    if scored["games"] != attempted:
        raise ValueError(f"the scored run covers {scored['games']} lobbies, "
                         f"the diagnostic attempted {attempted}")
    idx = completed_indices(diagnostic)
    pa = list(diagnostic["completed_games_diagnostic"]["placements"])
    pb = [scored["placements"][i] for i in idx]
    out = paired_diff(pa, pb, seed=seed)
    lo, hi = out["ci95"]
    out.update({
        "a": f"iter{diagnostic['ppo_iteration']:03d} (UNSCOREABLE)",
        "b": scored.get("agent"),
        "field": diagnostic["field"],
        "games_attempted": attempted,
        "games_paired": len(idx),
        "games_dropped": attempted - len(idx),
        "restricted": True,
        "ci_excludes_zero": not (lo <= 0 <= hi),
        "status": "restricted supplement",
        "bias_note": BIAS_NOTE,
    })
    return out


def restricted_pairs(diagnostic: Mapping, scored_by_iter: Mapping[int, Mapping],
                     references: Optional[Sequence[int]] = None,
                     seed: int = 0) -> Dict[str, Dict]:
    """The failing checkpoint against each reference budget, restricted."""
    refs = list(references if references is not None else sorted(scored_by_iter))
    it = int(diagnostic["ppo_iteration"])
    return {f"iter{it}-iter{ref}": restricted_pair(diagnostic,
                                                   scored_by_iter[ref],
                                                   seed=seed)
            for ref in refs}
