"""Decision categories for the env's fixed 28-action space.

Experiment 1 measured *how much* a PPO checkpoint disagrees with the greedy
expert (15-25% of decisions) but not *which* decisions change. The mapping
below is derived directly from ``hsbg_coach.bg_env``'s action constants —
not invented — so a category is a pure function of the action index:

    0..6    buy shop slot i        -> "buy"
    7..16   play hand slot i       -> "play"
    17..23  sell board slot i      -> "sell"
    24      roll                   -> "roll"
    25      tier up                -> "level"
    26      freeze (toggle)        -> "freeze"
    27      end turn               -> "end"

No observation context is needed: the env's action space is positional and
context-free at this granularity.
"""

from typing import Dict, List, Sequence

from hsbg_coach.bg_env import (A_BUY0, A_END, A_FREEZE, A_LEVEL, A_PLAY0,
                               A_ROLL, A_SELL0, N_ACTIONS, N_BUY, N_PLAY,
                               N_SELL)

CATEGORIES = ["buy", "play", "sell", "roll", "level", "freeze", "end"]


def action_category(action: int) -> str:
    """Category of one action index. Raises on an out-of-space index."""
    a = int(action)
    if A_BUY0 <= a < A_BUY0 + N_BUY:
        return "buy"
    if A_PLAY0 <= a < A_PLAY0 + N_PLAY:
        return "play"
    if A_SELL0 <= a < A_SELL0 + N_SELL:
        return "sell"
    if a == A_ROLL:
        return "roll"
    if a == A_LEVEL:
        return "level"
    if a == A_FREEZE:
        return "freeze"
    if a == A_END:
        return "end"
    raise ValueError(f"action {action} outside the {N_ACTIONS}-action space")


def confusion(reference: Sequence[int], compared: Sequence[int]) -> Dict:
    """Category-level accounting of how `compared` differs from `reference`.

    Returns the confusion matrix (reference category -> compared category ->
    count), the per-reference-category disagreement share ("of all states
    where the reference chose buy, what fraction did the other policy change
    away?"), and each category's contribution to TOTAL disagreements (which
    categories the drift is actually made of).
    """
    if len(reference) != len(compared):
        raise ValueError(f"action lists must be equal length "
                         f"(got {len(reference)} and {len(compared)})")
    matrix: Dict[str, Dict[str, int]] = {c: {} for c in CATEGORIES}
    totals: Dict[str, int] = {c: 0 for c in CATEGORIES}
    disagree: Dict[str, int] = {c: 0 for c in CATEGORIES}
    n_disagree = 0
    for r, c in zip(reference, compared):
        rc, cc = action_category(r), action_category(c)
        matrix[rc][cc] = matrix[rc].get(cc, 0) + 1
        totals[rc] += 1
        if int(r) != int(c):
            disagree[rc] += 1
            n_disagree += 1
    return {
        "n_states": len(reference),
        "n_disagreements": n_disagree,
        "overall_agreement": 1.0 - (n_disagree / len(reference)) if reference else None,
        "confusion_matrix": matrix,
        "reference_category_counts": totals,
        # "of the states where the reference chose X, what share changed"
        "disagreement_share_by_category": {
            c: (disagree[c] / totals[c]) if totals[c] else None
            for c in CATEGORIES},
        # "what fraction of ALL disagreements came from reference category X"
        "contribution_to_total_drift": {
            c: (disagree[c] / n_disagree) if n_disagree else None
            for c in CATEGORIES},
    }


def top_transitions(conf: Dict, limit: int = 10) -> List[Dict]:
    """The largest reference->compared category changes, biggest first."""
    rows = [{"from": rc, "to": cc, "count": n}
            for rc, tos in conf["confusion_matrix"].items()
            for cc, n in tos.items() if rc != cc]
    rows.sort(key=lambda r: -r["count"])
    return rows[:limit]
