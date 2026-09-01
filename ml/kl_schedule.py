"""Fixed KL anchoring schedules for PPO training.

Pre-specified schedules only — no adaptive rules based on DEV results.

    from ml.kl_schedule import parse_kl_schedule, EXPERIMENT_6_KL_SCHEDULE
    coef = parse_kl_schedule(EXPERIMENT_6_KL_SCHEDULE)
    beta = coef(161)  # iteration is 1-based
"""

from __future__ import annotations

from typing import Callable, List, Tuple

# Experiment 6 frozen schedule:
#   iterations 1–160:  β = 0.030
#   iterations 161–320: linear 0.030 → 0.010
EXPERIMENT_6_KL_SCHEDULE = "0.03@160,0.01@320"


def parse_kl_schedule(spec: str) -> Callable[[int], float]:
    """Parse ``hold@end,final@end`` into a 1-based iteration → β function.

    ``0.03@160,0.01@320`` means β=0.03 for iterations 1–160 inclusive, then
    linear β=0.03→0.01 for iterations 161–320 inclusive.
    """
    segments: List[Tuple[float, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if "@" not in part:
            raise ValueError(f"invalid schedule segment: {part!r}")
        beta_s, end_s = part.split("@", 1)
        segments.append((float(beta_s), int(end_s)))
    if len(segments) != 2:
        raise ValueError(
            "schedule requires exactly two segments: hold@end,final@end")
    hold_beta, hold_end = segments[0]
    final_beta, final_end = segments[1]
    if hold_end >= final_end:
        raise ValueError("first segment end must be before final end")
    if hold_end < 1:
        raise ValueError("hold segment must cover at least iteration 1")

    def coef(iter_1based: int) -> float:
        if iter_1based <= 0:
            return hold_beta
        if iter_1based <= hold_end:
            return hold_beta
        if iter_1based >= final_end:
            return final_beta
        span = final_end - hold_end - 1
        t = (iter_1based - hold_end - 1) / span
        return hold_beta + t * (final_beta - hold_beta)

    coef.schedule_spec = spec  # type: ignore[attr-defined]
    return coef


def schedule_table(spec: str, checkpoints: Tuple[int, ...]) -> List[dict]:
    """β values at selected 1-based iterations (for contract manifests)."""
    coef = parse_kl_schedule(spec)
    return [{"iteration": it, "kl_coef": coef(it)} for it in checkpoints]
