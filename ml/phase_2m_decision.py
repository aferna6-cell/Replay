"""Phase 2M decision tree — route from shop/pool audit findings."""

from __future__ import annotations

from typing import Dict, Optional

METHODOLOGY_VERSION = "2m_v1"

# How far observed zero-rate may sit above expected before we call it a gap.
ZERO_RATE_GAP_TOLERANCE = 0.05
# Expected hits that would make all-observed-zeros surprising.
MIN_EXPECTED_RAW_FOR_SURPRISE = 5.0


def evaluate_phase_2m_decision(analysis: Dict) -> Dict:
    headlines = analysis.get("headlines") or {}
    rules = analysis.get("rule_mismatches") or {}
    live = analysis.get("live_calibration") or {}
    catalogue = analysis.get("catalogue_synchronization") or {}

    obs_z = headlines.get("live_observed_zero_offer_rate")
    exp_z = headlines.get("live_expected_zero_offer_rate")
    sum_exp = float(headlines.get("live_sum_expected_raw") or 0.0)
    sum_obs = float(headlines.get("live_sum_observed_raw") or 0.0)
    a1 = float(headlines.get("phase_2l_a1_share") or 0.0)
    a3 = float(headlines.get("phase_2l_a3_share") or 0.0)
    n_mismatch = int(rules.get("n_demonstrated_mismatches") or 0)
    pct_missing_kb = float(catalogue.get("status_share", {}).get(
        "MISSING_FROM_KB") or 0.0)

    zero_gap: Optional[float] = None
    if obs_z is not None and exp_z is not None:
        zero_gap = float(obs_z) - float(exp_z)

    live_consistent = (
        zero_gap is not None
        and zero_gap <= ZERO_RATE_GAP_TOLERANCE
        and sum_obs <= sum_exp + 1.0
    )
    live_surprising = (
        zero_gap is not None
        and (zero_gap > ZERO_RATE_GAP_TOLERANCE
             or (sum_exp >= MIN_EXPECTED_RAW_FOR_SURPRISE and sum_obs == 0.0))
    )

    demonstrated_ids = list(rules.get("demonstrated_ids") or [])
    accounting_ids = [i for i in demonstrated_ids
                      if "elimination" in i or "freeze" in i]
    copy_ids = [i for i in demonstrated_ids if i.startswith("pool_copies")]
    slot_ids = [i for i in demonstrated_ids
                if i.startswith("shop_slots") and "spell_era" in i]

    base = {
        "n_demonstrated_rule_mismatches": n_mismatch,
        "zero_rate_gap": zero_gap,
        "live_consistent": live_consistent,
        "live_surprising": live_surprising,
        "phase_2l_a1_share": a1,
        "phase_2l_a3_share": a3,
        "pct_cores_missing_from_kb": pct_missing_kb,
        "headlines": headlines,
        "live_calibration_summary": {
            "n_card_windows": live.get("n_card_windows"),
            "observed_zero_offer_rate": obs_z,
            "expected_zero_offer_rate": exp_z,
            "sum_expected_raw_live": sum_exp,
            "sum_observed_raw": sum_obs,
        },
    }

    # Decision tree (priority order from Phase 2M brief).
    substantial_areas = set()
    if pct_missing_kb >= 0.15 or a1 >= 0.25:
        substantial_areas.add("catalogue_kb")
    if accounting_ids:
        substantial_areas.add("live_pool_accounting")
    if copy_ids or (slot_ids and False):  # spell-era slots alone ≠ gen rewrite
        substantial_areas.add("copy_counts")
    if live_surprising and not live_consistent:
        substantial_areas.add("live_draw_discrepancy")

    if len(substantial_areas) >= 2:
        return {
            "decision_branch": "multiple_substantial_mismatches",
            "recommended_next_step": (
                "Phase 2N: fix independently in scoped interventions — "
                "do not bundle into one 'better shops' patch. Order by "
                "mass impact: catalogue/KB sync (A1), then lifecycle "
                "accounting (death return / freeze top-up), then copy "
                "counts, then re-measure live calib."),
            "substantial_areas": sorted(substantial_areas),
            "demonstrated_mismatch_ids": demonstrated_ids,
            **base,
        }

    if pct_missing_kb >= 0.15 or a1 >= 0.50:
        return {
            "decision_branch": "catalogue_kb_mismatch_dominates",
            "recommended_next_step": (
                "Phase 2N: synchronize card/core data (KB + archetype cores) "
                "before rewriting shop generation."),
            "demonstrated_mismatch_ids": demonstrated_ids,
            **base,
        }

    if accounting_ids and live_surprising:
        return {
            "decision_branch": "live_pool_accounting_bug",
            "recommended_next_step": (
                "Phase 2N: fix shared-pool lifecycle (elimination return, "
                "freeze top-up) then re-run live calibration."),
            "demonstrated_mismatch_ids": demonstrated_ids,
            **base,
        }

    if copy_ids and live_surprising:
        return {
            "decision_branch": "shop_draw_probabilities_rules_mismatch",
            "recommended_next_step": (
                "Phase 2N: fix generation model / copy counts to match "
                "current Battlegrounds rules, then re-measure."),
            "demonstrated_mismatch_ids": demonstrated_ids,
            **base,
        }

    if live_consistent:
        return {
            "decision_branch": "scarcity_consistent_with_live_expectation",
            "recommended_next_step": (
                "Shop generation is not clearly broken under the live pool "
                "model. Investigate roll/opportunity horizon and core-set "
                "assumptions. Still document rule mismatches "
                f"({', '.join(demonstrated_ids)}) for scoped 2N follow-ups "
                "without a bundled generation rewrite."),
            "demonstrated_mismatch_ids": demonstrated_ids,
            **base,
        }

    if live_surprising:
        return {
            "decision_branch": "shop_draw_probabilities_rules_mismatch",
            "recommended_next_step": (
                "Phase 2N: live observed zeros exceed exact live-pool "
                "expectation — investigate draw path / pool accounting "
                "before a broad generation rewrite."),
            "demonstrated_mismatch_ids": demonstrated_ids,
            **base,
        }

    return {
        "decision_branch": "inconclusive_expand_or_inspect",
        "recommended_next_step": (
            "Inspect live calibration breakdowns and rule-mismatch list; "
            "do not implement Phase 2N from a weak story."),
        "demonstrated_mismatch_ids": demonstrated_ids,
        **base,
    }
