"""Phase 2M decision tree — route from shop/pool audit findings (2m_v2)."""

from __future__ import annotations

from typing import Dict, Optional

METHODOLOGY_VERSION = "2m_v2"

# Deal-level: observed/expected ratio band for "consistent".
RAW_RATIO_LO = 0.70
RAW_RATIO_HI = 1.30
# Lobby-clustered mean(obs−exp) CI containing 0 → consistent signal.
MIN_EXPECTED_RAW_FOR_SURPRISE = 5.0


def evaluate_phase_2m_decision(analysis: Dict) -> Dict:
    headlines = analysis.get("headlines") or {}
    rules = analysis.get("rule_mismatches") or {}
    live = analysis.get("live_calibration") or {}
    catalogue = analysis.get("catalogue_synchronization") or {}
    primary = live.get("primary_deal_level") or live

    sum_exp = float(headlines.get("live_sum_expected_raw")
                    or primary.get("sum_expected_raw") or 0.0)
    sum_obs = float(headlines.get("live_sum_observed_raw")
                    or primary.get("sum_observed_raw") or 0.0)
    sum_exp_hit = float(headlines.get("live_sum_expected_hit_probability")
                        or primary.get("sum_expected_hit_probability") or 0.0)
    sum_obs_hit = float(headlines.get("live_sum_observed_hit_deals")
                        or primary.get("sum_observed_hit_deals") or 0.0)
    raw_ratio = headlines.get("live_raw_ratio_obs_over_exp")
    if raw_ratio is None and sum_exp > 1e-12:
        raw_ratio = sum_obs / sum_exp
    hit_ratio = headlines.get("live_hit_ratio_obs_over_exp")
    if hit_ratio is None and sum_exp_hit > 1e-12:
        hit_ratio = sum_obs_hit / sum_exp_hit

    a1 = float(headlines.get("phase_2l_a1_share") or 0.0)
    a3 = float(headlines.get("phase_2l_a3_share") or 0.0)
    pct_missing_kb = float(catalogue.get("status_share", {}).get(
        "MISSING_FROM_KB") or 0.0)

    actionable_ids = list(rules.get("phase_2n_actionable_ids")
                          or rules.get("demonstrated_ids") or [])
    demonstrated_ids = list(rules.get("demonstrated_ids") or [])
    contextual_ids = list(rules.get("contextual_ids") or [])

    # Lobby-clustered CI for raw obs−exp
    clustered = (primary.get("lobby_clustered") or {})
    raw_ci = (clustered.get("raw_obs_minus_exp") or {}).get("ci95") or [None, None]
    raw_mean = (clustered.get("raw_obs_minus_exp") or {}).get("mean")
    ci_contains_zero = (
        raw_ci[0] is not None and raw_ci[1] is not None
        and raw_ci[0] <= 0.0 <= raw_ci[1])

    live_consistent = (
        raw_ratio is not None
        and RAW_RATIO_LO <= float(raw_ratio) <= RAW_RATIO_HI
        and (ci_contains_zero or abs(float(raw_ratio) - 1.0) <= 0.15)
    )
    # Undershoot: substantially fewer hits than deal-level expectation
    live_surprising = (
        sum_exp >= MIN_EXPECTED_RAW_FOR_SURPRISE
        and raw_ratio is not None
        and float(raw_ratio) < RAW_RATIO_LO
        and (raw_mean is not None and raw_mean < 0)
        and (raw_ci[1] is not None and raw_ci[1] < 0)
    )

    accounting_ids = [i for i in actionable_ids
                      if "elimination" in i or "freeze" in i]
    copy_ids = [i for i in actionable_ids if i.startswith("pool_copies")]

    base = {
        "n_demonstrated_rule_mismatches": rules.get("n_demonstrated_mismatches"),
        "n_phase_2n_actionable_mismatches": rules.get("n_phase_2n_actionable"),
        "raw_ratio_obs_over_exp": raw_ratio,
        "hit_ratio_obs_over_exp": hit_ratio,
        "live_consistent": live_consistent,
        "live_surprising": live_surprising,
        "lobby_raw_mean_obs_minus_exp": raw_mean,
        "lobby_raw_ci95": raw_ci,
        "phase_2l_a1_share": a1,
        "phase_2l_a3_share": a3,
        "pct_cores_missing_from_kb": pct_missing_kb,
        "headlines": headlines,
        "live_calibration_summary": {
            "n_deal_card_observations": primary.get("n_deal_card_observations"),
            "sum_expected_raw": sum_exp,
            "sum_observed_raw": sum_obs,
            "sum_expected_hit_probability": sum_exp_hit,
            "sum_observed_hit_deals": sum_obs_hit,
            "raw_ratio_obs_over_exp": raw_ratio,
            "hit_ratio_obs_over_exp": hit_ratio,
            "post_assembly_deal_boundary": live.get(
                "post_assembly_deal_boundary"),
        },
    }

    substantial_areas = set()
    if pct_missing_kb >= 0.15 or a1 >= 0.25:
        substantial_areas.add("catalogue_kb")
    if accounting_ids:
        substantial_areas.add("live_pool_accounting")
    if copy_ids:
        substantial_areas.add("copy_counts")
    if live_surprising and not live_consistent:
        substantial_areas.add("live_draw_discrepancy")

    if len(substantial_areas) >= 2:
        return {
            "decision_branch": "multiple_substantial_mismatches",
            "recommended_next_step": (
                "Phase 2N: fix independently in scoped interventions — "
                "do not bundle into one 'better shops' patch. Order by "
                "mass impact: catalogue/KB sync (A1; classify missing cores "
                "against current active pool before adding), then lifecycle "
                "accounting (death return / freeze top-up), then T6 copy "
                "counts, then re-measure deal-level live calib. "
                f"Contextual/out-of-scope (not 2N bugs): {contextual_ids}."),
            "substantial_areas": sorted(substantial_areas),
            "demonstrated_mismatch_ids": demonstrated_ids,
            "phase_2n_actionable_ids": actionable_ids,
            "contextual_ids": contextual_ids,
            **base,
        }

    if pct_missing_kb >= 0.15 or a1 >= 0.50:
        return {
            "decision_branch": "catalogue_kb_mismatch_dominates",
            "recommended_next_step": (
                "Phase 2N: synchronize card/core data — classify each missing "
                "entry (active missing / outdated / rename / token / bad "
                "mapping) before KB add or core removal."),
            "demonstrated_mismatch_ids": demonstrated_ids,
            "phase_2n_actionable_ids": actionable_ids,
            **base,
        }

    if accounting_ids and live_surprising:
        return {
            "decision_branch": "live_pool_accounting_bug",
            "recommended_next_step": (
                "Phase 2N: fix shared-pool lifecycle (elimination return, "
                "freeze top-up) then re-run deal-level live calibration."),
            "demonstrated_mismatch_ids": demonstrated_ids,
            "phase_2n_actionable_ids": actionable_ids,
            **base,
        }

    if copy_ids and live_surprising:
        return {
            "decision_branch": "shop_draw_probabilities_rules_mismatch",
            "recommended_next_step": (
                "Phase 2N: fix generation model / copy counts to match "
                "current Battlegrounds rules, then re-measure."),
            "demonstrated_mismatch_ids": demonstrated_ids,
            "phase_2n_actionable_ids": actionable_ids,
            **base,
        }

    if live_consistent:
        draw_note = (
            "Deal-level live calibration is consistent with exact pre-deal "
            "pool expectation — `_draw()` is not implicated. "
            if live_consistent else "")
        return {
            "decision_branch": "scarcity_consistent_with_live_expectation",
            "recommended_next_step": (
                f"{draw_note}"
                "Still apply scoped 2N fixes for actionable mismatches "
                f"({actionable_ids}) without a bundled generation rewrite."),
            "demonstrated_mismatch_ids": demonstrated_ids,
            "phase_2n_actionable_ids": actionable_ids,
            **base,
        }

    if live_surprising:
        return {
            "decision_branch": "shop_draw_probabilities_rules_mismatch",
            "recommended_next_step": (
                "Phase 2N: deal-level observed hits substantially undershoot "
                "exact live-pool expectation — investigate draw path before "
                "declaring generation healthy."),
            "demonstrated_mismatch_ids": demonstrated_ids,
            "phase_2n_actionable_ids": actionable_ids,
            **base,
        }

    return {
        "decision_branch": "inconclusive_expand_or_inspect",
        "recommended_next_step": (
            "Inspect deal-level calibration and actionable mismatch list; "
            "do not implement Phase 2N from a weak story."),
        "demonstrated_mismatch_ids": demonstrated_ids,
        "phase_2n_actionable_ids": actionable_ids,
        **base,
    }
