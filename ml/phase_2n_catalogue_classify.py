"""Phase 2N-A: classify Phase 2M problematic cores after KB refresh."""

from __future__ import annotations

import json
import os
from collections import Counter

from hsbg_coach import cards as cards_mod
from hsbg_coach.bg_env import MAX_TIER, TRIBES, build_pool

OUT = "results/sim_fidelity_phase_2n/catalogue_classification.json"


def main() -> None:
    with open("results/sim_fidelity_phase_2m/catalogue_sync.json", encoding="utf-8") as f:
        old = json.load(f)
    missing_names = old["catalogue_synchronization"]["missing_from_kb_names"]
    old_bad = [r for r in old["rows"] if r["status"] != "IN_EXACT_CATALOGUE"]

    kb = cards_mod.load_kb()
    by_name = {}
    for ck in kb.values():
        if ck.name and ck.name not in by_name:
            by_name[ck.name] = ck
    full = {m.name for m in build_pool(lobby_tribes=list(TRIBES))}

    rows = []
    for name in missing_names:
        ck = by_name.get(name)
        if ck is None:
            cls, action = "OUTDATED_OR_ABSENT_FROM_HSJSON", "remove_or_replace_core"
        elif ck.tier is None or not (1 <= int(ck.tier) <= MAX_TIER):
            cls = "TIER_OUT_OF_SIM_SCOPE"
            action = "remove_from_archetype_cores"
        elif name in full:
            cls = "ACTIVE_MISSING_FROM_KB_FIXED_BY_REFRESH"
            action = "none_cores_unchanged"
        else:
            cls, action = "IN_KB_BUT_BUILD_POOL_EXCLUDED", "investigate_filters"
        rows.append({
            "card": name,
            "classification": cls,
            "action": action,
            "kb_tier": None if ck is None else ck.tier,
            "kb_card_id": None if ck is None else ck.card_id,
            "in_exact_catalogue_after_refresh": name in full,
        })

    seen_tier = set()
    for r in old_bad:
        if r["status"] != "MISSING_OR_INVALID_TIER":
            continue
        name = r["card"]
        key = (r["archetype_key"], name)
        if key in seen_tier:
            continue
        seen_tier.add(key)
        ck = by_name.get(name)
        if ck and 1 <= int(ck.tier) <= MAX_TIER and name in full:
            cls, action = "TIER_REFRESH_FIXED", "none_cores_unchanged"
        elif ck and (ck.tier is None or int(ck.tier) > MAX_TIER):
            cls, action = "TIER_OUT_OF_SIM_SCOPE", "remove_from_archetype_cores"
        else:
            cls, action = "OTHER", "review"
        rows.append({
            "card": name,
            "archetype_key": r["archetype_key"],
            "classification": cls,
            "action": action,
            "kb_tier": None if ck is None else ck.tier,
            "kb_card_id": None if ck is None else ck.card_id,
            "in_exact_catalogue_after_refresh": name in full,
            "core_frequency": r.get("core_frequency"),
        })

    counts = Counter(r["classification"] for r in rows)
    out = {
        "methodology": "2n_a_v1",
        "source": "HearthstoneJSON refresh-cards + Phase 2M problematic slots",
        "kb_size_after_refresh": len(kb),
        "classification_counts": dict(counts),
        "rows": rows,
        "summary": {
            "active_missing_fixed_by_kb_refresh": counts.get(
                "ACTIVE_MISSING_FROM_KB_FIXED_BY_REFRESH", 0),
            "tier_out_of_sim_scope": counts.get("TIER_OUT_OF_SIM_SCOPE", 0),
            "tier_refresh_fixed": counts.get("TIER_REFRESH_FIXED", 0),
        },
        "note": (
            "Do not invent cards. Active missing cores were repaired by "
            "refreshing bg_cards.json from HearthstoneJSON. T7 cores are "
            "out of Simulator v1.x MAX_TIER=6 scope — remove from archetype "
            "references rather than adding Tier-7 shop support."),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["classification_counts"], indent=2))
    print(json.dumps(out["summary"], indent=2))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
