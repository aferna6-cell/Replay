"""Build eval-net population examples from the local HSReplay snapshot DB.

No Firestone fetch. No VOD corpus. Uses comps.json + minions.json to synthesize
boards from core/key/addon piece ids (and example_boards when present).
Perfect boards (thin150) are folded in separately via trajectory_examples.

Support channel (lightweight, does not redesign features):
  * heroes.json  — clone related active-comp boards with hero id + avg_placement
  * trinkets.json — same, with trinket id in state.trinkets
  * spells.json  — in-pool spells with coach grades mapped to placement labels,
                   paired with a small board from related active comps or
                   tribe-tagged in-pool minions
Comps stay highest weight (4x); support examples are 2x each and only emit when
they can attach to an active train comp (or a live-tribe minion sample).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from hsbg_coach import cards
from .board_features import minion_from_snapshot

UNKNOWN_HERO = "UNKNOWN"

_RANK_PLACEMENT = {
    "S": 1.8,
    "A": 2.5,
    "B": 3.5,
    "C": 4.5,
    "D": 5.5,
    "F": 6.5,
}

_WEIGHT_COPIES = {
    "highest": 4,
    "high": 3,
    "medium": 2,
    "support": 2,
    "low": 1,
}

_ENDGAME_CONTEXT = {
    "tavern_tier": 6,
    "gold": 0,
    "hero_health": 25,
    "turn": 13,
    "opponent_profiles": [],
    "trinkets": [],
    "anomaly": None,
}

_COACH_GRADE_PLACEMENT = {
    "S": 1.8,
    "A": 2.5,
    "B": 3.5,
    "C": 4.5,
    "D": 5.5,
    "F": 6.5,
}


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _minion_index(snapshot_dir: Path) -> Dict[str, dict]:
    data = _load_json(snapshot_dir / "minions.json")
    mins = data["minions"] if isinstance(data, dict) else data
    return {m["id"]: m for m in mins if m.get("id")}


def _resolve_piece(piece_id: str, mins: Dict[str, dict], kb: dict):
    m = mins.get(piece_id)
    if m and m.get("name"):
        atk = int(m.get("attack") or 1)
        hp = int(m.get("health") or 1)
        tier = int(m.get("tavern_tier") or m.get("tier") or 1)
        return {
            "name": m["name"],
            "card_id": piece_id,
            "attack": atk * 2,
            "health": hp * 2,
            "tags": {
                "ATK": atk * 2,
                "HEALTH": hp * 2,
                "TECH_LEVEL": tier,
                "PREMIUM": 1,
            },
        }
    ck = kb.get(piece_id)
    if ck is None:
        return None
    atk = int(getattr(ck, "attack", 1) or 1)
    hp = int(getattr(ck, "health", 1) or 1)
    tier = int(getattr(ck, "tier", 1) or 1)
    return {
        "name": ck.name,
        "card_id": piece_id,
        "attack": atk * 2,
        "health": hp * 2,
        "tags": {
            "ATK": atk * 2,
            "HEALTH": hp * 2,
            "TECH_LEVEL": tier,
            "PREMIUM": 1,
        },
    }


def _label_for_comp(comp: dict) -> Optional[float]:
    for key in ("avg_placement", "average_placement", "avgPlacement", "label_placement"):
        if comp.get(key) is not None:
            try:
                return float(comp[key])
            except (TypeError, ValueError):
                pass
    rank = str(comp.get("rank") or "").strip().upper()
    if rank in _RANK_PLACEMENT:
        return float(_RANK_PLACEMENT[rank])
    tr = comp.get("tier_rank")
    if tr is not None:
        try:
            return min(7.5, 1.5 + 0.35 * float(tr))
        except (TypeError, ValueError):
            pass
    return None


def _copies_for(comp: dict, meta: dict) -> int:
    tw = comp.get("train_weight")
    if tw is None:
        tw = (meta.get("train_weights") or {}).get("comps", "highest")
    if tw in (0, "0", 0.0):
        return 0
    if isinstance(tw, (int, float)) and float(tw) <= 0:
        return 0
    key = str(tw).strip().lower()
    return int(_WEIGHT_COPIES.get(key, 2))


def _board_from_comp(comp: dict, mins: Dict[str, dict], kb: dict, byname: dict):
    for key in ("example_boards", "boards", "sample_boards"):
        boards = comp.get(key) or []
        if boards:
            raw_board = boards[0]
            if isinstance(raw_board, dict):
                # either {"board": [...]} or a single minion-like dict wrongly nested
                if "board" in raw_board or "minions" in raw_board:
                    raw_board = raw_board.get("board") or raw_board.get("minions") or []
                elif "card_id" in raw_board or "name" in raw_board:
                    # list was actually the board already but boards[0] is first minion —
                    # fall through only if boards looks like a minion list
                    if all(isinstance(x, dict) and (x.get("card_id") or x.get("name")) for x in boards):
                        raw_board = boards
            minions = [
                m for m in (minion_from_snapshot(x, byname) for x in raw_board) if m
            ]
            if len(minions) >= 2:
                return minions

    # singular example_board (synth v2)
    eb = comp.get("example_board")
    if eb:
        raw_board = eb
        if isinstance(raw_board, dict):
            raw_board = raw_board.get("board") or raw_board.get("minions") or []
        minions = [
            m for m in (minion_from_snapshot(x, byname) for x in raw_board) if m
        ]
        if len(minions) >= 2:
            return minions

    ids = []
    for key in (
        "core_piece_ids",
        "key_piece_ids",
        "addon_piece_ids",
        "enabler_ids",
        "enabler_piece_ids",
        "core_ids",
        "key_ids",
    ):
        for pid in comp.get(key) or []:
            if pid and pid not in ids:
                ids.append(pid)
    raws = []
    for pid in ids:
        raw = _resolve_piece(pid, mins, kb)
        if raw:
            raws.append(raw)
        if len(raws) >= 7:
            break
    return [m for m in (minion_from_snapshot(x, byname) for x in raws) if m]


def _board_from_tribe(tribe: str, mins: Dict[str, dict], kb: dict, byname: dict, n: int = 4):
    tribe_l = str(tribe or "").lower()
    picked = []
    for mid, m in mins.items():
        tribes = [str(t).lower() for t in (m.get("tribes") or [])]
        if tribe_l and tribe_l not in tribes:
            continue
        tier = int(m.get("tavern_tier") or m.get("tier") or 99)
        if tier > 6:
            continue
        raw = _resolve_piece(mid, mins, kb)
        if raw:
            picked.append(raw)
        if len(picked) >= n:
            break
    return [m for m in (minion_from_snapshot(x, byname) for x in picked) if m]


def _active_comp_boards(comps, mins, kb, byname, meta, excluded):
    """Return {comp_id: (minions, label, tribe)} for active train comps."""
    out = {}
    for comp in comps:
        tribe = str(comp.get("tribe") or "").lower()
        if tribe in excluded or tribe == "naga":
            continue
        if comp.get("in_pool") is False:
            continue
        if _copies_for(comp, meta) <= 0:
            continue
        label = _label_for_comp(comp)
        if label is None:
            continue
        minions = _board_from_comp(comp, mins, kb, byname)
        if len(minions) < 2:
            continue
        cid = comp.get("id") or comp.get("slug") or comp.get("name")
        out[cid] = (minions, float(label), tribe)
        # also index numeric hsreplay ids when present
        hid = comp.get("hsreplay_comp_id")
        if hid is not None:
            out[f"comp_{hid}"] = out[cid]
            out[hid] = out[cid]
    return out


def _normalize_related_ids(raw):
    ids = []
    for x in raw or []:
        if x is None:
            continue
        if isinstance(x, int):
            ids.append(f"comp_{x}")
            ids.append(x)
        else:
            s = str(x)
            ids.append(s)
            if s.isdigit():
                ids.append(f"comp_{s}")
                ids.append(int(s))
    return ids


def _support_from_heroes(root, active_boards, mins, kb, byname, meta):
    path = root / "heroes.json"
    if not path.exists():
        return [], {"missing_file": 1}
    blob = _load_json(path)
    heroes = blob["heroes"] if isinstance(blob, dict) else blob
    out = []
    skipped = {"no_label": 0, "no_board": 0, "no_link": 0}
    tw = (meta.get("train_weights") or {}).get("heroes") or (
        meta.get("train_weights") or {}
    ).get("heroes_trinkets", "support")
    if tw in (0, "0", 0.0) or (isinstance(tw, (int, float)) and float(tw) <= 0):
        return [], {"zero_weight": len(heroes)}

    for h in heroes:
        label = h.get("avg_placement")
        if label is None:
            skipped["no_label"] += 1
            continue
        board = None
        for rid in _normalize_related_ids(h.get("related_comp_ids")):
            if rid in active_boards:
                board = active_boards[rid][0]
                break
        if board is None:
            fav = h.get("favorable_tribes") or []
            for t in fav:
                board = _board_from_tribe(t, mins, kb, byname)
                if len(board) >= 2:
                    break
        if board is None or len(board) < 2:
            skipped["no_board" if board is not None else "no_link"] += 1
            continue
        hid = h.get("id") or UNKNOWN_HERO
        out.append({
            "minions": board,
            "hero": hid,
            "label": float(label),
            "state": dict(_ENDGAME_CONTEXT),
            "group": f"hero_support:{hid}",
            "source": "heroes_support",
        })
    return out, skipped


def _support_from_trinkets(root, active_boards, mins, kb, byname, meta):
    path = root / "trinkets.json"
    if not path.exists():
        return [], {"missing_file": 1}
    blob = _load_json(path)
    trinkets = blob["trinkets"] if isinstance(blob, dict) else blob
    out = []
    skipped = {"no_label": 0, "no_board": 0, "no_link": 0}
    tw = (meta.get("train_weights") or {}).get("trinkets") or (
        meta.get("train_weights") or {}
    ).get("heroes_trinkets", "support")
    if tw in (0, "0", 0.0) or (isinstance(tw, (int, float)) and float(tw) <= 0):
        return [], {"zero_weight": len(trinkets)}

    for t in trinkets:
        label = t.get("avg_placement")
        if label is None:
            skipped["no_label"] += 1
            continue
        board = None
        for rid in _normalize_related_ids(t.get("related_comp_ids")):
            if rid in active_boards:
                board = active_boards[rid][0]
                break
        if board is None:
            tribe = t.get("tribe") or (t.get("affinity") if t.get("affinity") not in (None, "neutral") else None)
            if tribe:
                board = _board_from_tribe(tribe, mins, kb, byname)
        if board is None or len(board) < 2:
            skipped["no_board" if board is not None else "no_link"] += 1
            continue
        tid = t.get("id") or t.get("name") or "trinket"
        state = dict(_ENDGAME_CONTEXT)
        state["trinkets"] = [tid]
        out.append({
            "minions": board,
            "hero": UNKNOWN_HERO,
            "label": float(label),
            "state": state,
            "group": f"trinket_support:{tid}",
            "source": "trinkets_support",
        })
    return out, skipped


def _support_from_spells(root, active_boards, mins, kb, byname, meta):
    path = root / "spells.json"
    if not path.exists():
        return [], {"missing_file": 1}
    blob = _load_json(path)
    spells = blob["spells"] if isinstance(blob, dict) else blob
    out = []
    skipped = {"out_of_pool": 0, "no_grade": 0, "no_board": 0}
    tw = (meta.get("train_weights") or {}).get("spells", "support")
    if tw in (0, "0", 0.0) or (isinstance(tw, (int, float)) and float(tw) <= 0):
        return [], {"zero_weight": len(spells)}

    # Prefer attaching to an active spell-synergy-ish comp (mech / demon / dragon)
    fallback_tribe_order = ["mech", "demon", "dragon", "quilboar", "aberration", "elemental"]
    fallback_board = None
    for tribe in fallback_tribe_order:
        for _cid, (board, _lab, t) in active_boards.items():
            if t == tribe:
                fallback_board = board
                break
        if fallback_board:
            break
    if fallback_board is None and active_boards:
        fallback_board = next(iter(active_boards.values()))[0]

    for s in spells:
        if s.get("in_pool") is False:
            skipped["out_of_pool"] += 1
            continue
        grade = str(s.get("coach_grade") or "").strip().upper()
        if grade not in _COACH_GRADE_PLACEMENT:
            skipped["no_grade"] += 1
            continue
        label = float(_COACH_GRADE_PLACEMENT[grade])
        board = None
        for rid in _normalize_related_ids(s.get("related_comp_ids")):
            if rid in active_boards:
                board = active_boards[rid][0]
                break
        if board is None:
            tags = [str(x).lower() for x in (s.get("synergy_tags") or [])]
            for tag in tags:
                # crude tag→tribe hints
                for tribe in fallback_tribe_order:
                    if tribe in tag or tag in tribe:
                        board = _board_from_tribe(tribe, mins, kb, byname)
                        if board:
                            break
                if board:
                    break
        if board is None:
            board = fallback_board
        if board is None or len(board) < 2:
            skipped["no_board"] += 1
            continue
        sid = s.get("id") or s.get("name") or "spell"
        out.append({
            "minions": board,
            "hero": UNKNOWN_HERO,
            "label": label,
            "state": dict(_ENDGAME_CONTEXT),
            "group": f"spell_support:{sid}",
            "source": "spells_support",
        })
    return out, skipped


def build_hsreplay_snapshot_examples(snapshot_dir: str) -> List[Dict]:
    """Population prior from the local HSReplay JSON snapshot (no Firestone)."""
    root = Path(snapshot_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"hsreplay snapshot not found: {root}")

    meta = _load_json(root / "meta.json") if (root / "meta.json").exists() else {}
    comps_blob = _load_json(root / "comps.json")
    comps = comps_blob["comps"] if isinstance(comps_blob, dict) else comps_blob
    mins = _minion_index(root)
    kb = cards.load_kb()
    byname = cards.by_name(kb)

    excluded = {str(t).lower() for t in (meta.get("excluded_tribes") or ["naga"])}
    out: List[Dict] = []
    skipped = {
        "naga": 0,
        "out_of_pool": 0,
        "zero_weight": 0,
        "no_board": 0,
        "no_label": 0,
    }

    for comp in comps:
        tribe = str(comp.get("tribe") or "").lower()
        if tribe in excluded or tribe == "naga":
            skipped["naga"] += 1
            continue
        if comp.get("in_pool") is False:
            skipped["out_of_pool"] += 1
            continue
        copies = _copies_for(comp, meta)
        if copies <= 0:
            skipped["zero_weight"] += 1
            continue
        label = _label_for_comp(comp)
        if label is None:
            skipped["no_label"] += 1
            continue
        minions = _board_from_comp(comp, mins, kb, byname)
        if len(minions) < 2:
            skipped["no_board"] += 1
            continue
        group = comp.get("id") or comp.get("slug") or comp.get("name") or "comp"
        for i in range(copies):
            out.append({
                "minions": minions,
                "hero": UNKNOWN_HERO,
                "label": float(label),
                "state": dict(_ENDGAME_CONTEXT),
                "group": f"{group}#{i}" if copies > 1 else group,
                "source": "comps",
            })

    active_boards = _active_comp_boards(comps, mins, kb, byname, meta, excluded)

    hero_ex, hero_skip = _support_from_heroes(root, active_boards, mins, kb, byname, meta)
    trinket_ex, trinket_skip = _support_from_trinkets(root, active_boards, mins, kb, byname, meta)
    spell_ex, spell_skip = _support_from_spells(root, active_boards, mins, kb, byname, meta)
    out.extend(hero_ex)
    out.extend(trinket_ex)
    out.extend(spell_ex)

    print(
        f"  hsreplay snapshot: {len(out)} boards from {root} "
        f"(comps_skipped {skipped}; "
        f"support heroes={len(hero_ex)} skip={hero_skip}; "
        f"trinkets={len(trinket_ex)} skip={trinket_skip}; "
        f"spells={len(spell_ex)} skip={spell_skip})"
    )
    return out
