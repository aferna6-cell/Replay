import json
from pathlib import Path
from collections import defaultdict

root = Path("data/hsreplay/36.6.1")
meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
mins = json.loads((root / "minions.json").read_text(encoding="utf-8"))["minions"]
comps = json.loads((root / "comps.json").read_text(encoding="utf-8"))["comps"]
tribes_blob = json.loads((root / "tribes.json").read_text(encoding="utf-8"))
spells = json.loads((root / "spells.json").read_text(encoding="utf-8"))
slist = spells["spells"] if isinstance(spells, dict) else spells
in_pool = {m["id"]: m for m in mins if m.get("in_pool", True)}

by_tribe = defaultdict(list)
for m in in_pool.values():
    ts = list(m.get("tribes") or [])
    if isinstance(m.get("tribes"), str):
        ts = [m["tribes"]]
    if m.get("tribe"):
        ts.append(m["tribe"])
    for t in ts:
        by_tribe[str(t).lower()].append(m)

def uniq(ms):
    seen=set(); out=[]
    for m in ms:
        if m["id"] not in seen:
            seen.add(m["id"]); out.append(m)
    return out

def score_minion(m):
    # prefer higher tier + having synergy tags / related comps
    tier = int(m.get("tavern_tier") or m.get("tier") or 1)
    syn = len(m.get("synergy_tags") or [])
    comps_n = len(m.get("comp_ids") or m.get("related_comp_ids") or [])
    fp = float(m.get("first_place_rate") or 0) or 0
    return (tier, syn, comps_n, fp)

def pick_board(tribe, n_core=4, n_total=7):
    pool = uniq(by_tribe.get(tribe, []))
    pool = sorted(pool, key=score_minion, reverse=True)
    # prefer T4-T6 as cores, fill with T3+
    cores = [m for m in pool if int(m.get("tavern_tier") or m.get("tier") or 1) >= 4][:n_core]
    if len(cores) < n_core:
        cores = pool[:n_core]
    rest = [m for m in pool if m["id"] not in {c["id"] for c in cores}]
    board = cores + rest[: max(0, n_total - len(cores))]
    return board[:n_total]

def board_rec(m):
    atk = int(m.get("attack") or 1)
    hp = int(m.get("health") or 1)
    tier = int(m.get("tavern_tier") or m.get("tier") or 1)
    # golden-ish endgame scale for synth finals
    return {
        "name": m["name"],
        "card_id": m["id"],
        "attack": atk * 2,
        "health": hp * 2,
        "tier": tier,
        "tribes": m.get("tribes") or ([m["tribe"]] if m.get("tribe") else []),
        "tags": {"ATK": atk * 2, "HEALTH": hp * 2, "TECH_LEVEL": tier, "PREMIUM": 1},
    }

# load prior audit
audit = json.loads(Path("data/train_review_audit.json").read_text(encoding="utf-8"))
kept = audit["comps_kept"]
need = audit["tribes_needing_synth"]

# Build synth comps for missing tribes using top in-pool minions
synth_comps = []
for tribe in need:
    board_ms = pick_board(tribe)
    if len(board_ms) < 4:
        print("WARN not enough minions for", tribe, len(board_ms))
        continue
    core = board_ms[:3]
    key = board_ms[:5]
    addons = board_ms[5:7]
    # placement prior by tribe strength proxy: mid A
    rec = {
        "id": f"synth_{tribe}_1",
        "name": f"{tribe.title()} - Snapshot Synth A",
        "tribe": tribe,
        "rank": "A",
        "train_weight": "highest",
        "source": "synth_from_snapshot_pool",
        "core_piece_ids": [m["id"] for m in core],
        "key_piece_ids": [m["id"] for m in key],
        "addon_piece_ids": [m["id"] for m in addons],
        "core_names": [m["name"] for m in core],
        "key_names": [m["name"] for m in key],
        "notes": "Synthesized from current in-pool minions because no kept snapshot comp had all key pieces intact.",
        "example_board": [board_rec(m) for m in board_ms],
        "label_placement": 2.5,
    }
    synth_comps.append(rec)

# Boards for kept comps (only intact piece ids)
def resolve(pid):
    return in_pool.get(pid)

kept_boards = []
for c in kept:
    ids = []
    for lst in (c.get("core") or [], c.get("key") or [], c.get("enablers") or [], c.get("addons") or []):
        for pid in lst:
            if pid and pid not in ids:
                ids.append(pid)
    ms = [resolve(pid) for pid in ids]
    ms = [m for m in ms if m]
    # pad to 6-7 from tribe pool if short
    if len(ms) < 6:
        tribe = c["tribe"]
        for m in pick_board(tribe, n_core=2, n_total=12):
            if m["id"] not in {x["id"] for x in ms}:
                ms.append(m)
            if len(ms) >= 7:
                break
    rank = str(c.get("rank") or "A").upper()
    label = {"S": 1.8, "A": 2.5, "B": 3.5, "C": 4.5}.get(rank, 2.8)
    kept_boards.append({
        "id": c["id"],
        "name": c["name"],
        "tribe": c["tribe"],
        "rank": c.get("rank"),
        "source": "snapshot_comp_intact",
        "core_names": [in_pool[p]["name"] for p in (c.get("core") or []) if p in in_pool],
        "key_names": [in_pool[p]["name"] for p in (c.get("key") or []) if p in in_pool],
        "example_board": [board_rec(m) for m in ms[:7]],
        "label_placement": label,
    })

# Clean thin150 export for review
thin_path = Path("data/train_perfect_thin150/expert_perfect_thin150.jsonl")
rows = [json.loads(l) for l in thin_path.read_text(encoding="utf-8").splitlines() if l.strip()]
clean_thin = []
for i, r in enumerate(rows):
    board = (r.get("state") or {}).get("board") or []
    bad = []
    shown = []
    for m in board:
        cid = m.get("card_id") or m.get("cardId") or m.get("id")
        name = (m.get("name") or "").strip()
        atk = m.get("attack"); hp = m.get("health")
        shown.append(f"{name} [{cid}] {atk}/{hp}")
        if not cid or cid not in in_pool:
            bad.append(f"{name}|{cid}")
    if not bad and len(board) >= 2:
        clean_thin.append({
            "idx": i,
            "composition_id": r.get("composition_id"),
            "placement": r.get("placement"),
            "source": r.get("source"),
            "patch_note": r.get("patch_note"),
            "board": shown,
        })

review = {
    "patch": meta.get("patch"),
    "rules": {
        "comps": "keep only if in_pool and EVERY core/key/enabler/addon id is in current minions in_pool; naga excluded",
        "thin150": "keep only if every board card_id is in current minions in_pool",
        "synth": "one A-rank synth comp per live tribe that had zero kept comps, pieces from current in_pool only",
        "train": "NOT run yet — awaiting approval",
    },
    "summary": {
        "kept_snapshot_comps": len(kept_boards),
        "dropped_snapshot_comps": len(audit["comps_dropped"]),
        "synth_comps": len(synth_comps),
        "thin150_clean": len(clean_thin),
        "thin150_dirty_excluded": audit["thin150_dirty_count"],
        "tribes_covered": sorted({b["tribe"] for b in kept_boards} | {s["tribe"] for s in synth_comps}),
    },
    "dropped_comps": [
        {"id": d["id"], "name": d["name"], "tribe": d["tribe"], "missing": d["missing"]}
        for d in audit["comps_dropped"]
    ],
    "kept_comps_with_boards": kept_boards,
    "synth_comps_with_boards": synth_comps,
    "thin150_clean_boards": clean_thin,
}

out = Path("data/train_review_proposed.json")
out.write_text(json.dumps(review, indent=2), encoding="utf-8")

# also write a readable markdown for Aidan
lines = []
lines.append(f"# Train review — patch {meta.get('patch')} (NOT trained yet)\n")
lines.append("## Rules")
lines.append("- Comps: all key/core/enabler/addon pieces must be in current in-pool minions")
lines.append("- Perfect boards: every card_id must be in current in-pool minions")
lines.append("- Synth: at least one comp per live tribe missing a kept snapshot comp")
lines.append("")
lines.append("## Summary")
for k,v in review["summary"].items():
    lines.append(f"- **{k}**: {v}")
lines.append("")
lines.append("## Dropped snapshot comps (missing pieces)")
for d in review["dropped_comps"]:
    lines.append(f"- `{d['id']}` **{d['name']}** ({d['tribe']}): missing `{', '.join(d['missing'])}`")
lines.append("")
lines.append("## Kept snapshot comps + synthesized boards")
for b in kept_boards:
    lines.append(f"### {b['id']} — {b['name']} ({b['tribe']}, rank {b['rank']})")
    lines.append(f"- cores: {', '.join(b['core_names']) or '(none)'}")
    lines.append(f"- keys: {', '.join(b['key_names']) or '(none)'}")
    lines.append("- board:")
    for m in b["example_board"]:
        lines.append(f"  - {m['name']} [{m['card_id']}] {m['attack']}/{m['health']} T{m['tier']}")
    lines.append("")
lines.append("## Synthesized comps (new)")
for b in synth_comps:
    lines.append(f"### {b['id']} — {b['name']}")
    lines.append(f"- cores: {', '.join(b['core_names'])}")
    lines.append(f"- keys: {', '.join(b['key_names'])}")
    lines.append(f"- note: {b['notes']}")
    lines.append("- board:")
    for m in b["example_board"]:
        lines.append(f"  - {m['name']} [{m['card_id']}] {m['attack']}/{m['health']} T{m['tier']}")
    lines.append("")
lines.append(f"## Perfect thin150 boards kept ({len(clean_thin)} / 150)")
for b in clean_thin:
    lines.append(f"### thin#{b['idx']} comp={b['composition_id']} placement={b['placement']}")
    for s in b["board"]:
        lines.append(f"  - {s}")
    lines.append("")

md = Path("data/TRAIN_REVIEW_PROPOSED.md")
md.write_text("\n".join(lines), encoding="utf-8")
print("wrote", out, "and", md)
print("summary", json.dumps(review["summary"], indent=2))
print("synth ids", [s["id"] for s in synth_comps])
