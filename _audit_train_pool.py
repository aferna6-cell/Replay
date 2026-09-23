import json
from pathlib import Path
from collections import defaultdict, Counter

root = Path("data/hsreplay/36.6.1")
meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
mins = json.loads((root / "minions.json").read_text(encoding="utf-8"))["minions"]
comps = json.loads((root / "comps.json").read_text(encoding="utf-8"))["comps"]
tribes_blob = json.loads((root / "tribes.json").read_text(encoding="utf-8"))
tlist = tribes_blob["tribes"] if isinstance(tribes_blob, dict) else tribes_blob

in_pool = {m["id"]: m for m in mins if m.get("in_pool", True)}
live_tribes = [str(t).lower() for t in (meta.get("live_tribes") or [])]
if not live_tribes:
    live_tribes = [str(x.get("id") or x.get("name") or "").lower() for x in tlist]
print("live_tribes", live_tribes)
print("excluded", meta.get("excluded_tribes"))
print("in_pool_minions", len(in_pool))

def piece_ids(c):
    ids = []
    for k in (
        "core_piece_ids", "key_piece_ids", "enabler_ids", "addon_piece_ids",
        "core_ids", "key_ids",
    ):
        for pid in c.get(k) or []:
            if pid and pid not in ids:
                ids.append(pid)
    return ids

def missing_ids(c):
    return [pid for pid in piece_ids(c) if pid not in in_pool]

print("\n=== COMP AUDIT ===")
kept, dropped = [], []
for c in comps:
    miss = missing_ids(c)
    tribe = str(c.get("tribe") or "").lower()
    ok = (
        c.get("in_pool") is not False
        and not miss
        and tribe != "naga"
        and str(c.get("train_weight")) not in ("0", "0.0")
    )
    row = {
        "id": c.get("id"),
        "name": c.get("name"),
        "tribe": tribe,
        "rank": c.get("rank"),
        "train_weight": c.get("train_weight"),
        "core": list(c.get("core_piece_ids") or []),
        "key": list(c.get("key_piece_ids") or []),
        "enablers": list(c.get("enabler_ids") or []),
        "addons": list(c.get("addon_piece_ids") or []),
        "missing": miss,
        "ok": ok,
        "source": "snapshot",
    }
    (kept if ok else dropped).append(row)
    flag = "KEEP" if ok else "DROP"
    print(f"{flag} {row['id']} tribe={tribe} rank={row['rank']} tw={row['train_weight']} missing={miss} name={row['name']}")

print("kept", len(kept), "dropped", len(dropped))
print("tribes_with_kept", sorted({r["tribe"] for r in kept}))
need = [t for t in live_tribes if t not in {r["tribe"] for r in kept} and t not in ("naga", "")]
print("tribes_needing_synth", need)

# tribe buckets for minions
by_tribe = defaultdict(list)
for m in in_pool.values():
    ts = m.get("tribes") or []
    if isinstance(ts, str):
        ts = [ts]
    if m.get("tribe"):
        ts = list(ts) + [m.get("tribe")]
    for t in ts:
        by_tribe[str(t).lower()].append(m)
print("minion_tribe_counts", {k: len({x["id"] for x in v}) for k, v in sorted(by_tribe.items())})

# thin150 audit
thin_path = Path("data/train_perfect_thin150/expert_perfect_thin150.jsonl")
rows = [json.loads(l) for l in thin_path.read_text(encoding="utf-8").splitlines() if l.strip()]
print("\n=== THIN150 AUDIT ===", "rows", len(rows))
clean, dirty = [], []
unk_counter = Counter()
for i, r in enumerate(rows):
    board = (r.get("state") or {}).get("board") or []
    bad = []
    names = []
    for m in board:
        cid = m.get("card_id") or m.get("cardId") or m.get("id")
        name = (m.get("name") or "").strip()
        names.append(f"{name}({cid})")
        if not cid or cid not in in_pool:
            bad.append(f"{name}|{cid}")
            unk_counter[f"{cid}|{name}"] += 1
    entry = {
        "idx": i,
        "composition_id": r.get("composition_id"),
        "placement": r.get("placement"),
        "source": r.get("source"),
        "patch_note": r.get("patch_note"),
        "board": names,
        "bad": bad,
        "ok": not bad and len(board) >= 2,
    }
    (clean if entry["ok"] else dirty).append(entry)
print("clean", len(clean), "dirty", len(dirty))
print("top_unknown", unk_counter.most_common(20))

out = {
    "meta": {
        "patch": meta.get("patch"),
        "live_tribes": live_tribes,
        "excluded_tribes": meta.get("excluded_tribes"),
        "in_pool_minions": len(in_pool),
    },
    "comps_kept": kept,
    "comps_dropped": dropped,
    "tribes_needing_synth": need,
    "thin150_clean": clean,
    "thin150_dirty_count": len(dirty),
    "thin150_unknown_top": unk_counter.most_common(40),
    "minion_tribe_counts": {k: len({x["id"] for x in v}) for k, v in sorted(by_tribe.items())},
}
Path("data/train_review_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("wrote data/train_review_audit.json")
