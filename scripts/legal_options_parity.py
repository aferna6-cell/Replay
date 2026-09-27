"""Re-measure encode.legal_options against the server's option lists.

Given label rows (labels.v1 jsonl / jsonl.gz, see docs/labels_schema.md on the
Labeler branch), compare the options legal_options(snapshot) emits with each
row's own `options` list (the server ground truth) and report:

  * exact option-set match %, and the same ignoring targets
  * chosen-option producible %, and the same ignoring targets
  * extra (encoder-only) and missing (server-only) option counts by type, with
    `play` / `hero_power` split into sub-kinds (hand / activation / dark_gift /
    choose_one_variant / targeted ...)
  * unproducible chosen options by type and sub-kind
  * the same extra / missing counts with targets collapsed, and the number of
    rows with at least one extra / missing option of each type

Options are compared on encode.option_key (type, card_id, source, target,
position, choice_kind); debug fields (src_entity / target_entity) are ignored,
so a hero-targeted option (target zone none) matches an untargeted one.

  python scripts/legal_options_parity.py --data "data/firestone/labels/v1/*.jsonl.gz" \
      --out results/pilot/parity_after.json
"""

import argparse
import collections
import glob
import gzip
import json
import os
import sys
from typing import Dict, Iterable, List

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from hsbg_coach import encode as enc  # noqa: E402


def read_rows(patterns: Iterable[str]) -> List[Dict]:
    rows = []
    for pat in patterns:
        for path in (sorted(glob.glob(pat)) or [pat]):
            opener = gzip.open if path.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as fh:
                rows += [json.loads(line) for line in fh if line.strip()]
    return rows


def row_snapshot(row: Dict):
    """labels.v1 rows carry the snapshot as `state`; older rows as `snapshot`."""
    snap = row.get("snapshot")
    return snap if isinstance(snap, dict) else row.get("state")


def _untargeted(key: tuple) -> tuple:
    return key[:4] + ("none", None) + key[6:]


def sub_kind(snapshot, option: Dict) -> str:
    """Finer label for play / hero_power options (the gaps differ by kind)."""
    t = option.get("type")
    src = option.get("source") or {}
    tgt = option.get("target") or {}
    targeted = (tgt.get("zone") or "none") != "none" or option.get("target_entity") is not None
    if t == "hero_power":
        return "hero_power:" + ("targeted" if targeted else "untargeted")
    if t != "play":
        return t
    zone = src.get("zone") or "none"
    if zone == "none":
        return "play:dark_gift"
    if zone == "board":
        return "play:activation" + (":targeted" if targeted else "")
    card = enc.option_card(snapshot, option) or {}
    if card.get("card_id") and option.get("card_id") != card.get("card_id"):
        return "play:choose_one_variant"
    tags = card.get("tags") or {}
    if str(tags.get("CHOOSE_ONE")) == "1":
        return "play:choose_one_parent"
    kind = "spell" if enc.is_spell(card) else "minion"
    return f"play:{kind}" + (":targeted" if targeted else "")


def parity(rows: List[Dict], legal=enc.legal_options, n_examples: int = 3) -> Dict:
    c = collections.Counter()
    extra, missing = collections.Counter(), collections.Counter()
    extra_kind, missing_kind = collections.Counter(), collections.Counter()
    chosen_by, unprod, unprod_kind = collections.Counter(), collections.Counter(), collections.Counter()
    miss_u, extra_u = collections.Counter(), collections.Counter()
    rows_miss, rows_extra = collections.Counter(), collections.Counter()
    examples = collections.defaultdict(list)
    for row in rows:
        snap, opts, ci = row_snapshot(row), row.get("options") or [], row.get("chosen")
        if not isinstance(snap, dict) or not isinstance(ci, int) or not 0 <= ci < len(opts):
            c["rows_skipped_invalid"] += 1
            continue
        c["rows"] += 1
        ours = legal(snap)
        ours_k = {enc.option_key(o): o for o in ours}
        srv_k = {enc.option_key(o): o for o in opts}
        ours_u = {_untargeted(k) for k in ours_k}
        srv_u = {_untargeted(k) for k in srv_k}
        c["exact_match"] += set(ours_k) == set(srv_k)
        c["match_ignoring_targets"] += ours_u == srv_u
        for keys, other, out in ((ours_k, srv_u, extra_u), (srv_k, ours_u, miss_u)):
            seen = set()
            for k, o in keys.items():
                u = _untargeted(k)
                if u not in other and u not in seen:
                    seen.add(u)
                    out[sub_kind(snap, dict(o, target={"zone": "none", "slot": None},
                                            target_entity=None))] += 1
        for keys, other, out in ((ours_k, srv_k, rows_extra), (srv_k, ours_k, rows_miss)):
            for t in {o.get("type", "?") for k, o in keys.items() if k not in other}:
                out[t] += 1
        chosen = opts[ci]
        ck = enc.option_key(chosen)
        ctype = chosen.get("type", "?")
        chosen_by[ctype] += 1
        c["chosen_producible"] += ck in ours_k
        c["chosen_producible_ignoring_targets"] += _untargeted(ck) in ours_u
        if ck not in ours_k:
            unprod[ctype] += 1
            kind = sub_kind(snap, chosen)
            unprod_kind[kind] += 1
            if len(examples["unproducible:" + kind]) < n_examples:
                examples["unproducible:" + kind].append(
                    {"game_id": row.get("game_id"), "dp_index": row.get("dp_index"),
                     "gold": snap.get("gold"), "option": {k: chosen.get(k) for k in enc.OPTION_KEYS}})
        for k, o in ours_k.items():
            if k not in srv_k:
                extra[o["type"]] += 1
                kind = sub_kind(snap, o)
                extra_kind[kind] += 1
                if len(examples["extra:" + kind]) < n_examples:
                    examples["extra:" + kind].append(
                        {"game_id": row.get("game_id"), "dp_index": row.get("dp_index"),
                         "gold": snap.get("gold"), "option": o})
        for k, o in srv_k.items():
            if k not in ours_k:
                missing[o.get("type", "?")] += 1
                kind = sub_kind(snap, o)
                missing_kind[kind] += 1
                if len(examples["missing:" + kind]) < n_examples:
                    examples["missing:" + kind].append(
                        {"game_id": row.get("game_id"), "dp_index": row.get("dp_index"),
                         "gold": snap.get("gold"), "option": {x: o.get(x) for x in enc.OPTION_KEYS}})
    n = max(c["rows"], 1)

    def pct(x):
        return round(100.0 * x / n, 1)

    return {
        "encoder_version": enc.ENCODER_VERSION,
        "rows": c["rows"], "rows_skipped_invalid": c["rows_skipped_invalid"],
        "exact_match": c["exact_match"], "exact_match_pct": pct(c["exact_match"]),
        "match_ignoring_targets": c["match_ignoring_targets"],
        "match_ignoring_targets_pct": pct(c["match_ignoring_targets"]),
        "chosen_producible": c["chosen_producible"],
        "chosen_producible_pct": pct(c["chosen_producible"]),
        "chosen_producible_ignoring_targets": c["chosen_producible_ignoring_targets"],
        "chosen_producible_ignoring_targets_pct": pct(c["chosen_producible_ignoring_targets"]),
        "chosen_by_type": dict(sorted(chosen_by.items())),
        "unproducible_chosen_by_type": dict(sorted(unprod.items())),
        "unproducible_chosen_by_kind": dict(sorted(unprod_kind.items())),
        "extra_by_type": dict(sorted(extra.items())), "extra_total": sum(extra.values()),
        "extra_by_kind": dict(sorted(extra_kind.items())),
        "missing_by_type": dict(sorted(missing.items())), "missing_total": sum(missing.values()),
        "missing_by_kind": dict(sorted(missing_kind.items())),
        "extra_ignoring_targets_by_kind": dict(sorted(extra_u.items())),
        "missing_ignoring_targets_by_kind": dict(sorted(miss_u.items())),
        "rows_with_extra_by_type": dict(sorted(rows_extra.items())),
        "rows_with_missing_by_type": dict(sorted(rows_miss.items())),
        "examples": dict(sorted(examples.items())),
    }


def summary_lines(r: Dict) -> List[str]:
    return [
        f"encoder {r['encoder_version']}  rows {r['rows']} (skipped invalid {r['rows_skipped_invalid']})",
        f"exact option-set match   {r['exact_match']:>5} ({r['exact_match_pct']}%)   "
        f"ignoring targets {r['match_ignoring_targets']} ({r['match_ignoring_targets_pct']}%)",
        f"chosen producible        {r['chosen_producible']:>5} ({r['chosen_producible_pct']}%)   "
        f"ignoring targets {r['chosen_producible_ignoring_targets']} "
        f"({r['chosen_producible_ignoring_targets_pct']}%)",
        f"unproducible chosen by type {r['unproducible_chosen_by_type']}",
        f"unproducible chosen by kind {r['unproducible_chosen_by_kind']}",
        f"extra (encoder only) {r['extra_total']}: {r['extra_by_type']}",
        f"   by kind {r['extra_by_kind']}",
        f"missing (server only) {r['missing_total']}: {r['missing_by_type']}",
        f"   by kind {r['missing_by_kind']}",
        f"ignoring targets: extra {r['extra_ignoring_targets_by_kind']}",
        f"                  missing {r['missing_ignoring_targets_by_kind']}",
        f"rows with extra {r['rows_with_extra_by_type']}",
        f"rows with missing {r['rows_with_missing_by_type']}",
    ]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="legal_options vs server option lists")
    p.add_argument("--data", nargs="+", required=True, help="label jsonl(.gz) files or globs")
    p.add_argument("--out", default=None, help="write the full report (JSON) here")
    p.add_argument("--examples", type=int, default=3, help="examples kept per category")
    a = p.parse_args(argv)
    rows = read_rows(a.data)
    if not rows:
        print("error: no rows read", file=sys.stderr)
        return 2
    r = parity(rows, n_examples=a.examples)
    print("\n".join(summary_lines(r)))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=1)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
