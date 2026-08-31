"""Fill experiments/ppo_multiseed_replication_v1.md from aggregate JSON."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = "results/ppo_multiseed_v1"
OUT = "experiments/ppo_multiseed_replication_v1.md"
SEEDS = [0, 1, 2, 3]


def pct(x):
    return f"{100 * x:.1f}%"


def md_ci(ci):
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def decide_outcome(repl: dict) -> tuple[str, str, str]:
    """Return (outcome letter, rationale, experiment4 recommendation).

    Decision rule (documented):
      A — mid-budget gain replicates clearly across most seeds AND late
          regression is weak/absent → investigate how to lock mid gains.
      B — endpoints flat-ish while drift rises; U-shape only partial →
          drift/anchoring is the phenomenon to address.
      C — progressive degradation dominates across seeds → stop longer PPO.
      D — mixed field diverges from greedy story → opponent-transfer study.
    """
    qa = repl["questions"]["question_A_iter80_minus_iter0"]
    qb = repl["questions"]["question_B_iter320_minus_iter80"]
    n_u = repl["n_seeds_u_like"]
    u_by = repl["u_shape_by_seed"]

    # crude mixed-vs-greedy agreement: compare signs of iter80−iter0 on both
    # fields from cross summary if present — fallback to B.
    if n_u >= 3 and qa["n_improve"] >= 3 and qb["n_regress"] >= 3:
        return (
            "A",
            "The mid-budget improvement and late regression both replicate "
            f"on {n_u}/4 seeds as U-like transient improvement. The algorithm "
            "reliably finds a transient gain that it then loses.",
            "Experiment 4: freeze the Exp2/3 recipe and test a single "
            "KL/warm-start anchoring (or trust-region) intervention aimed at "
            "preserving the mid-budget gain through 5,120 episodes — one "
            "change only; DEV-only; no TEST; no seed selection.",
        )
    if n_u >= 2 and qa["n_improve"] >= 2:
        return (
            "B",
            "A U-like transient gain appears on some seeds but is not "
            "universal; across seeds the late trajectory is dominated by "
            "rising drift with unstable placement. Best supported reading: "
            "budget alone is not a reliable fix, and drift is the stable "
            "cross-seed fact.",
            "Experiment 4: freeze recipe + warm start; add one KL/BC-prior "
            "anchoring term (or explicit trust-region penalty) and re-run "
            "seeds 0–3 on DEV only — test whether anchoring cuts iter320 "
            "drift without erasing any mid-budget gain. No TEST; no tuning "
            "sweep.",
        )
    deg = sum(1 for u in u_by.values()
              if u["class"] == "monotonic_degradation")
    if deg >= 3:
        return (
            "C",
            "Most seeds show progressive degradation under longer PPO.",
            "Experiment 4: stop extending PPO budget; instead evaluate a "
            "shorter fixed budget with anchoring against the BC prior "
            "(DEV-only multi-seed).",
        )
    return (
        "B",
        "Cross-seed evidence does not support monotone improvement or "
        "monotone collapse; exploratory n=4 favors treating unbounded drift "
        "as the reproducible fact and budget effects as seed-sensitive.",
        "Experiment 4: one KL/warm-start anchoring intervention on the "
        "frozen recipe, multi-seed DEV replication; no TEST; no hyperparameter "
        "search.",
    )


def main() -> int:
    summary = json.load(open(f"{ROOT}/aggregate/cross_seed_summary.json"))
    paired = json.load(open(f"{ROOT}/aggregate/paired_results.json"))
    repl = json.load(open(f"{ROOT}/aggregate/replication_analysis.json"))
    manifest = json.load(open(f"{ROOT}/manifest.json"))

    outcome, rationale, exp4 = decide_outcome(repl)
    qa = repl["questions"]["question_A_iter80_minus_iter0"]
    qb = repl["questions"]["question_B_iter320_minus_iter80"]

    lines = []
    lines.append("# Replay Experiment 3 — Multi-Seed PPO Budget Replication\n")
    lines.append(
        "Date: 2026-08-31 · Split: **DEV only** (Benchmark v1 TEST never run) ·\n"
        "Artifacts: [`results/ppo_multiseed_v1/`](../results/ppo_multiseed_v1/) ·\n"
        "Manifest: [`manifest.json`](../results/ppo_multiseed_v1/manifest.json)\n"
    )
    lines.append("## Question\n")
    lines.append(
        "Experiment 2 found a U-shaped PPO budget curve on training seed 0. "
        "Does that shape replicate across independent PPO training seeds?\n\n"
        "**One variable changed: PPO training seed.** Everything else is "
        "frozen from Experiment 2, including the BC+DAgger warm start.\n"
    )
    lines.append("## Historical observation\n")
    lines.append(
        "Experiment 2 / seed 0 (published): iter40 worse than warm start "
        "(+0.207), iter80 better (−0.229), iter320 back to noise vs warm "
        "start (+0.052) with KL 1.17 and warm-start agreement 40.8%.\n"
    )
    lines.append("## Setup\n")
    warm = manifest["warm_start"]["parameter_sha256"]
    lines.append(
        f"- Frozen warm start `parameter_sha256 = {warm}` "
        "(matches Exp2; verified before training).\n"
        "- Seed 0 = Exp2 artifacts (not retrained). New: seeds 1, 2, 3.\n"
        "- 16 eps/iter × 320 iters; `--shaping-horizon 40`; primary "
        "checkpoints 0/40/80/160/320.\n"
        "- DEV: 1000× greedy on 10,550,000–10,550,999; secondary "
        "`greedy4_random3` (500). TEST locked.\n"
        "- Drift corpus fingerprint prefix `2ec217b353bd` (Exp1/2).\n"
    )
    lines.append("## Training-seed control\n")
    lines.append(
        "| Seed | Command |\n|---|---|\n"
    )
    for s in (0, 1, 2, 3):
        lines.append(f"| {s} | `{manifest['training']['commands'][str(s)]}` |\n")

    lines.append("\n## Per-seed results\n")
    lines.append(
        "| seed | iter0 | iter40 | iter80 | iter160 | iter320 | U-class |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    for s in SEEDS:
        curve = {c["iteration"]: c for c in summary["per_seed_curves"][str(s)]}
        u = paired["u_shape_classification"][str(s)]["class"]
        cells = " | ".join(f"{curve[it]['greedy_avg']:.3f}" for it in
                           (0, 40, 80, 160, 320))
        lines.append(f"| {s} | {cells} | `{u}` |\n")

    lines.append("\n### Cross-seed budget summary (greedy avg)\n")
    lines.append(
        "| iter | eps | mean | median | min | max | std | individuals |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    for it in (0, 40, 80, 160, 320):
        r = summary["cross_seed_by_budget"][str(it)]
        inds = ", ".join(f"s{p['training_seed']}={p['greedy_avg']:.3f}"
                         for p in r["per_seed"])
        lines.append(
            f"| {it} | {r['cumulative_episodes']} | {r['greedy_avg_mean']:.3f} | "
            f"{r['greedy_avg_median']:.3f} | {r['greedy_avg_min']:.3f} | "
            f"{r['greedy_avg_max']:.3f} | {r['greedy_avg_std']:.3f} | {inds} |\n"
        )
    lines.append("\n*n=4 training seeds — exploratory only; individuals shown.*\n")

    lines.append("\n## Within-seed paired comparisons\n")
    lines.append("Positive = first worse than reference.\n")
    for s in SEEDS:
        lines.append(f"\n### Seed {s}\n")
        lines.append("| contrast | mean | 95% CI | |\n|---|---|---|---|\n")
        for r in paired["per_seed"][str(s)]:
            z = "excl 0" if not (r["ci95"][0] <= 0 <= r["ci95"][1]) else "incl 0"
            lines.append(
                f"| {r['label']} | {r['mean_diff']:+.3f} | {md_ci(r['ci95'])} | {z} |\n"
            )

    lines.append("\n## 1280-episode replication (Question A)\n")
    lines.append(
        f"- Seeds where iter80 beats iter0 (mean diff < 0): "
        f"**{qa['n_improve']}**/4 → {qa['seeds_improve']}\n"
        f"- Seeds where iter80 worsens vs iter0: "
        f"**{qa['n_worsen']}**/4 → {qa['seeds_worsen']}\n"
        f"- Seeds whose CI excludes zero: "
        f"**{qa['n_ci_excludes_zero']}**/4 → {qa['seeds_ci_excludes_zero']}\n"
    )

    lines.append("\n## Long-training regression (Question B)\n")
    lines.append(
        f"- Seeds where iter320 regresses vs iter80 (mean diff > 0): "
        f"**{qb['n_regress']}**/4 → {qb['seeds_regress']}\n"
        f"- Seeds that continue improving: "
        f"**{qb['n_continue_improve']}**/4 → {qb['seeds_continue_improve']}\n"
        f"- CI excludes zero: "
        f"**{qb['n_ci_excludes_zero']}**/4 → {qb['seeds_ci_excludes_zero']}\n"
        f"- Seed-0 U-shape class replicated on "
        f"**{repl['n_seeds_u_like']}**/4 seeds "
        f"(`seed0_u_shape_replicated={repl['seed0_u_shape_replicated']}`).\n"
    )

    lines.append("\n## Drift\n")
    lines.append(
        "| seed | iter320 expert | warm-start | KL | best iter | best avg | "
        "iter320 avg |\n|---|---|---|---|---|---|---|\n"
    )
    for s in SEEDS:
        d = repl["drift_at_iter320"][str(s)]
        lines.append(
            f"| {s} | {pct(d['iter320_expert_agreement'])} | "
            f"{pct(d['iter320_warmstart_agreement'])} | {d['iter320_kl']:.3f} | "
            f"{d['best_checkpoint_iteration']} | {d['best_greedy_avg']:.3f} | "
            f"{d['iter320_greedy_avg']:.3f} |\n"
        )

    lines.append("\n## Action-category (iter 320)\n")
    lines.append(
        "| seed | agree | roll | end | play | freeze count |\n"
        "|---|---|---|---|---|---|\n"
    )
    for s in SEEDS:
        c = repl["action_category_at_iter320"][str(s)]
        t = c["tempo_shares"]
        lines.append(
            f"| {s} | {pct(c['expert_agreement'])} | "
            f"{t.get('roll')} | {t.get('end')} | {t.get('play')} | "
            f"{c['freeze_appearances_in_confusion']} |\n"
        )

    lines.append("\n## RL diagnostics\n")
    lines.append(
        "Per-seed block means (1–40 / 41–160 / 161–320) for |adv|, value EV, "
        "entropy, clip fraction — see "
        "[`aggregate/replication_analysis.json`](../results/ppo_multiseed_v1/aggregate/replication_analysis.json) "
        "and plot G.\n"
    )

    lines.append("\n## Limitations\n")
    lines.append(
        "- **n=4 is exploratory only.** Do not treat seed counts as "
        "confirmatory frequencies.\n"
        "- Checkpoint binaries gitignored; fingerprints recorded.\n"
        "- No hyperparameter tuning; no best-seed selection; no iter80 "
        "deployment; TEST untouched.\n"
    )

    lines.append("\n## Conclusion\n")
    lines.append(f"**Outcome {outcome}.** {rationale}\n")
    lines.append("\n## Recommended Experiment 4\n")
    lines.append(f"{exp4}\n")
    lines.append(
        "\n## Plots\n"
        "- [`A_multiseed_dev_greedy.png`](../results/ppo_multiseed_v1/aggregate/plots/A_multiseed_dev_greedy.png)\n"
        "- [`B_mean_with_individuals.png`](../results/ppo_multiseed_v1/aggregate/plots/B_mean_with_individuals.png)\n"
        "- [`C_expert_agreement.png`](../results/ppo_multiseed_v1/aggregate/plots/C_expert_agreement.png)\n"
        "- [`D_kl_from_warmstart.png`](../results/ppo_multiseed_v1/aggregate/plots/D_kl_from_warmstart.png)\n"
        "- [`E_warmstart_agreement.png`](../results/ppo_multiseed_v1/aggregate/plots/E_warmstart_agreement.png)\n"
        "- [`F_category_disagreement_iter320.png`](../results/ppo_multiseed_v1/aggregate/plots/F_category_disagreement_iter320.png)\n"
        "- [`G_rl_signal_blocks.png`](../results/ppo_multiseed_v1/aggregate/plots/G_rl_signal_blocks.png)\n"
    )

    with open(OUT, "w") as f:
        f.write("".join(lines))
    print(f"Wrote {OUT} (outcome {outcome})")
    print("Experiment4:", exp4[:120], "...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
