"""Daily Firestone pipeline: fetch new games -> states -> labels -> train -> promote.

Runs the existing stages unchanged, one batch of new games at a time:

  1. fetch     new first-place games from Firestone (hsbg_coach.firestone_replays)
               into ``batches/<id>/raw``
  2. states    hsbg_coach.replay_states on the batch only (sharded over --workers)
  3. merge     batch replays/states/quarantine -> the corpus folders; batch
               entries appended to the corpus manifest
  4. labels    hsbg_coach.replay_labels on the batch, weights from the whole corpus
  5. merge     batch labels/quarantine -> the corpus labels folder
  6. train     ml.train_bc_policy on every label file (frozen held-out set,
               --compare against the live ml/policy_net.pt), in a subprocess
  7. promote   install the candidate ONLY if the trainer's gate is PASS, it is not
               worse than the live policy on held-out top-1, and
               scripts/policy_smoke.py passes; roll back if the live smoke fails

Each stage runs on the batch folder and only then moves results into the corpus,
so a stage can never touch (or quarantine) games from earlier days. A crash
in a data stage resumes at that stage on the next run; a failed train/promote is
recorded and retried next run (``train_pending``) even if no new games arrived.

    python -m hsbg_coach.daily_pipeline                    # the daily job
    python -m hsbg_coach.daily_pipeline --no-train         # data only
    python -m hsbg_coach.daily_pipeline --status           # last runs + corpus size

Layout (all under data/firestone/, gitignored):
  raw/<reviewId>.xml.gz, raw/manifest.json   corpus replays + manifest
  states/v2/, labels/v2/                     corpus states / labels (+ quarantine/)
  batches/<id>/                              one run: raw, shards, states, labels,
                                             status.json, train.log
  pipeline_state.json                        train_pending + last install
  pipeline_runs.jsonl                        one summary line per run
"""

import argparse
import datetime as dt
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List, Optional

from . import firestone_replays as fr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGES = ("fetch", "states", "merge_states", "labels", "merge_labels", "train", "promote")
LOCK_STALE_S = 20 * 3600


class Paths:
    def __init__(self, root: str = os.path.join(REPO, "data", "firestone"), repo: str = REPO,
                 manifest: Optional[str] = None):
        self.repo = repo
        self.root = root
        self.raw = os.path.join(root, "raw")
        self.manifest = manifest or os.path.join(self.raw, "manifest.json")
        self.states = os.path.join(root, "states", "v2")
        self.labels = os.path.join(root, "labels", "v2")
        self.cards = os.path.join(root, "cards.json")
        self.batches = os.path.join(root, "batches")
        self.state_file = os.path.join(root, "pipeline_state.json")
        self.runs_log = os.path.join(root, "pipeline_runs.jsonl")
        self.lock = os.path.join(root, ".daily_pipeline.lock")
        self.live_policy = os.path.join(repo, "ml", "policy_net.pt")
        self.prev_policy = os.path.join(repo, "ml", "policy_net.prev.pt")
        self.results = os.path.join(repo, "results")


# --- small helpers -------------------------------------------------------------

def _read_json(path: str, default=None):
    if not os.path.isfile(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, path)


def _move_tree_files(src: str, dst: str, suffix: str) -> int:
    """Move src/*<suffix> into dst (overwriting). Returns the count moved."""
    if not os.path.isdir(src):
        return 0
    os.makedirs(dst, exist_ok=True)
    n = 0
    for name in os.listdir(src):
        if name.endswith(suffix) and os.path.isfile(os.path.join(src, name)):
            os.replace(os.path.join(src, name), os.path.join(dst, name))
            n += 1
    return n


def _link_or_copy(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


class Lock:
    """Single-flight guard: one daily run at a time; a lock older than 20h is stale."""

    def __init__(self, path: str):
        self.path = path

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if os.path.exists(self.path) and time.time() - os.path.getmtime(self.path) > LOCK_STALE_S:
            os.remove(self.path)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(f"another run holds {self.path} (delete it if no run is active)")
        with os.fdopen(fd, "w") as fh:
            fh.write(f"{os.getpid()} {dt.datetime.now().isoformat()}\n")
        return self

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass


# --- states sharding (module level so ProcessPoolExecutor can pickle it) ---------

def _states_shard(job) -> Dict:
    from . import replay_states
    shard_raw, shard_manifest, shard_out, cards, track_opponents = job
    return replay_states.run(shard_raw, shard_manifest, shard_out, cards,
                             track_opponents=track_opponents)


# --- promotion decision (pure, unit-tested) ---------------------------------------

def promotion_decision(metrics: Dict, min_gain: float = 0.0) -> Dict:
    """Install only on gate PASS and not worse than the live BC policy on held-out top-1."""
    reasons = []
    gate = metrics.get("gate")
    if not gate:
        reasons.append("no promotion gate in metrics (advisor baseline unavailable)")
    elif gate.get("result") != "PASS":
        failing = gate.get("failing_types") or []
        overall = gate.get("overall") or {}
        reasons.append("gate FAIL" + (f": types {failing}" if failing else "")
                       + ("" if overall.get("pass", True) else
                          f"; overall top-1 {overall.get('model_coarse_top1')} < "
                          f"{overall.get('best_advisor')} {overall.get('best_advisor_top1')}"))
    compare = metrics.get("compare") or {}
    gain = None
    if compare.get("kind") == "bc_checkpoint":
        if compare.get("error"):
            reasons.append(f"could not score the live policy: {compare['error']}")
        else:
            gain = (compare.get("model_minus_compare") or {}).get("top1")
            if gain is None:
                reasons.append("no held-out comparison against the live policy")
            elif gain < min_gain:
                reasons.append(f"held-out top-1 {gain:+.4f} vs live policy (need >= {min_gain:+.4f})")
    return {"install": not reasons, "reasons": reasons, "gain_vs_live_top1": gain,
            "gate": (gate or {}).get("result")}


# --- the pipeline ------------------------------------------------------------------

class Pipeline:
    def __init__(self, paths: Paths, args, log: Callable[[str], None] = print,
                 get: fr.HttpGet = fr.http_get, run_cmd: Optional[Callable] = None):
        self.p = paths
        self.a = args
        self.log = log
        self.get = get
        self.run_cmd = run_cmd or self._run_cmd

    # plumbing
    def _run_cmd(self, cmd: List[str], log_path: str) -> int:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"$ {' '.join(cmd)}\n")
            fh.flush()
            return subprocess.call(cmd, cwd=self.p.repo, stdout=fh, stderr=subprocess.STDOUT)

    def _status_path(self, batch: str) -> str:
        return os.path.join(batch, "status.json")

    def _save(self, batch: str, status: Dict) -> None:
        _write_json(self._status_path(batch), status)

    def unfinished_batch(self) -> Optional[str]:
        for name in sorted(os.listdir(self.p.batches)) if os.path.isdir(self.p.batches) else []:
            st = _read_json(os.path.join(self.p.batches, name, "status.json"), {})
            if st and not st.get("finished"):
                return os.path.join(self.p.batches, name)
        return None

    def new_batch(self) -> str:
        base = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        stamp, n = base, 1
        while os.path.exists(os.path.join(self.p.batches, stamp)):   # never reuse a batch
            n += 1
            stamp = f"{base}-{n}"
        batch = os.path.join(self.p.batches, stamp)
        os.makedirs(batch)
        self._save(batch, {"id": stamp, "created": dt.datetime.now().isoformat(),
                           "done": [], "finished": False})
        return batch

    # stages -------------------------------------------------------------------
    def stage_fetch(self, batch: str, st: Dict) -> None:
        raw = os.path.join(batch, "raw")
        shutil.rmtree(raw, ignore_errors=True)       # a resumed fetch starts clean
        res = fr.fetch_new(raw, self.p.manifest, [self.p.raw], self.a.max_new, self.get,
                           self.a.pause, self.log)
        fr.save_manifest(os.path.join(batch, "manifest.json"), {"games": []}, res["entries"])
        builds: Dict[str, int] = {}
        for e in res["entries"]:
            builds[str(e.get("buildNumber"))] = builds.get(str(e.get("buildNumber")), 0) + 1
        st["fetch"] = {"listed": res["listed"], "selected": res["selected"],
                       "downloaded": len(res["entries"]), "failed": res["failed"],
                       "builds": builds}

    def stage_states(self, batch: str, st: Dict) -> None:
        from . import replay_states
        games = fr.load_manifest(os.path.join(batch, "manifest.json"))[1]
        out = os.path.join(batch, "states")
        shutil.rmtree(out, ignore_errors=True)
        shutil.rmtree(os.path.join(batch, "shards"), ignore_errors=True)
        if not games:
            st["states"] = {"games_processed": 0, "passed": 0}
            return
        replay_states.load_card_names(self.p.cards)  # download once, before the shards
        n = max(1, min(self.a.workers, len(games)))
        jobs = []
        for k in range(n):
            shard = os.path.join(batch, "shards", str(k))
            os.makedirs(os.path.join(shard, "raw"), exist_ok=True)
            part = games[k::n]
            for g in part:
                _link_or_copy(os.path.join(batch, "raw", f"{g['reviewId']}.xml.gz"),
                              os.path.join(shard, "raw", f"{g['reviewId']}.xml.gz"))
            fr.save_manifest(os.path.join(shard, "manifest.json"), {"games": []}, part)
            jobs.append((os.path.join(shard, "raw"), os.path.join(shard, "manifest.json"),
                         os.path.join(shard, "out"), self.p.cards, not self.a.no_opponents))
        self.log(f"states: {len(games)} games over {n} process(es)")
        if n == 1:
            reports = [_states_shard(jobs[0])]
        else:
            with ProcessPoolExecutor(max_workers=n) as ex:
                reports = list(ex.map(_states_shard, jobs))
        for job in jobs:
            _move_tree_files(job[2], out, ".jsonl.gz")
            _move_tree_files(os.path.join(job[2], "quarantine"),
                             os.path.join(out, "quarantine"), ".json")
        failures: Dict[str, List[str]] = {}
        for r in reports:
            for reason, ids in r.get("failures_by_reason", {}).items():
                failures.setdefault(reason, []).extend(ids)
        summary = {"games_processed": sum(r["games_processed"] for r in reports),
                   "passed": sum(r["passed"] for r in reports),
                   "rows_written": sum(r["rows_written"] for r in reports),
                   "failures_by_reason": failures}
        _write_json(os.path.join(out, "report.json"), dict(summary, shards=reports))
        shutil.rmtree(os.path.join(batch, "shards"), ignore_errors=True)
        st["states"] = summary
        self.log(f"states: {summary['passed']}/{summary['games_processed']} passed")

    def stage_merge_states(self, batch: str, st: Dict) -> None:
        moved = _move_tree_files(os.path.join(batch, "raw"), self.p.raw, ".xml.gz")
        out = os.path.join(batch, "states")
        _move_tree_files(out, self.p.states, ".jsonl.gz")
        _move_tree_files(os.path.join(out, "quarantine"),
                         os.path.join(self.p.states, "quarantine"), ".json")
        entries = fr.load_manifest(os.path.join(batch, "manifest.json"))[1]
        added = fr.append_to_manifest(self.p.manifest, entries)
        st["merge_states"] = {"replays_moved": moved, "manifest_added": added}

    def stage_labels(self, batch: str, st: Dict) -> None:
        from . import replay_labels
        out = os.path.join(batch, "labels")
        shutil.rmtree(out, ignore_errors=True)
        games = fr.load_manifest(os.path.join(batch, "manifest.json"))[1]
        if not games:
            st["labels"] = {"games_labeled": 0, "rows": 0}
            return
        s = replay_labels.run(self.p.states, self.p.raw, os.path.join(batch, "manifest.json"),
                              out, corpus_manifest=self.p.manifest, workers=self.a.workers)
        labeled = len(glob.glob(os.path.join(out, "*.jsonl.gz")))
        st["labels"] = {"games_labeled": labeled, "rows": s.get("rows"),
                        "match_rate": s.get("match_rate"),
                        "games_dropped": len(s.get("games_dropped") or [])}
        self.log(f"labels: {labeled} games, {s.get('rows')} rows, match {s.get('match_rate')}")

    def stage_merge_labels(self, batch: str, st: Dict) -> None:
        out = os.path.join(batch, "labels")
        moved = _move_tree_files(out, self.p.labels, ".jsonl.gz")
        _move_tree_files(os.path.join(out, "quarantine"),
                         os.path.join(self.p.labels, "quarantine"), ".jsonl")
        st["merge_labels"] = {"games_added": moved}
        if moved:
            state = _read_json(self.p.state_file, {})
            state["train_pending"] = True
            _write_json(self.p.state_file, state)

    def stage_train(self, batch: str, st: Dict) -> None:
        state = _read_json(self.p.state_file, {})
        if self.a.no_train:
            st["train"] = {"skipped": "--no-train"}
            return
        if not (state.get("train_pending") or self.a.force_train):
            st["train"] = {"skipped": "no new labeled games since the last train"}
            return
        stamp = st["id"]
        os.makedirs(self.p.results, exist_ok=True)
        cand = os.path.join(self.p.results, f"policy_net_{stamp}.pt")
        metrics = os.path.join(self.p.results, f"policy_net_{stamp}.metrics.json")
        cmd = [self.a.python, "-m", "ml.train_bc_policy",
               "--data", os.path.join(self.p.labels, "*.jsonl.gz"),
               "--make-heldout", "--compare", self.p.live_policy,
               "--out", cand, "--metrics", metrics, "--workers", str(self.a.workers)]
        if self.a.build:
            cmd += ["--build", self.a.build]
        cmd += self.a.train_arg
        self.log("train: " + " ".join(cmd))
        t0 = time.time()
        code = self.run_cmd(cmd, os.path.join(batch, "train.log"))
        st["train"] = {"exit_code": code, "candidate": cand, "metrics": metrics,
                       "minutes": round((time.time() - t0) / 60, 1)}
        if code != 0 or not os.path.isfile(cand) or not os.path.isfile(metrics):
            raise RuntimeError(f"training failed (exit {code}); see {batch}/train.log")

    def stage_promote(self, batch: str, st: Dict) -> None:
        tr = st.get("train") or {}
        if "candidate" not in tr or tr.get("exit_code") != 0 or not os.path.isfile(tr["candidate"]):
            st["promote"] = {"skipped": "nothing trained" if "candidate" not in tr
                             else "training failed; train_pending stays set"}
            return
        metrics = _read_json(tr["metrics"], {})
        decision = promotion_decision(metrics, self.a.min_gain)
        log_path = os.path.join(batch, "train.log")
        smoke = [self.a.python, os.path.join("scripts", "policy_smoke.py")]
        if decision["install"]:
            if self.run_cmd(smoke + ["--checkpoint", tr["candidate"]], log_path) != 0:
                decision = dict(decision, install=False,
                                reasons=["candidate failed scripts/policy_smoke.py"])
        if decision["install"] and self.a.no_install:
            decision = dict(decision, install=False, reasons=["--no-install (would have installed)"])
        if decision["install"]:
            code = self.run_cmd([self.a.python, "-c",
                                 "import sys; from ml.train_bc_policy import install; "
                                 "install(sys.argv[1])", tr["candidate"]], log_path)
            if code != 0:
                raise RuntimeError(f"install failed (exit {code}); see {log_path}")
            if self.run_cmd(smoke, log_path) != 0:
                if os.path.isfile(self.p.prev_policy):
                    shutil.copy2(self.p.prev_policy, self.p.live_policy)
                decision = dict(decision, install=False, rolled_back=True,
                                reasons=["live smoke failed after install; rolled back"])
        st["promote"] = decision
        state = _read_json(self.p.state_file, {})
        state["train_pending"] = False
        state["last_train"] = {"batch": st["id"], "candidate": tr["candidate"],
                               "installed": decision["install"]}
        if decision["install"]:
            state["last_install"] = state["last_train"]
        _write_json(self.p.state_file, state)
        self.log("promote: " + ("INSTALLED " + tr["candidate"] if decision["install"]
                                else "kept live policy (" + "; ".join(decision["reasons"]) + ")"))

    # driver -------------------------------------------------------------------
    def run_batch(self, batch: str) -> Dict:
        """Data stages block (a failure leaves the batch to resume next run); a
        train/promote failure is recorded, the batch still finishes, and
        train_pending makes the next run train again."""
        st = _read_json(self._status_path(batch))
        t0 = time.time()
        error = None
        for stage in STAGES:
            if stage in st["done"]:
                continue
            try:
                getattr(self, f"stage_{stage}")(batch, st)
            except Exception as exc:
                error = f"{stage}: {type(exc).__name__}: {exc}"
                st.setdefault("errors", []).append(
                    {"stage": stage, "error": error, "at": dt.datetime.now().isoformat()})
                self.log(f"!! {error}")
                if stage not in ("train", "promote"):
                    self._save(batch, st)
                    break
                if stage == "train":
                    st["done"].append("train")    # promote then skips: nothing to install
                    continue
            st["done"].append(stage)
            self._save(batch, st)
        data_done = "merge_labels" in st["done"]
        if data_done:
            st["finished"] = True
            st["runtime_min"] = round((time.time() - t0) / 60, 1)
            self._save(batch, st)
        summary = {"batch": st["id"], "at": dt.datetime.now().isoformat(),
                   "ok": error is None, "error": error, "finished": data_done,
                   "new_games": (st.get("fetch") or {}).get("downloaded"),
                   "fetch_failed": len((st.get("fetch") or {}).get("failed") or []),
                   "states_passed": (st.get("states") or {}).get("passed"),
                   "labeled_games": (st.get("labels") or {}).get("games_labeled"),
                   "train": st.get("train"), "promote": st.get("promote")}
        with open(self.p.runs_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary) + "\n")
        if data_done and not summary["new_games"] and not (st.get("train") or {}).get("candidate") \
                and not st.get("errors"):
            shutil.rmtree(batch, ignore_errors=True)       # keep only batches that did work
        return summary

    def check_corpus(self) -> None:
        """Refuse to start a fresh manifest next to an existing corpus: the new
        one would hold only new games and skew the label MMR weights."""
        if os.path.isfile(self.p.manifest) or not os.path.isdir(self.p.raw):
            return
        replays = [f for f in os.listdir(self.p.raw) if f.endswith(".xml.gz")]
        if replays:
            other = sorted(f for f in os.listdir(self.p.raw) if f.endswith(".json"))
            raise RuntimeError(
                f"{self.p.manifest} not found, but {self.p.raw} already holds {len(replays)} "
                f"replays. Pass --manifest <your corpus manifest>"
                + (f" (candidates in raw/: {', '.join(other)})" if other else "")
                + " so new games are added to it.")

    def run(self) -> List[Dict]:
        """Finish an interrupted batch first, then run today's batch."""
        self.check_corpus()
        with Lock(self.p.lock):
            summaries = []
            batch = self.unfinished_batch()
            if batch:
                self.log(f"resuming unfinished batch {os.path.basename(batch)}")
                summaries.append(self.run_batch(batch))
                if not summaries[-1]["finished"]:
                    return summaries
            summaries.append(self.run_batch(self.new_batch()))
            return summaries


def status(paths: Paths, n: int = 5) -> str:
    lines = []
    games = fr.load_manifest(paths.manifest)[1]
    labels = len(glob.glob(os.path.join(paths.labels, "*.jsonl.gz")))
    lines.append(f"corpus: {len(games)} games in manifest, {labels} labeled")
    state = _read_json(paths.state_file, {})
    lines.append(f"train pending: {state.get('train_pending', False)} · last install: "
                 f"{(state.get('last_install') or {}).get('candidate', 'never')}")
    if os.path.isfile(paths.runs_log):
        with open(paths.runs_log, encoding="utf-8") as fh:
            for row in [json.loads(x) for x in fh.readlines()[-n:]]:
                pr = row.get("promote") or {}
                lines.append(f"  {row['at'][:16]} new={row.get('new_games')} "
                             f"labeled={row.get('labeled_games')} "
                             f"{'installed' if pr.get('install') else pr.get('reasons') or (row.get('train') or {}).get('skipped', '')}"
                             f"{'' if row['ok'] else ' ERROR ' + str(row['error'])}")
    return "\n".join(lines)


def _redirect_output(path: str) -> None:
    """Point fd 1/2 at a log file so stage subprocesses and workers log there too
    (Task Scheduler / cron have no console)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fh = open(path, "a", encoding="utf-8", buffering=1)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(fh.fileno(), 1)
    os.dup2(fh.fileno(), 2)
    sys.stdout = sys.stderr = fh
    print(f"\n===== daily_pipeline {dt.datetime.now().isoformat()} =====")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", default=os.path.join(REPO, "data", "firestone"))
    ap.add_argument("--manifest", default=None,
                    help="corpus manifest (default <data-root>/raw/manifest.json); new games "
                         "are appended to it and it sets the label MMR weights")
    ap.add_argument("--log", default=None,
                    help="append all output (this process and its stages) to this file")
    ap.add_argument("--max-new", type=int, default=None, help="cap games downloaded per run")
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--pause", type=float, default=0.5, help="seconds between replay downloads")
    ap.add_argument("--no-opponents", action="store_true",
                    help="states without per-event opponent capture (~10x faster)")
    ap.add_argument("--no-train", action="store_true", help="fetch/states/labels only")
    ap.add_argument("--force-train", action="store_true", help="train even with no new games")
    ap.add_argument("--no-install", action="store_true", help="train + evaluate, never install")
    ap.add_argument("--min-gain", type=float, default=0.0,
                    help="required held-out top-1 gain over the live policy (default 0)")
    ap.add_argument("--build", default=None, help="passed to train_bc_policy --build")
    ap.add_argument("--train-arg", action="append", default=[],
                    help="extra train_bc_policy argument (repeatable, e.g. --train-arg=--epochs=20)")
    ap.add_argument("--python", default=sys.executable, help="interpreter for training")
    ap.add_argument("--status", action="store_true", help="print corpus + last runs and exit")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.log:
        _redirect_output(args.log)
    paths = Paths(args.data_root, manifest=args.manifest)
    if args.status:
        print(status(paths))
        return 0
    try:
        summaries = Pipeline(paths, args).run()
    except RuntimeError as exc:                   # lock held
        print(f"!! {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summaries, indent=1))
    return 0 if all(s["ok"] for s in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
