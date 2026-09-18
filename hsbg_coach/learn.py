"""Fold recorded watch games into better recommendations (continual / retrain).

Advisory only: this never autoplays. Training runs offline (or as a background
nice process via continual.BackgroundTrainer). The live coach hot-swaps
ml/eval_net.pt by mtime when a retrain finishes.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional

from . import config


def recorded_game_paths(data_dir: Optional[str] = None) -> List[str]:
    root = data_dir or config.DATA_DIR
    return sorted(glob.glob(os.path.join(root, "game-*.jsonl")))


def validate_trajectories(data_dir: Optional[str] = None) -> Dict[str, Any]:
    """Schema check over recorded JSONL — fast, no torch / no network."""
    paths = recorded_game_paths(data_dir)
    n_rows = 0
    n_with_placement = 0
    n_partial = 0
    errors: List[str] = []
    required = ("state", "action_type", "placement")
    for path in paths:
        if path.endswith(".partial"):
            n_partial += 1
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    n_rows += 1
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        errors.append(f"{path}:{i}: {exc}")
                        continue
                    for k in required:
                        if k not in row:
                            errors.append(f"{path}:{i}: missing {k}")
                    if not isinstance(row.get("state"), dict):
                        errors.append(f"{path}:{i}: state must be object")
                    if row.get("placement") is not None:
                        n_with_placement += 1
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    n_games = len(paths)
    return {
        "data_dir": data_dir or config.DATA_DIR,
        "n_games": n_games,
        "n_rows": n_rows,
        "n_with_placement": n_with_placement,
        "n_partial_skipped": n_partial,
        "personal_weight": round(config.personal_weight(n_games), 4),
        "population_weight": round(config.population_weight(n_games), 4),
        "ok": not errors,
        "errors": errors[:20],
    }


def model_sidecar(model_path: Optional[str] = None) -> Dict[str, Any]:
    """Read eval_net.pt.meta.json (+ file fingerprint) for live-coach verify."""
    path = model_path or config.eval_net_path()
    meta_path = path + ".meta.json"
    out: Dict[str, Any] = {
        "model_path": path,
        "exists": os.path.isfile(path),
        "mtime": None,
        "file_sha256_12": None,
        "meta": None,
    }
    if out["exists"]:
        try:
            out["mtime"] = os.path.getmtime(path)
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 16), b""):
                    h.update(chunk)
            out["file_sha256_12"] = h.hexdigest()[:12]
        except OSError:
            pass
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as fh:
                out["meta"] = json.load(fh)
        except (OSError, json.JSONDecodeError):
            out["meta"] = None
    return out


def learn_status(data_dir: Optional[str] = None,
                 model_path: Optional[str] = None) -> Dict[str, Any]:
    traj = validate_trajectories(data_dir)
    side = model_sidecar(model_path)
    return {"trajectories": traj, "model": side}


def write_learn_meta(model_path: str, *, n_personal_games: int,
                     n_population: int, n_personal_examples: int,
                     personal_w: float, population_w: float,
                     with_context: bool, epochs: int) -> str:
    """Enrich eval_net.pt.meta.json so watch can verify the retrain landed."""
    meta_path = model_path + ".meta.json"
    meta: Dict[str, Any] = {}
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh) or {}
        except (OSError, json.JSONDecodeError):
            meta = {}
    meta.update({
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_personal_games": int(n_personal_games),
        "n_population_examples": int(n_population),
        "n_personal_examples": int(n_personal_examples),
        "personal_weight": round(float(personal_w), 4),
        "population_weight": round(float(population_w), 4),
        "with_context": bool(with_context),
        "epochs": int(epochs),
        "source": "hsbg_coach.learn",
    })
    if os.path.isfile(model_path):
        try:
            meta["mtime"] = os.path.getmtime(model_path)
            h = hashlib.sha256()
            with open(model_path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 16), b""):
                    h.update(chunk)
            meta["file_sha256_12"] = h.hexdigest()[:12]
        except OSError:
            pass
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    return meta_path


def run_retrain(*, data_dir: Optional[str] = None, epochs: int = 40,
                dry_run: bool = False, with_context: bool = True,
                out: Optional[str] = None, comp_source: Optional[str] = None
                ) -> int:
    """CLI entry: validate → (optional) train eval net with adaptive mix."""
    data_dir = data_dir or config.DATA_DIR
    out = out or config.eval_net_path()
    status = validate_trajectories(data_dir)
    print(f"Recorded games: {status['n_games']}  "
          f"rows: {status['n_rows']}  with_placement: {status['n_with_placement']}")
    print(f"Adaptive mix: population={status['population_weight']:.2f}  "
          f"personal={status['personal_weight']:.2f}  "
          f"(WEIGHTING personal_full_at={config.WEIGHTING['personal_full_at_games']})")
    if status["errors"]:
        print("Trajectory issues:")
        for e in status["errors"]:
            print(" ", e)
        if not status["ok"]:
            return 1
    if dry_run:
        side = model_sidecar(out)
        print(f"Dry-run OK. Current model exists={side['exists']} "
              f"sha={side.get('file_sha256_12')} "
              f"meta_personal_games="
              f"{(side.get('meta') or {}).get('n_personal_games')}")
        print("Live coach picks up a new eval_net.pt via mtime hot-swap "
              "(no restart needed).")
        return 0

    # Delegate to the existing trainer; it applies adaptive mix internally.
    from ml.train_eval_net import main as train_main
    argv = ["--epochs", str(epochs), "--trajectories", data_dir, "--out", out]
    if with_context:
        argv.append("--with-context")
    if comp_source:
        argv += ["--comp-source", comp_source]
    rc = train_main(argv)
    if rc == 0:
        side = model_sidecar(out)
        print(f"Model ready: {out}")
        print(f"  fingerprint: {side.get('file_sha256_12')}  "
              f"mtime={side.get('mtime')}")
        meta = side.get("meta") or {}
        if meta:
            print(f"  meta: personal_games={meta.get('n_personal_games')}  "
                  f"personal_weight={meta.get('personal_weight')}  "
                  f"with_context={meta.get('with_context')}")
        print("Next: python -m hsbg_coach watch   "
              "# overlay hot-swaps the new net automatically")
    return int(rc or 0)
