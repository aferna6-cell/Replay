"""daily_pipeline orchestration with the network, states, labels and training stubbed.

Checks the contract that matters for an unattended daily job: each batch only
ever processes its own new games, results merge into the corpus, a crash
resumes, a failed train is retried, and a candidate is installed only when the
trainer's gate passes and it does not lose to the live policy.
"""

import io
import json
import os
import types
import zipfile

import pytest

from hsbg_coach import daily_pipeline as dp
from hsbg_coach import firestone_replays as fr
from hsbg_coach import replay_labels, replay_states

XML = b'<?xml version="1.0" encoding="utf-8"?><HSReplay><Game/></HSReplay>'


def _zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("replay.xml", XML)
    return buf.getvalue()


def _record(rid, ts):
    return {"reviewId": rid, "replayKey": f"k/{rid}.xml.zip", "buildNumber": 253216,
            "playerRank": "7000", "creationTimestamp": ts,
            "creationDate": "2026-09-29T00:00:00.000Z", "finalComp": {"turn": 5, "board": []}}


class World:
    """Fake Firestone + fake stage outputs + recorded commands."""

    def __init__(self, tmp_path, monkeypatch):
        self.root = tmp_path / "firestone"
        self.repo = tmp_path / "repo"
        (self.repo / "ml").mkdir(parents=True)
        self.listing = []
        self.cmds = []
        self.states_seen = []
        self.labels_seen = []
        self.fail_states = False
        self.train_code = 0
        self.metrics = {"gate": {"result": "PASS"},
                        "compare": {"kind": "bc_checkpoint", "model_minus_compare": {"top1": 0.01}}}
        self.smoke_code = {"candidate": 0, "live": 0}
        monkeypatch.setattr(replay_states, "load_card_names", lambda path: {})
        monkeypatch.setattr(dp, "_states_shard", self._states)
        monkeypatch.setattr(replay_labels, "run", self._labels)

    def get(self, url):
        if url == fr.PERFECT_GAMES_URL:
            return json.dumps(self.listing).encode()
        return _zip()

    def _states(self, job):
        raw, manifest, out, _cards, _opp = job
        if self.fail_states:
            raise OSError("disk full")
        ids = sorted(f[:-7] for f in os.listdir(raw))
        self.states_seen.append(ids)
        os.makedirs(os.path.join(out, "quarantine"), exist_ok=True)
        for gid in ids:
            if gid.startswith("q"):
                open(os.path.join(out, "quarantine", gid + ".json"), "w").write("{}")
            else:
                open(os.path.join(out, gid + ".jsonl.gz"), "w").write("s")
        passed = [i for i in ids if not i.startswith("q")]
        return {"games_processed": len(ids), "passed": len(passed), "rows_written": len(passed),
                "failures_by_reason": {"final_board_mismatch": [i for i in ids if i.startswith("q")]}}

    def _labels(self, states_dir, replays_dir, manifest, out, corpus_manifest=None, **_):
        games = fr.load_manifest(manifest)[1]
        self.labels_seen.append((sorted(g["reviewId"] for g in games),
                                 len(fr.load_manifest(corpus_manifest)[1])))
        os.makedirs(out, exist_ok=True)
        for g in games:
            if os.path.isfile(os.path.join(states_dir, g["reviewId"] + ".jsonl.gz")):
                open(os.path.join(out, g["reviewId"] + ".jsonl.gz"), "w").write("l")
        return {"rows": len(games), "match_rate": 0.97, "games_dropped": []}

    def run_cmd(self, cmd, log_path):
        self.cmds.append(cmd)
        if "ml.train_bc_policy" in cmd:
            if self.train_code == 0:
                out = cmd[cmd.index("--out") + 1]
                open(out, "w").write("candidate")
                json.dump(self.metrics, open(cmd[cmd.index("--metrics") + 1], "w"))
            return self.train_code
        if any("policy_smoke.py" in c for c in cmd):
            return self.smoke_code["candidate" if "--checkpoint" in cmd else "live"]
        if "-c" in cmd:                                    # install(candidate)
            live = self.repo / "ml" / "policy_net.pt"
            if live.exists():
                (self.repo / "ml" / "policy_net.prev.pt").write_text(live.read_text())
            live.write_text(open(cmd[-1]).read())
            return 0
        return 0

    def pipeline(self, **kw):
        args = dp.parse_args(["--data-root", str(self.root), "--workers", "1", "--pause", "0"])
        for k, v in kw.items():
            setattr(args, k, v)
        paths = dp.Paths(str(self.root), repo=str(self.repo))
        return dp.Pipeline(paths, args, log=lambda *_: None, get=self.get, run_cmd=self.run_cmd)

    def state(self):
        return json.load(open(self.root / "pipeline_state.json"))

    def live(self):
        p = self.repo / "ml" / "policy_net.pt"
        return p.read_text() if p.exists() else None


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_two_days_each_batch_only_sees_its_new_games(world):
    world.listing = [_record("a", 1), _record("qbad", 2)]
    [s1] = world.pipeline().run()
    assert s1["ok"] and s1["new_games"] == 2 and s1["states_passed"] == 1
    world.listing += [_record("b", 3)]
    [s2] = world.pipeline().run()
    assert s2["new_games"] == 1
    assert world.states_seen == [["a", "qbad"], ["b"]]          # never re-processes day 1
    assert world.labels_seen == [(["a", "qbad"], 2), (["b"], 3)]  # weights from the corpus
    root = world.root
    assert sorted(os.listdir(root / "raw")) == ["a.xml.gz", "b.xml.gz", "manifest.json", "qbad.xml.gz"]
    assert sorted(os.listdir(root / "states" / "v2" / "quarantine")) == ["qbad.json"]
    assert sorted(f for f in os.listdir(root / "labels" / "v2") if f.endswith(".gz")) == [
        "a.jsonl.gz", "b.jsonl.gz"]
    assert [g["reviewId"] for g in fr.load_manifest(str(root / "raw" / "manifest.json"))[1]] == [
        "a", "qbad", "b"]


def test_trains_on_all_labels_and_installs_on_pass(world):
    world.listing = [_record("a", 1)]
    [s] = world.pipeline().run()
    train = [c for c in world.cmds if "ml.train_bc_policy" in c][0]
    assert train[train.index("--data") + 1].endswith(os.path.join("labels", "v2", "*.jsonl.gz"))
    assert "--make-heldout" in train and "--install" not in train
    assert train[train.index("--compare") + 1].endswith(os.path.join("ml", "policy_net.pt"))
    assert s["promote"]["install"] is True and world.live() == "candidate"
    assert world.state()["train_pending"] is False


def test_gate_fail_keeps_live_policy(world):
    (world.repo / "ml" / "policy_net.pt").write_text("old")
    world.metrics["gate"] = {"result": "FAIL", "failing_types": ["buy"], "overall": {"pass": True}}
    world.listing = [_record("a", 1)]
    [s] = world.pipeline().run()
    assert s["promote"]["install"] is False and world.live() == "old"
    assert not any("-c" in c for c in world.cmds)


def test_candidate_smoke_failure_blocks_install_and_live_smoke_failure_rolls_back(world):
    (world.repo / "ml" / "policy_net.pt").write_text("old")
    world.listing = [_record("a", 1)]
    world.smoke_code["candidate"] = 1
    [s] = world.pipeline().run()
    assert s["promote"]["install"] is False and world.live() == "old"

    world.smoke_code = {"candidate": 0, "live": 1}
    world.listing += [_record("b", 2)]
    [s] = world.pipeline().run()
    assert s["promote"].get("rolled_back") and world.live() == "old"


def test_no_new_games_no_train_and_no_batch_left_behind(world):
    world.listing = [_record("a", 1)]
    world.pipeline().run()
    trains = len([c for c in world.cmds if "ml.train_bc_policy" in c])
    [s] = world.pipeline().run()
    assert s["new_games"] == 0 and s["train"] == {"skipped": "no new labeled games since the last train"}
    assert len([c for c in world.cmds if "ml.train_bc_policy" in c]) == trains
    assert len(os.listdir(world.root / "batches")) == 1       # only day 1's batch kept


def test_failed_train_is_retried_next_run_without_new_games(world):
    world.listing = [_record("a", 1)]
    world.train_code = 1
    [s] = world.pipeline().run()
    assert not s["ok"] and s["finished"] and world.state()["train_pending"] is True
    assert s["promote"]["skipped"].startswith("training failed")
    world.train_code = 0
    [s] = world.pipeline().run()
    assert s["new_games"] == 0 and s["promote"]["install"] is True
    assert world.state()["train_pending"] is False


def test_crash_in_a_data_stage_resumes_then_runs_today(world):
    world.listing = [_record("a", 1)]
    world.fail_states = True
    [s] = world.pipeline().run()
    assert not s["ok"] and not s["finished"]
    assert not (world.root / "raw" / "a.xml.gz").exists()     # nothing merged yet
    world.fail_states = False
    world.listing += [_record("b", 2)]
    resumed, today = world.pipeline().run()
    assert resumed["finished"] and today["new_games"] == 1
    assert world.states_seen == [["a"], ["b"]]


def test_lock_blocks_a_second_run(world):
    os.makedirs(world.root, exist_ok=True)
    (world.root / ".daily_pipeline.lock").write_text("1")
    with pytest.raises(RuntimeError):
        world.pipeline().run()


def test_no_install_flag(world):
    world.listing = [_record("a", 1)]
    [s] = world.pipeline(no_install=True).run()
    assert s["promote"]["install"] is False and world.live() is None


@pytest.mark.parametrize("metrics,install", [
    ({"gate": {"result": "PASS"}, "compare": {"kind": "advisor"}}, True),   # no live policy yet
    ({"gate": {"result": "PASS"},
      "compare": {"kind": "bc_checkpoint", "model_minus_compare": {"top1": -0.002}}}, False),
    ({"gate": {"result": "PASS"}, "compare": {"kind": "bc_checkpoint", "error": "bad file"}}, False),
    ({"gate": None, "compare": None}, False),
])
def test_promotion_decision(metrics, install):
    assert dp.promotion_decision(metrics)["install"] is install
