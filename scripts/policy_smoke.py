"""End-to-end smoke test for the behaviour-cloned NEXT policy.

Feeds the repo-root sample Power.log through BGTracker / LiveCoach with
HSBG_NEXT_POLICY=1 and prints NEXT. Exits nonzero if the checkpoint does not
load or the policy did not produce NEXT.

  python scripts/policy_smoke.py                 # checks ml/policy_net.pt
  python scripts/policy_smoke.py --checkpoint results/policy_net_20260927.pt

The log is replayed up to its last recruit-phase line with no hero / trinket /
discover choice on screen (the sample log ends on a trinket offer).
"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def _last_decision_line(lines):
    """Index of the last line in the recruit phase with no choice on screen."""
    from hsbg_coach.bg import BGTracker, Phase
    from hsbg_coach.choices import ChoiceParser
    from hsbg_coach.parser import parse_line
    tracker, choices, offer, last = BGTracker(), ChoiceParser(), None, None
    for i, line in enumerate(lines):
        o = choices.feed(line)
        if o is not None:
            offer = o
        elif "SendChoices" in line:
            offer = None
        ev = parse_line(line)
        if ev is not None:
            tracker.feed(ev)
        if tracker.in_bg and tracker.phase == Phase.RECRUIT and offer is None:
            last = i
    return last


def _feed(coach, lines):
    """Per line, what LiveCoach._consume does (minus the file tail / recorder)."""
    from hsbg_coach.parser import parse_line
    for line in lines:
        coach._active = True
        offer = coach.choices.feed(line)
        if offer is not None:
            coach._offer = offer
        elif "SendChoices" in line:
            coach._offer = None
        ev = parse_line(line)
        if ev is not None:
            coach.tracker.feed(ev)
            coach._version += 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Smoke-test the NEXT policy on the sample Power.log.")
    p.add_argument("--checkpoint", default=os.path.join(REPO, "ml", "policy_net.pt"))
    p.add_argument("--log", default=os.path.join(REPO, "Power.log"))
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    if not os.path.isfile(a.checkpoint):
        print(f"FAIL: checkpoint not found: {a.checkpoint}")
        return 2
    try:
        from ml.bc_policy import load_bc_policy
        load_bc_policy(a.checkpoint)
    except Exception as exc:
        print(f"FAIL: checkpoint did not load: {exc}")
        return 2
    if not os.path.isfile(a.log):
        print(f"FAIL: log not found: {a.log}")
        return 2
    os.environ["HSBG_NEXT_POLICY"] = "1"
    os.environ["HSBG_NEXT_POLICY_PATH"] = os.path.abspath(a.checkpoint)

    from hsbg_coach.encode import describe_option
    from hsbg_coach.live import LiveCoach
    from hsbg_coach.overlay import format_next

    with open(a.log, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    cut = _last_decision_line(lines)
    if cut is None:
        print("FAIL: no recruit-phase decision point in the log")
        return 1
    coach = LiveCoach(power_log=a.log)
    if coach._next_policy is None:
        print("FAIL: LiveCoach did not load the policy")
        return 1
    _feed(coach, lines[:cut + 1])
    snap, odds, recs = coach.frame()
    if not recs:
        print("FAIL: no NEXT produced")
        return 1
    option = coach._next_policy.best(snap)
    expected = describe_option(snap, option) if option else None
    print(f"turn {snap.get('turn')}  tier {snap.get('tavern_tier')}  gold {snap.get('gold')}")
    print(format_next(snap, odds, recs))
    if recs[0] != expected:
        print(f"FAIL: NEXT {recs[0]!r} is not the policy pick {expected!r}")
        return 1
    print(f"OK: NEXT from policy -> {expected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())