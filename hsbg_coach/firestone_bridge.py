"""Bridge to Firestone's open-source combat simulator (optional upgrade).

When Node + the `bridge/` sidecar are installed, this runs
`@firestone-hs/simulate-bgs-battle` for full-accuracy combat odds. When they
aren't, it transparently falls back to the pure-Python sim in ``sim.py`` — so
callers can always use ``firestone_bridge.simulate(...)`` and get the best
backend available, never an error.

Conversion (our board -> Firestone ``BgsBattleInfo``) is isolated in
``to_bgs_battle_info`` so that when the package's field names shift between
versions, there is exactly one place to fix. See ADR
``decisions/2026-06-24-firestone-bridge.md`` and ``bridge/README.md``.

Decision contract (matches Firestone types as of 2026-06-24):
- input:  BgsBattleInfo { playerBoard, opponentBoard, options, gameState }
- output: SimulationResult { wonPercent, tiedPercent, lostPercent,
                             averageDamageWon, averageDamageLost }
"""

import json
import os
import shutil
import subprocess
from typing import Dict, List, Optional, Sequence

from . import sim as pysim
from .sim import Combatant, SimResult

_BRIDGE_DIR = os.path.join(os.path.dirname(__file__), "..", "bridge")
_SIDECAR = os.path.join(_BRIDGE_DIR, "firestone_sim.js")
_NODE_MODULES = os.path.join(_BRIDGE_DIR, "node_modules")

# Default placeholder hero — overridden when a real hero card id is known.
_DEFAULT_HERO = "TB_BaconShop_HERO_KelThuzad"


def is_available() -> bool:
    """True only if Node, the sidecar, and its node_modules all exist."""
    return bool(
        shutil.which("node")
        and os.path.isfile(_SIDECAR)
        and os.path.isdir(_NODE_MODULES)
    )


def _entity(c: Combatant, entity_id: int) -> Dict:
    """One Firestone BoardEntity. card_id required for full effect modeling;
    stats-only entities still simulate (just without card text)."""
    return {
        "cardId": getattr(c, "card_id", None) or c.name or "",
        "entityId": entity_id,
        "attack": int(c.attack),
        "health": int(c.health),
        "divineShield": bool(c.divine_shield),
        "taunt": bool(c.taunt),
        "poisonous": bool(c.poisonous),
        "reborn": bool(c.reborn),
        "windfury": bool(c.windfury),
        "cleave": bool(c.cleave),
        "enchantments": [],
    }


def _board_info(board: List[Combatant], hero: Optional[str],
                tavern_tier: int, hp: int, start_id: int) -> Dict:
    return {
        "board": [_entity(c, start_id + i) for i, c in enumerate(board)],
        "player": {
            "cardId": hero or _DEFAULT_HERO,
            "heroPowerId": None,
            "tavernTier": int(tavern_tier),
            "hpLeft": int(hp),
        },
        "secrets": [],
    }


def to_bgs_battle_info(
    my_board: Sequence,
    enemy_board: Sequence,
    turn: int = 1,
    my_tier: int = 1,
    enemy_tier: int = 1,
    my_hp: int = 30,
    enemy_hp: int = 30,
    my_hero: Optional[str] = None,
    enemy_hero: Optional[str] = None,
    runs: int = 1000,
) -> Dict:
    """Convert two boards into a Firestone BgsBattleInfo dict."""
    me = [m if isinstance(m, Combatant) else Combatant.from_minion(m) for m in my_board]
    opp = [m if isinstance(m, Combatant) else Combatant.from_minion(m) for m in enemy_board]
    return {
        "playerBoard": _board_info(me, my_hero, my_tier, my_hp, 1000),
        "opponentBoard": _board_info(opp, enemy_hero, enemy_tier, enemy_hp, 2000),
        "options": {"numberOfSimulations": int(runs), "maxAcceptableDuration": 4000},
        "gameState": {"currentTurn": int(turn)},
    }


def _run_sidecar(battle_info: Dict, timeout: float = 15.0) -> Optional[Dict]:
    try:
        proc = subprocess.run(
            ["node", _SIDECAR],
            input=json.dumps(battle_info),
            capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def _to_sim_result(raw: Dict, runs: int) -> SimResult:
    won = raw.get("wonPercent", 0.0) or 0.0
    tied = raw.get("tiedPercent", 0.0) or 0.0
    lost = raw.get("lostPercent", 0.0) or 0.0
    return SimResult(
        runs=runs,
        wins=round(runs * won / 100.0),
        ties=round(runs * tied / 100.0),
        losses=round(runs * lost / 100.0),
        avg_damage_dealt=raw.get("averageDamageWon", 0.0) or 0.0,
        avg_damage_taken=raw.get("averageDamageLost", 0.0) or 0.0,
    )


def simulate(
    my_board: Sequence,
    enemy_board: Sequence,
    runs: int = 1000,
    seed: Optional[int] = None,
    force_fallback: bool = False,
    **info_kwargs,
) -> SimResult:
    """Best-available combat sim: Firestone if installed, else the pure sim.

    Extra kwargs (turn, my_tier, my_hp, my_hero, ...) are passed to the Firestone
    converter and ignored by the fallback.
    """
    if not force_fallback and is_available():
        battle_info = to_bgs_battle_info(my_board, enemy_board, runs=runs, **info_kwargs)
        raw = _run_sidecar(battle_info)
        if raw is not None:
            return _to_sim_result(raw, runs)
        # sidecar failed at runtime — fall through to pure sim
    return pysim.simulate(
        my_board, enemy_board, runs=runs, seed=seed,
        tier_a=info_kwargs.get("my_tier", 1),
        tier_b=info_kwargs.get("enemy_tier", 1),
    )


def backend_name() -> str:
    return "firestone" if is_available() else "python-fallback"
