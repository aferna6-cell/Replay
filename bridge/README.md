# Firestone sim bridge (optional, opt-in)

Wraps Firestone's open-source [`@firestone-hs/simulate-bgs-battle`](https://www.npmjs.com/package/@firestone-hs/simulate-bgs-battle)
so the coach can use full-accuracy combat simulation (all deathrattles,
battlecry-summons, hero powers, etc.) instead of the built-in Python sim.

**This is optional.** Without it, `hsbg_coach` uses its own pure-Python sim and
everything works — just with the representative (not exhaustive) card coverage.

## Enable it

```bash
cd bridge
npm install          # pulls the Firestone simulator + reference data
npm run check        # prints the package's exports — sanity check the install
```

Then Python auto-detects the sidecar:

```python
from hsbg_coach import firestone_bridge
print(firestone_bridge.is_available())   # True once node + node_modules present
```

`firestone_bridge.simulate(my_board, enemy_board, ...)` returns the same
`SimResult` shape as the pure sim, so it's a drop-in upgrade.

## How it works

```
Python (firestone_bridge.py)                 Node (firestone_sim.js)
  board -> BgsBattleInfo JSON  --stdin-->     simulateBattle(...)
  SimResult  <--stdout-- {wonPercent,...}     SimulationResult
```

## Validating the template

The field names in `firestone_sim.js` (and the `BgsBattleInfo` builder in
`firestone_bridge.py`) follow the documented Firestone types, but exact
`BoardEntity` fields and the `AllCardsService`/`CardsData` init can shift between
package versions. After `npm install`, confirm against:

```
node_modules/@firestone-hs/simulate-bgs-battle/dist/bgs-battle-info.d.ts
node_modules/@firestone-hs/simulate-bgs-battle/dist/simulation-result.d.ts
```

All Python-side conversion lives in `hsbg_coach/firestone_bridge.py:to_bgs_battle_info`,
so corrections are one function.
