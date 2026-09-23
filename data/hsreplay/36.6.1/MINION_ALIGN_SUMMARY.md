# Minion DB vs HSReplay live pool (2026-09-23)

Source: HSReplay `/battlegrounds/minions/` `react_context` (season 14) + HearthstoneJSON `isBattlegroundsPoolMinion`.

## Live lobby tribes (HSReplay)
dragon, quilboar, mech, pirate, beast, demon, elemental, undead, murloc, aberration

**Naga is NOT in the live lobby types** (expected).

## Alignment
- Our DB: 247 minions; after fix, in_pool excludes remaining Naga stragglers
- All non-Naga live-pool minions are present in our DB
- 0 tavern-tier mismatches
- Tribe label diffs were only `mech` vs HearthstoneJSON `mechanical` (normalized to `mech`)
- 22 Naga cards still marked pool in HearthstoneJSON were correctly absent from our live pool (except Ominous Seer / Firescale Hoarder which are now `in_pool: false`)

## Verdict
**Aligned for the current lobby.** Missing-from-ours list was Naga-only noise from HSJSON still flagging Naga as pool minions.
