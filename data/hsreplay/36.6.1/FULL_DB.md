# HSReplay DB snapshot — patch 36.6.1
path: `/workspace/hsreplay-db-schema/data/36.6.1`
last_updated: 2026-09-23T18:06:53.667245Z

## Counts
- comps: **24**
- heroes: **118**
- minions: **247** (in_pool 247)
- trinkets: **293** (in_pool 293)
- spells: **78** (in_pool **72**, excluded 6)
- tribes: **10**
- training_notes: **8**

## Train weights (locked)
- comps: highest
- heroes_trinkets: support
- old_patch: 0
- affinity_note: comp membership links seed affinities; model learns hero/trinket/minion/spell affinity to comps during training

## Live tribes
aberration, beast, demon, dragon, elemental, mech, murloc, pirate, quilboar, undead

## Excluded tribes
naga

## Training notes
- **aberration_early_pivot**: Aberrations are strong early–mid; plan to transition into another comp later rather than forcing all-Aberration endgame.
- **murloc_activate_scam**: Murlocs are stronger; Murloc scam uses Activate to put Venomous+Stealth on the T2 hand-summon Murloc, then floods hand scam (like Venom Scam).
- **quilboar_support_blood_gems**: Quilboar are somewhat better as a supporting tribe: Blood Gems excel when casting many spells on minions, and as Aberration discard fuel.
- **spell_play_is_legal**: Playing tavern spells from hand is a legal action the coach must suggest when applicable.
- **discard_is_legal**: Discard is a legal action — suggest it when applicable (Aberration fodder, Sludge/Corrupted Coin double-cast, etc.).
- **sell_before_place_full_board**: If the board is full, suggest selling a card before placing a new one.
- **balinda_spell_on_minion_buys**: Balinda / spell-on-minion comps: always buy 1–2 cost spells that TARGET a minion.
- **t7_spells_rare**: Tier 7 tavern spells are almost never seen — do not plan the game around them.

## Spells in pool by grade
### S (4)
- **Corrupted Coin** (2g T5) — broken if you have a way to discard it (double cast). otherwise still cycleable coin-like value.
- **Eyes of the Earth Mother** (4g T6) — great to golden a powerful under-T4 card. (T7 spells are basically never seen — do not plan around t
- **Lost Staff of Hamuul** (2g T6) — always good — refill tavern with your type; premium finish tool.
- **Weapons Forge** (2g T4) — S-tier in comps that cast a lot of spells on minions (Pointy Arrows / spell-density).
### A (21)
- **Armor Stash** (3g T5) — must-buy when you need to go above 15 health before top 4.
- **Boundless Potential** (3g T4) — really good — fish for an enabler or a premium high-tier spell (Lost Staff, Eyes of the Earth Mother
- **Careful Investment** (1g T3) — always buy unless it is the last turn of the game.
- **Channel the Devourer** (4g T5) — useful when you want to sell a huge-stat minion and park those stats on a keeper.
- **Chef's Choice** (2g T2) — good; great in early game to pivot type while keeping a body.
- **Corrupted Cupcakes** (4g T5) — solid in Demon builds when the shop is buffed (big consumes).
- **Energizing Chamber** (1g T5) — good (buddy +7/+7; double on discard).
- **Hasty Excavation** (3g T2) — good early; always buy on Demons (can rewind the health cost).
- **Hired Headhunter** (3g T5) — must-buy with Brann; battlecry discover is otherwise situational.
- **Leaf Through the Pages** (1g T2) — good. especially strong early for free refreshes.
- **Methodical Madness** (3g T4) — really good for Demons (consume tavern for stats + keywords).
- **Mounting Avalanche** (2g T3) — good to dump sell-stats onto a real piece (Wildfire Elemental, Crackling Cyclone, etc.).
- **Overconfidence** (1g T3) — good especially when you are strong; generally worth buying.
- **Planar Telescope** (4g T3) — good discover of your most common type.
- **Sludge Corrosion** (1g T4) — really good, especially as discard fuel (cast-on-discard).
- **Strike Oil** (3g T2) — great early game economy; bad anywhere else — do not buy mid/late.
- **Tavern Coin** (1g T1) — always buy — free cycle / gold smoothing.
- **Temperature Shift** (4g T4) — really good for Elementals; also strong to copy to improve Ballers.
- **Tomb Turning** (2g T4) — good in Undead comps.
- **Unmasked Identity** (3g T5) — good late to fish strong late HPs (Drek'Thar, Vanndar, Tavish, etc.).
- **Upper Hand** (3g T5) — good late-game combat tech (set enemy health to 1).
### B (24)
- **A New Sprout** (3g T1) — good on turn 1 if the shop is bad; discover T1.
- **Alliance Flag** (1g T1) — good in Balinda / spell-on-minion comps (cheap targetted spell). otherwise fine early stick.
- **Butchering** (2g T5) — good only in Undead comps; skip otherwise.
- **Contracted Corpse** (3g T5) — situational Discover Deathrattle — fine for Undead/Beast DR lines.
- **Deepwater Clan** (2g T4) — fine murloc/tribe buff.
- **Defender's Rites** (2g T4) — fine; good when you specifically want a big Taunt body.
- **Easterly Winds** (1g T4) — fine permanent shop high-roll (+8/+8 on refresh).
- **Enchanted Lasso** (2g T1) — fine in the early game.
- **Fandral's Fortune** (3g T6) — only good in comps that generate Blood Gems / want Choose One combined (spell-on-minion / quilboar s
- **Forest's Bounty** (2g T5) — fine choose-one buff.
- **Fortify** (1g T1) — good in Balinda / spell-on-minion comps; otherwise fine early taunt health.
- **Friendly Bounty** (2g T3) — only really good if you have Privateer (or similar type-payoff); otherwise skip most bounties.
- **Gem Confiscation** (1g T4) — can be good with Quilboar / blood-gem boards; otherwise niche.
- **Golden Touch** (5g T5) — only good if you have a lot of gold to spare.
- **Mighty Dragonbreath** (2g T4) — fine; not the best board buff — better with Dragons / Divine Shield.
- **Recruit a Trainee** (2g T1) — fine in the early game for a body.
- **Repair Job** (2g T3) — fine single-target +4/+8.
- **Saloon's Finest** (2g T5) — okay in comps that want to cast a lot of cheap spells (spell-density).
- **Seafood Stew** (2g T3) — provisional — keyword-stacking board buff (warband bonus keywords). Aidan has not graded this one ye ⚠️ provisional
- **Search Through Time** (2g T2) — fine mid to fish for an enabler; better early. discover your tier and lock 1 turn.
- **Staff of Enrichment** (2g T3) — okay permanent tavern buff — situational, not a must.
- **Tavern Dish Banana** (1g T1) — good in Balinda / spell-on-minion comps; otherwise fine early +2/+2.
- **Wealthy Bounty** (2g T3) — the one bounty worth buying (economy). Friendly Bounty only with Privateer.
- **Winner's Bread** (2g T2) — fine in early game for tempo stats + win bonus.
### C (15)
- **Azerite Empowerment** (4g T6) — generally not worth buying (+4/+4 total for 4g).
- **Blood Gem Barrage** (1g T3) — not great. weak permanent shop buff; skip most lobbies.
- **Boon of Beetles** (1g T4) — bad unless playing Beasts.
- **Brood of Nozdormu** (2g T5) — not great start-of-combat double attack.
- **Cloning Conch** (4g T4) — really only okay with Brann, and even then not the best use of gold.
- **Eonar's Favor** (2g T4) — not great. shop-type buff this game is usually too slow/narrow.
- **Hallowed Ritual** (5g T7) — T7 — almost never seen; do not plan around it.
- **Menagerie Tableware** (4g T7) — T7 — almost never seen; do not plan around it.
- **Might of Stormwind** (2g T2) — not good. skip most of the time.
- **Misplaced Tea Set** (3g T4) — generally not worth buying.
- **Natural Blessing** (2g T4) — generally not worth buying.
- **Sacred Gift** (4g T7) — T7 — almost never seen; do not plan around it.
- **Sharing is Caring** (2g T7) — T7 — almost never seen; do not plan around it.
- **Time Management** (4g T3) — bad. expensive choose-one board buff; usually skip.
- **Tricky Trousers** (1g T3) — only used to give something Taunt (or strip it).
### F (8)
- **Healthy Bounty** (2g T3) — not worth buying (only Wealthy + Friendly bounties are real).
- **Hostile Bounty** (2g T3) — not worth buying (only Wealthy + Friendly bounties are real).
- **Perfect Vision** (2g T6) — sucks. skip the 20/20 set.
- **Robust Evolution** (1g T3) — terrible. do not buy — random higher-tier transform keeping stats is a brick.
- **Selfish Bounty** (2g T3) — not worth buying (only Wealthy + Friendly bounties are real).
- **Shiny Ring** (2g T3) — bad. skip.
- **Them Apples** (1g T1) — terrible. do not buy.
- **Wave of Gold** (2g T5) — sucks. skip.

## Spells excluded
- Bargain Bundle — excluded_duos
- Portal in a Crystal — excluded_duos
- Portal in a Fountain — excluded_duos
- Queen's Command — excluded_naga_out
- Shifting Tide — excluded_naga_out
- Spitescale Special — excluded_naga_out

## Comps
- [None] Dragons - APM Evoker (dragon)
- [None] Pirates - APM Golden (pirate)
- [None] Demons - APM Shop Buff (demon)
- [None] Undead - APM Undead (undead)
- [None] Undead - Attack Scaling (undead)
- [None] Dragons - Battlecries (dragon)
- [None] Beasts - Beetles (beast)
- [None] Quilboar - Bristlemane (quilboar)
- [None] Quilboar - Choose One (quilboar)
- [None] Mechs - Deathrattle (mech)
- [None] Nagas - End Of Turn/Spell Buff (naga)
- [None] Murlocs - Family (murloc)
- [None] Nagas - Groundbreaker (naga)
- [None] Beasts - Leviathan (beast)
- [None] Mechs - Magnetics (mech)
- [None] Mechs - Magnetics/Spells (mech)
- [None] Demons - Self Damage (demon)
- [None] Demons - Shop Buff (demon)
- [None] Elementals - Stat scaling (elemental)
- [None] Beasts - Summons (beast)
- [None] Beasts - Tasty Lobstah (beast)
- [None] Murlocs - Tidecaller (murloc)
- [None] Elementals - Unbound Tempest (elemental)
- [None] Murlocs - Venom Scam (murloc)

## Tribes
- Undead 
- Murloc 
- Demon 
- Mech 
- Elemental 
- Beast 
- Pirate 
- Dragon 
- Quilboar 
- Aberration 

## Hero exclusives present
- Morchie: sources 3, cards 89
- Murozond, Unbounded: sources 3, cards 109
- Jim Raynor: sources 3, cards 9
- E.T.C., Band Manager: sources 2, cards 114


## Gaps / still open
- 2 Naga comps still in comps.json (Naga out of lobby) — filter before train or weight 0
- hero first_place_rate / avg_placement / leveling_curve need authenticated heroes API or manual
- example_boards not on public comp_guides
- minion placement stats not pulled yet
- 54 hidden tier-list comps omitted (out of current meta)
- HSReplay UI still lists Naga comps/composition stats; minions snapshot excludes naga from live lobby pool
- Seafood Stew coach take is provisional — Aidan verify