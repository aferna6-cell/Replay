# Train review — patch 36.6.1 (NOT trained yet)

> Synth comps rewritten v2 (playable archetypes + playbooks, no T7). Kept/dropped/thin150 unchanged.

> Synth comps rewritten v2 (playable archetypes + playbooks, no T7). Kept/dropped/thin150 unchanged.

## Rules
- Comps: all key/core/enabler/addon pieces must be in current in-pool minions
- Perfect boards: every card_id must be in current in-pool minions
- Synth: at least one comp per live tribe missing a kept snapshot comp

## Summary
- **kept_snapshot_comps**: 11
- **dropped_snapshot_comps**: 13
- **synth_comps**: 5
- **thin150_clean**: 72
- **thin150_dirty_excluded**: 78
- **tribes_covered**: ['aberration', 'beast', 'demon', 'dragon', 'elemental', 'mech', 'murloc', 'pirate', 'quilboar', 'undead']

## Dropped snapshot comps (missing pieces)
- `comp_73` **Dragons - APM Evoker** (dragon): missing `BG32_822, BG24_004, BG29_810`
- `comp_15` **Dragons - Battlecries** (dragon): missing `BG24_004, BG32_822, BG29_810`
- `comp_90` **Quilboar - Bristlemane** (quilboar): missing `BG36_510`
- `comp_88` **Quilboar - Choose One** (quilboar): missing `BG36_510`
- `comp_68` **Mechs - Deathrattle** (mech): missing `BGS_012`
- `comp_55` **Nagas - End Of Turn/Spell Buff** (naga): missing `BG32_837, BG36_622, BG32_835`
- `comp_93` **Murlocs - Family** (murloc): missing `BG35_142, BG35_141`
- `comp_20` **Nagas - Groundbreaker** (naga): missing `BG31_035, BG34_925, BG23_008, BG31_920`
- `comp_83` **Beasts - Leviathan** (beast): missing `BG23_008`
- `comp_2` **Mechs - Magnetics** (mech): missing `BG26_148, BG36_760, BG34_925`
- `comp_91` **Mechs - Magnetics/Spells** (mech): missing `BG36_760`
- `comp_37` **Elementals - Stat scaling** (elemental): missing `BG32_846, BG36_351, BG32_842, BG31_843, BGS_127`
- `comp_92` **Elementals - Unbound Tempest** (elemental): missing `BGS_104, BG32_846`

## Kept snapshot comps + synthesized boards
### comp_89 — Pirates - APM Golden (pirate, rank A)
- cores: Sky Admiral Rogers, Hooktusk, Master Marauder, Enterprising Escapee
- keys: Sky Admiral Rogers, Hooktusk, Master Marauder, Enterprising Escapee
- board:
  - Sky Admiral Rogers [BG33_823] 8/10 T6
  - Hooktusk, Master Marauder [BG36_344] 8/8 T6
  - Enterprising Escapee [BG36_523] 12/12 T5
  - Proud Privateer [BG33_825] 16/16 T5
  - Brann Bronzebeard [BG_LOE_077] 4/8 T5
  - Blade Collector [BG26_817] 6/4 T4

### comp_76 — Demons - APM Shop Buff (demon, rank A)
- cores: Sky Admiral Rogers, Proud Privateer, Felboar, Devilish Distractor, Balinda Stonehearth, Enterprising Escapee
- keys: Sky Admiral Rogers, Proud Privateer, Felboar, Devilish Distractor, Balinda Stonehearth, Enterprising Escapee
- board:
  - Sky Admiral Rogers [BG33_823] 8/10 T6
  - Proud Privateer [BG33_825] 16/16 T5
  - Felboar [BG28_633] 4/12 T5
  - Devilish Distractor [BG36_762] 8/14 T5
  - Balinda Stonehearth [BG35_883] 12/12 T6
  - Enterprising Escapee [BG36_523] 12/12 T5
  - Brann Bronzebeard [BG_LOE_077] 4/8 T5

### comp_3 — Undead - APM Undead (undead, rank A)
- cores: Handless Forsaken, Snazzy Phantom, Forsaken Weaver, Proud Privateer, Sky Admiral Rogers
- keys: Handless Forsaken, Snazzy Phantom, Forsaken Weaver, Proud Privateer, Sky Admiral Rogers
- board:
  - Handless Forsaken [BG25_010] 4/2 T3
  - Snazzy Phantom [BG36_515] 12/16 T6
  - Forsaken Weaver [BG34_692] 6/16 T6
  - Proud Privateer [BG33_825] 16/16 T5
  - Sky Admiral Rogers [BG33_823] 8/10 T6
  - Enterprising Escapee [BG36_523] 12/12 T5
  - Brann Bronzebeard [BG_LOE_077] 4/8 T5

### comp_14 — Undead - Attack Scaling (undead, rank S)
- cores: Mummifier, Dead Bellringer, Handless Forsaken, Snazzy Phantom, Drustfallen Butcher
- keys: Mummifier, Dead Bellringer, Handless Forsaken, Snazzy Phantom, Drustfallen Butcher
- board:
  - Mummifier [BG28_309] 10/4 T3
  - Dead Bellringer [BG36_511] 6/12 T4
  - Handless Forsaken [BG25_010] 4/2 T3
  - Snazzy Phantom [BG36_515] 12/16 T6
  - Drustfallen Butcher [BG32_324] 4/14 T5
  - Titus Rivendare [BG25_354] 2/14 T5
  - Eternal Summoner [BG25_009] 16/2 T6

### comp_38 — Beasts - Beetles (beast, rank S)
- cores: Turquoise Skitterer, Banana Slamma, Ravaging Scorpid, Headhunter Gryphon
- keys: Turquoise Skitterer, Banana Slamma, Ravaging Scorpid, Headhunter Gryphon
- board:
  - Turquoise Skitterer [BG31_809] 10/10 T5
  - Banana Slamma [BG26_802] 6/12 T4
  - Ravaging Scorpid [BG36_209] 12/14 T6
  - Headhunter Gryphon [BG36_204] 6/10 T4
  - Buzzing Vermin [BG31_803] 2/2 T1
  - Sewer Lord [BG35_604] 8/12 T5
  - Lurking Leviathan [BG35_602] 6/18 T5

### comp_13 — Demons - Self Damage (demon, rank S)
- cores: Brann Bronzebeard, Devilish Distractor, Balinda Stonehearth, Ashen Corruptor, Eredar Escapist
- keys: Brann Bronzebeard, Devilish Distractor, Balinda Stonehearth, Ashen Corruptor, Eredar Escapist
- board:
  - Brann Bronzebeard [BG_LOE_077] 4/8 T5
  - Devilish Distractor [BG36_762] 8/14 T5
  - Balinda Stonehearth [BG35_883] 12/12 T6
  - Ashen Corruptor [BG32_873] 12/12 T4
  - Eredar Escapist [BG36_733] 12/16 T6
  - Wrath Weaver [BGS_004] 2/6 T1
  - Malchezaar, Prince of Dance [BG26_524] 8/6 T3

### comp_41 — Demons - Shop Buff (demon, rank A)
- cores: Felboar, Imp-lusionist, Devilish Distractor, Balinda Stonehearth
- keys: Felboar, Imp-lusionist, Devilish Distractor, Balinda Stonehearth
- board:
  - Felboar [BG28_633] 4/12 T5
  - Imp-lusionist [BG36_731] 8/4 T4
  - Devilish Distractor [BG36_762] 8/14 T5
  - Balinda Stonehearth [BG35_883] 12/12 T6
  - Twisted Wrathguard [BG35_155] 16/16 T6
  - Ashen Corruptor [BG32_873] 12/12 T4
  - Insatiable Ur'zul [BG21_004] 8/12 T5

### comp_12 — Beasts - Summons (beast, rank A)
- cores: Goldrinn, the Great Wolf, Titus Rivendare, Headhunter Gryphon, Deathstrider, Banana Slamma, Sewer Lord
- keys: Goldrinn, the Great Wolf, Titus Rivendare, Headhunter Gryphon, Deathstrider, Banana Slamma, Sewer Lord
- board:
  - Goldrinn, the Great Wolf [BGS_018] 14/14 T5
  - Titus Rivendare [BG25_354] 2/14 T5
  - Headhunter Gryphon [BG36_204] 6/10 T4
  - Deathstrider [BG36_208] 20/22 T6
  - Banana Slamma [BG26_802] 6/12 T4
  - Sewer Lord [BG35_604] 8/12 T5
  - Lurking Lionfish [BG36_201] 6/8 T2

### comp_87 — Beasts - Tasty Lobstah (beast, rank A)
- cores: Deathstrider, Titus Rivendare, Tasty Lobster, Headhunter Gryphon
- keys: Deathstrider, Titus Rivendare, Tasty Lobster, Headhunter Gryphon
- board:
  - Deathstrider [BG36_208] 20/22 T6
  - Titus Rivendare [BG25_354] 2/14 T5
  - Tasty Lobster [BG36_202] 4/2 T3
  - Headhunter Gryphon [BG36_204] 6/10 T4
  - Lurking Lionfish [BG36_201] 6/8 T2
  - Hoarding Hyena [BG36_210] 8/12 T4
  - Sin'dorei Straight Shot [BG25_016] 6/8 T4

### comp_67 — Murlocs - Tidecaller (murloc, rank A)
- cores: Brann Bronzebeard, Diremuck Forager, Choral Mrrrglr, Shamanic Tidecaller, Balinda Stonehearth, Magicfin Mycologist
- keys: Brann Bronzebeard, Diremuck Forager, Choral Mrrrglr, Shamanic Tidecaller, Balinda Stonehearth, Magicfin Mycologist
- board:
  - Brann Bronzebeard [BG_LOE_077] 4/8 T5
  - Diremuck Forager [BG27_556] 8/10 T3
  - Choral Mrrrglr [BG26_354] 12/12 T6
  - Shamanic Tidecaller [BG36_704] 10/14 T5
  - Balinda Stonehearth [BG35_883] 12/12 T6
  - Magicfin Mycologist [BG33_891] 8/16 T6
  - Bile Spitter [BG33_318] 2/20 T5

### comp_8 — Murlocs - Venom Scam (murloc, rank A)
- cores: Leeroy the Reckless, Bile Spitter, Expert Aviator
- keys: Leeroy the Reckless, Bile Spitter, Expert Aviator
- board:
  - Leeroy the Reckless [BG23_318] 12/4 T5
  - Bile Spitter [BG33_318] 2/20 T5
  - Expert Aviator [BG34_140] 6/8 T2
  - Sky-hatch Runaway [BG36_243] 8/14 T4
  - Diremuck Forager [BG27_556] 8/10 T3
  - Deadly Spore [BGS_131] 2/2 T3
  - Heroic Underdog [BG34_604] 2/20 T4

## Synthesized comps (new) — v2 playable, no T7

Rewritten as real archetypes with playbooks (not highest-tier sorts). All cores/keys/boards are in-pool and Tier ≤6. Kept snapshot comps above are unchanged. **NOT trained.**

### synth_aberration_1 — Aberrations - Discard Engine (aberration, rank A)
- cores: Mindbender Ghur'sha, Dark Puppeteer, The Shadow of Doubt
- keys: Mindbender Ghur'sha, Dark Puppeteer, The Shadow of Doubt, Harbinger Aph'lass, Cutthroat K'Thir
- enablers: Parasitic Fleshling, N'raqi Frostcaller, Mindbending Recruiter, Abyssal Envoy
- addons: N'raqi Sapper, De-volition-ist, Mysterious K'Thir
- note: Real discard/Deity engine (not top-tier sort). Meta: early–mid launchpad; Blood Gems = discard fodder; no T7.
- playbook: Aberrations win early–mid by discarding fodder to pump your Deity and board, then often pivot rather than forcing a pure late Aberration endgame. Key enablers: Activate discard pieces (Zoatroid, Abyssal Envoy, Mindbending Recruiter) plus hand generators; Blood Gems are excellent discard fuel. Mindbender Ghur'sha is the board-wide payoff on every discard; Cutthroat K'Thir and Harbinger Aph'lass stack Deity/self stats; The Shadow of Doubt scales whenever cards enter hand. Turn plan: keep a discard Activate ready, generate cheap cards/spells, discard into Mindbender Ghur'sha/Cutthroat K'Thir/Harbinger Aph'lass, and improve spell power with Dark Puppeteer / N'raqi Frostcaller. Late: Parasitic Fleshling and Mysterious K'Thir convert total discards/spells into stats; De-volition-ist is the attack finisher. Do not plan around Tier-7.
- board:
  - Dark Puppeteer [BG36_104] 12/10 T6 (aberration)
  - The Shadow of Doubt [BG36_109] 10/14 T6 (aberration)
  - Harbinger Aph'lass [BGFYM_005] 8/14 T6 (aberration)
  - Mindbender Ghur'sha [BG36_097] 8/16 T5 (aberration)
  - Cutthroat K'Thir [BG36_106] 14/14 T4 (aberration)
  - Parasitic Fleshling [BG36_114] 10/12 T4 (aberration)
  - De-volition-ist [BG36_102] 12/14 T5 (aberration)

### synth_dragon_1 — Dragons - Rally Poet (dragon, rank A)
- cores: Crimson Vindicator, Persistent Poet, Kalecgos, Arcane Aspect
- keys: Crimson Vindicator, Persistent Poet, Kalecgos, Arcane Aspect, Amber Guardian, Draconic Warden
- enablers: Sky-hatch Runaway, Felfire Conjurer, Firescale Hoarder, Bronze Timewalker
- addons: Scarlet Survivor, Tarecgosa, Runic Arcanist, Brann Bronzebeard
- note: Non-Evoker rebuild: Permanent Poet + Rally Chromadrake from in-pool pieces. Evoker keys out of pool. No T7.
- playbook: Classic Dragons APM Evoker is dead this lobby (Fire-Forged Evoker / key discover pieces out of pool) — play Rally + permanent-keyword dragons instead. Park Persistent Poet beside your carry (Crimson Vindicator or Scarlet Survivor/Tarecgosa) so combat keywords and stats stick forever; Amber Guardian attaches permanent Divine Shield through Poet. Hunt Chromadrake generators (Draconic Warden, Bronze Timewalker, Hired Mount) and Sky-hatch Runaway to force Rally triggers on demand; Crimson Vindicator's Rally casts Mighty Dragonbreath for the combat spike. Kalecgos, Arcane Aspect (+ Brann Bronzebeard if available) rewards battlecry cycling (Draconic Warden, Tavern Tempest, Firescale Hoarder). Felfire Conjurer / Firescale Hoarder drip permanent tavern-spell buffs. Do not draft around Tier-7 Obsidian Ravager.
- board:
  - Crimson Vindicator [BG36_241] 14/16 T6 (dragon)
  - Persistent Poet [BG29_813] 6/8 T4 (dragon)
  - Kalecgos, Arcane Aspect [BGS_041] 10/20 T5 (dragon)
  - Amber Guardian [BG24_500] 8/10 T3 (dragon)
  - Draconic Warden [BG34_633] 12/10 T5 (dragon)
  - Sky-hatch Runaway [BG36_243] 10/12 T4 (dragon)
  - Felfire Conjurer [BG32_821] 10/10 T5 (dragon,demon)

### synth_elemental_1 — Elementals - Unbound Tempest (elemental, rank A)
- cores: Unbound Tempest, Air Revenant, Tavern Tempest
- keys: Unbound Tempest, Air Revenant, Tavern Tempest, Brann Bronzebeard, Kelp Keeper
- enablers: Flourishing Frostling, Flaming Enforcer, Refreshing Anomaly, En-Djinn Blazer
- addons: Elemental of Surprise, Air Baller, Living Prison
- note: Classic Unbound Tempest shop-scaling without Nomi/out-of-pool keys; Air Revenant is the shop buffer. No T7.
- playbook: Elementals Unbound Tempest is still the scaling plan: buff the shop, then feed Unbound Tempest the highest-Health tavern minion every three Elementals played. Nomi is out of pool — Air Revenant (Easterly Winds after gold) and En-Djinn Blazer / Waveling refresh buffs are the shop enablers. Economy engine: Brann Bronzebeard + Tavern Tempest + Kelp Keeper to chain Elemental battlecries and stay infinite; cycle Sellemental, Refreshing Anomaly, and Ballers. Flaming Enforcer stabilizes mid by consuming tavern Health; Flourishing Frostling is a passive payoff for Elementals played this game; Elemental of Surprise helps triple awkwardly. Do not plan around Tier-7 Stone Age Slab.
- board:
  - Unbound Tempest [BG36_352] 18/28 T6 (elemental)
  - Air Revenant [BG34_858] 8/14 T5 (elemental)
  - Flourishing Frostling [BG26_537] 16/10 T5 (elemental)
  - Flaming Enforcer [BG34_500] 14/16 T4 (demon,elemental)
  - Tavern Tempest [BGS_123] 6/6 T4 (elemental)
  - Brann Bronzebeard [BG_LOE_077] 4/8 T5 (neutral)
  - Kelp Keeper [BG36_701] 6/10 T4 (murloc)

### synth_mech_1 — Mechs - Glambot Magnetics (mech, rank A)
- cores: Glambot, Utility Drone, Spark Snapper
- keys: Glambot, Utility Drone, Spark Snapper, Balinda Stonehearth, Drone Duplicator
- enablers: Charging Czarina, Falling Sky Golem, Gearfin, Enchanted Sentinel
- addons: Accord-o-Tron, Annoy-o-Module, Titus Rivendare
- note: Rebuild of Magnetics/Spells without Scrap Scraper / Beatboxer; Glambot+Snapper+Utility Drone+Balinda. No T7.
- playbook: Mechs Magnetics/Spells: while Glambot is on board, every spell cast on a Mech Magnetizes a Satellite; Balinda Stonehearth doubles targeted friendly spells so you attach twice as often. Spark Snapper improves Satellite size whenever you play Mechs (Scrap Scraper is out — Snapper is the generation core). Drone Duplicator Activate makes the next Magnetize happen an extra time. Endgame Utility Drone pays +4/+5 per Magnetization at end of turn — that is the win condition. Charging Czarina spikes Divine Shield mechs when you cast spells. Turn plan: buy 1–2 cost targeted spells / Fearless Foodie / Gearfin spell gen, cast onto mechs, Activate Duplicator on the best Magnetize, then land Utility Drone. Optional deathrattle line: Falling Sky Golem + Titus Rivendare if magnetics are slow. No Tier-7 Polarizing Beatboxer.
- board:
  - Utility Drone [BG26_152] 10/14 T6 (mech)
  - Glambot [BG36_853] 8/10 T4 (mech)
  - Spark Snapper [BG36_851] 12/12 T5 (mech)
  - Drone Duplicator [BG36_506] 8/8 T4 (mech)
  - Balinda Stonehearth [BG35_883] 12/12 T6 (neutral)
  - Charging Czarina [BG28_741] 10/6 T5 (mech)
  - Falling Sky Golem [BG35_342] 20/14 T6 (mech)

### synth_quilboar_1 — Quilboar - Choose One Gems (quilboar, rank A)
- cores: Turbo Hogrider, Gem Rat, Sanguine Champion
- keys: Turbo Hogrider, Gem Rat, Sanguine Champion, Sanguine Refiner, Veteran Brigand
- enablers: Fearless Foodie, Bramble Tunneler, Thorned Trailblazer, Trench Fighter
- addons: Bonker, Felboar, Geomagus Roogug
- note: Choose One quilboar without Vigilant Bristlemane / Jailbird T7; Turbo Hogrider + gem quality + Choose One generation.
- playbook: Quilboar Choose One: scale Blood Gem quality first (Gem Rat's Gem Day, Fearless Foodie +1/+1 choose, Sanguine Refiner Rally, Sanguine Champion battlecry/deathrattle), then pay off with Turbo Hogrider. Every Choose One you play makes Turbo Hogrider blood-gem all other Quilboar — generate Choose Ones via Bramble Tunneler, Gem Rat, Snare Trapper, Crater Miner; Thorned Trailblazer combines both Choose effects. Veteran Brigand is the late Choose One bomb (gems everyone or triple Barrage). Trench Fighter's Gem Confiscation concentrates gems onto one unit when you need a carry. Meta: Quilboar are often support (Blood Gems for spell-on-minion comps or Aberration discard fodder) — force pure Quilboar when the lobby is soft or you hit Hogrider + gem quality early. No Tier-7 Jailbird Juggernaut; Vigilant Bristlemane is out of pool.
- board:
  - Turbo Hogrider [BG31_323] 12/16 T6 (quilboar)
  - Veteran Brigand [BG36_341] 14/14 T6 (quilboar)
  - Sanguine Champion [BG23_017] 14/10 T6 (quilboar)
  - Sanguine Refiner [BG33_885] 8/14 T5 (quilboar)
  - Gem Rat [BG31_326] 6/8 T3 (quilboar)
  - Fearless Foodie [BG30_123] 8/10 T3 (quilboar)
  - Bonker [BG20_104] 10/12 T4 (quilboar)
## Perfect thin150 boards kept (72 / 150)
### thin#2 comp=47 placement=1.0
  - Cord Puller [BG29_611] 7/10
  - Suspicious Prisonguard [BG36_345] 3/3

### thin#6 comp=47 placement=1.0
  - Harmless Bonehead [BG28_300] 1/1
  - Crackling Cyclone [BGS_119] 2/1
  - Southsea Busker [BG26_135] 4/4
  - Drustfallen Butcher [BG32_324] 2/7

### thin#8 comp=9 placement=1.0
  - Eredar Escapist [BG36_733] 2451/2476
  - Eredar Escapist [BG36_733] 10333/10373
  - Devilish Distractor [BG36_762] 7380/7426
  - Ashen Corruptor [BG32_873] 3844/3875
  - Wrath Weaver [BGS_004] 2661/2671
  - Balinda Stonehearth [BG35_883] 6/6

### thin#9 comp=36 placement=1.0
  - Sky Admiral Rogers [BG33_823] 311/301
  - Brann Bronzebeard [BG_LOE_077] 3/6
  - Enterprising Escapee [BG36_523] 6/6
  - Hooktusk, Master Marauder [BG36_344] 387/377
  - Hooktusk, Master Marauder [BG36_344] 289/279
  - Hooktusk, Master Marauder [BG36_344] 281/279
  - Proud Privateer [BG33_825] 291/280

### thin#12 comp=8 placement=1.0
  - Brann Bronzebeard [BG_LOE_077] 14/20
  - Utility Drone [BG26_152] 17/27
  - Falling Sky Golem [BG35_342] 2912/3356
  - Glambot [BG36_853] 946/1125
  - Polarizing Beatboxer [BG26_149] 3325/3974
  - Utility Drone [BG26_152] 6/7
  - Balinda Stonehearth [BG35_883] 21/26

### thin#14 comp=20 placement=1.0
  - Scarlet Survivor [BG35_814] 10/12
  - Suspicious Prisonguard [BG36_345] 3/3

### thin#16 comp=20 placement=1.0
  - Crimson Vindicator [BG36_241] 21/16
  - Bronze Timewalker [BG36_242] 164/166
  - Persistent Poet [BG29_813] 65/66
  - Persistent Poet [BG29_813] 2/3
  - Kalecgos, Arcane Aspect [BGS_041] 65/77
  - Brann Bronzebeard [BG_LOE_077] 7/9
  - Amber Guardian [BG24_500] 30/37

### thin#22 comp=47 placement=1.0
  - Mummifier [BG28_309] 21/18
  - Sewer Lord [BG35_604] 4/6
  - Ashen Corruptor [BG32_873] 55/59
  - Malchezaar, Prince of Dance [BG26_524] 17/109
  - Snazzy Phantom [BG36_515] 10/16
  - Drustfallen Butcher [BG32_324] 2/7

### thin#24 comp=36 placement=1.0
  - Blade Collector [BG26_817] 974/944
  - Clever Castaway [BG36_342] 956/939
  - Clever Castaway [BG36_342] 1334/1316
  - Enterprising Escapee [BG36_523] 68/68
  - Hooktusk, Master Marauder [BG36_344] 501/489
  - Hooktusk, Master Marauder [BG36_344] 469/495

### thin#26 comp=3 placement=1.0
  - Wolf Pup [BG36_207] 3/9
  - Lurking Lionfish [BG36_201] 3/4

### thin#27 comp=41 placement=1.0
  - Glim Guardian [BG29_888] 1/4
  - Glim Guardian [BG29_888] 1/4

### thin#28 comp=8 placement=1.0
  - Cord Puller [BG29_611] 8/6
  - Glambot [BG36_853] 149/115
  - Drone Duplicator [BG36_506] 8/3
  - Fruit Vendor [BG36_346] 3/6
  - Glambot [BG36_853] 39/33
  - Brann Bronzebeard [BG_LOE_077] 2/4
  - Rescue Bot [BG36_854] 5/2

### thin#31 comp=41 placement=1.0
  - Scarlet Survivor [BG35_814] 11/11
  - Suspicious Prisonguard [BG36_345] 3/3

### thin#37 comp=20 placement=1.0
  - Suspicious Prisonguard [BG36_345] 6/6
  - Suspicious Prisonguard [BG36_345] 6/6

### thin#39 comp=36 placement=1.0
  - Leeroy the Reckless [BG23_318] 6/2
  - Blade Collector [BG26_817] 1198/1196
  - Enterprising Escapee [BG36_523] 2616/2620
  - Enterprising Escapee [BG36_523] 2422/2419
  - Enterprising Escapee [BG36_523] 2417/2414
  - Hooktusk, Master Marauder [BG36_344] 98/91
  - Brann Bronzebeard [BG_LOE_077] 2650/2647

### thin#40 comp=9 placement=1.0
  - Wrath Weaver [BGS_004] 4/8
  - Wrath Weaver [BGS_004] 1/3

### thin#42 comp=41 placement=1.0
  - Razorfen Geomancer [BG20_100] 6/5
  - Razorfen Geomancer [BG20_100] 2/1
  - Buzzing Vermin [BG31_803] 1/1

### thin#43 comp=9 placement=1.0
  - Twisted Wrathguard [BG35_155] 217/202
  - Ashen Corruptor [BG32_873] 995/994
  - Imposing Percussionist [BG26_525] 215/209
  - Kelp Keeper [BG36_701] 26/16
  - Enterprising Escapee [BG36_523] 35/31
  - Malchezaar, Prince of Dance [BG26_524] 4/3
  - Brann Bronzebeard [BG_LOE_077] 9/7

### thin#46 comp=41 placement=1.0
  - Bonker [BG20_104] 131/149
  - Bramble Tunneler [BG36_331] 149/157
  - Fearless Foodie [BG30_123] 86/100
  - Veteran Brigand [BG36_341] 50/51
  - Veteran Brigand [BG36_341] 80/86
  - Turbo Hogrider [BG31_323] 55/58
  - Gem Rat [BG31_326] 100/110

### thin#50 comp=20 placement=1.0
  - Scarlet Survivor [BG35_814] 8/8
  - Gearfin [BG36_764] 6/5

### thin#51 comp=41 placement=1.0
  - Bonker [BG20_104] 215/242
  - Bramble Tunneler [BG36_331] 266/286
  - Veteran Brigand [BG36_341] 134/144
  - Veteran Brigand [BG36_341] 164/179
  - Turbo Hogrider [BG31_323] 65/75
  - Fearless Foodie [BG30_123] 43/49
  - Gem Rat [BG31_326] 184/203

### thin#52 comp=9 placement=1.0
  - Imp-lusionist [BG36_731] 175/173
  - Imp-lusionist [BG36_731] 585/585
  - Malchezaar, Prince of Dance [BG26_524] 1688/1699
  - Devilish Distractor [BG36_762] 7317/7293
  - Ashen Corruptor [BG32_873] 3237/3231
  - Wrath Weaver [BGS_004] 580/572
  - Eredar Escapist [BG36_733] 1703/1702

### thin#53 comp=47 placement=1.0
  - Balinda Stonehearth [BG35_883] 18/10
  - Forsaken Weaver [BG34_692] 7/12
  - Dead Bellringer [BG36_511] 27/30

### thin#55 comp=41 placement=1.0
  - Sanguine Refiner [BG33_885] 63/37
  - Bonker [BG20_104] 54/53
  - Bonker [BG20_104] 47/43
  - Razorfen Vineweaver [BG33_883] 44/43
  - Falling Sky Golem [BG35_342] 98/62
  - Turbo Hogrider [BG31_323] 31/33
  - Balinda Stonehearth [BG35_883] 32/32

### thin#57 comp=9 placement=1.0
  - Imp-lusionist [BG36_731] 26/24
  - Imp-lusionist [BG36_731] 25/23
  - Tichondrius [BG26_523] 100/100
  - Devilish Distractor [BG36_762] 185/190
  - Twisted Wrathguard [BG35_155] 24/24
  - Balinda Stonehearth [BG35_883] 30/30
  - Ashen Corruptor [BG32_873] 223/213

### thin#58 comp=3 placement=1.0
  - Hoarding Hyena [BG36_210] 834/428
  - Headhunter Gryphon [BG36_204] 439/233
  - Tasty Lobster [BG36_202] 814/411
  - Lurking Lionfish [BG36_201] 7/8
  - Deathstrider [BG36_208] 861/445
  - Deathstrider [BG36_208] 859/438
  - Titus Rivendare [BG25_354] 8/20

### thin#60 comp=3 placement=1.0
  - Headhunter Gryphon [BG36_204] 50/42
  - Tasty Lobster [BG36_202] 6/3
  - Forest Rover [BG31_801] 1/1
  - Deathstrider [BG36_208] 14/13
  - Deathstrider [BG36_208] 10/11
  - Cage Gnawer [BG36_211] 18/17
  - Crackling Cyclone [BGS_119] 14/17

### thin#61 comp=9 placement=1.0
  - Balinda Stonehearth [BG35_883] 6/6
  - Imp-lusionist [BG36_731] 283/273
  - Felboar [BG28_633] 1091/1097
  - Devilish Distractor [BG36_762] 1746/1745
  - Ashen Corruptor [BG32_873] 381/192
  - Twisted Wrathguard [BG35_155] 181/185
  - Malchezaar, Prince of Dance [BG26_524] 85/84

### thin#62 comp=8 placement=1.0
  - Glambot [BG36_853] 1756/1986
  - Glambot [BG36_853] 646/732
  - Drone Duplicator [BG36_506] 4991/5844
  - Balinda Stonehearth [BG35_883] 30/30
  - Brann Bronzebeard [BG_LOE_077] 2/4
  - Utility Drone [BG26_152] 4/5
  - Utility Drone [BG26_152] 1330/1574

### thin#64 comp=20 placement=1.0
  - Bronze Timewalker [BG36_242] 8/9
  - Persistent Poet [BG29_813] 23/7
  - Persistent Poet [BG29_813] 6/7
  - Sky-hatch Runaway [BG36_243] 13/11
  - Sky-hatch Runaway [BG36_243] 8/11
  - Scarlet Survivor [BG35_814] 26/17
  - Amber Guardian [BG24_500] 3/2

### thin#65 comp=41 placement=1.0
  - Bramble Tunneler [BG36_331] 2777/2363
  - Bramble Tunneler [BG36_331] 6394/5438
  - Gem Rat [BG31_326] 1651/1403
  - Felboar [BG28_633] 13482/11773
  - Turbo Hogrider [BG31_323] 1347/1141
  - Turbo Hogrider [BG31_323] 1029/881
  - Drakkari Enchanter [BG26_ICC_901] 499/433

### thin#67 comp=47 placement=1.0
  - Dead Bellringer [BG36_511] 11/14
  - Eternal Summoner [BG25_009] 8/1
  - Eternal Knight [BG25_008] 28/23
  - Titus Rivendare [BG25_354] 1/7
  - Titus Rivendare [BG25_354] 1/7
  - Snazzy Phantom [BG36_515] 188/192

### thin#68 comp=9 placement=1.0
  - Wrath Weaver [BGS_004] 13/19
  - Soul Rewinder [BG26_174] 4/10
  - Malchezaar, Prince of Dance [BG26_524] 4/3

### thin#69 comp=9 placement=1.0
  - Wrath Weaver [BGS_004] 6/8
  - Ominous Seer [BG31_330] 2/1

### thin#76 comp=41 placement=1.0
  - Bonker [BG20_104] 2/17
  - Scarlet Survivor [BG35_814] 16/15
  - Geomagus Roogug [BG28_583] 19/15
  - Sly Infiltrator [BG36_330] 8/7
  - Sly Infiltrator [BG36_330] 7/7
  - Gem Rat [BG31_326] 7/6
  - Thorned Trailblazer [BG31_327] 7/7

### thin#79 comp=47 placement=1.0
  - Titus Rivendare [BG25_354] 4/10
  - Drustfallen Butcher [BG32_324] 1357/1367
  - Snazzy Phantom [BG36_515] 18/22

### thin#80 comp=47 placement=1.0
  - Eternal Knight [BG25_008] 6/4
  - Harmless Bonehead [BG28_300] 1/1

### thin#82 comp=9 placement=1.0
  - Scarlet Survivor [BG35_814] 6/6
  - Suspicious Prisonguard [BG36_345] 3/3

### thin#83 comp=47 placement=1.0
  - Plaguerunner [BG34_690] 8/4
  - Eternal Summoner [BG25_009] 8/1
  - Eternal Knight [BG25_008] 24/19
  - Maw Caster [BG32_340] 4/5
  - Snazzy Phantom [BG36_515] 6/8

### thin#88 comp=36 placement=1.0
  - Scarlet Survivor [BG35_814] 6/4
  - Suspicious Prisonguard [BG36_345] 3/3
  - Clever Castaway [BG36_342] 2/3

### thin#92 comp=47 placement=1.0
  - Titus Rivendare [BG25_354] 1/7
  - Snazzy Phantom [BG36_515] 10/16
  - Drustfallen Butcher [BG32_324] 14/19
  - Drustfallen Butcher [BG32_324] 376/381

### thin#93 comp=47 placement=1.0
  - Mummifier [BG28_309] 10/3
  - Handless Forsaken [BG25_010] 2/1
  - Eternal Knight [BG25_008] 18/19
  - Handless Forsaken [BG25_010] 4/3
  - Cadaver Caretaker [BG30_125] 6/6
  - Snazzy Phantom [BG36_515] 6/8
  - Drustfallen Butcher [BG32_324] 2/7

### thin#94 comp=20 placement=1.0
  - Scarlet Survivor [BG35_814] 3/3
  - Scarlet Survivor [BG35_814] 3/3

### thin#95 comp=3 placement=1.0
  - Bile Spitter [BG33_318] 21/30
  - Headhunter Gryphon [BG36_204] 13322/6695
  - Hoarding Hyena [BG36_210] 14483/7258
  - Tasty Lobster [BG36_202] 6498/3254
  - Deathstrider [BG36_208] 12021/6028
  - Deathstrider [BG36_208] 17375/8699
  - Titus Rivendare [BG25_354] 18/30

### thin#98 comp=3 placement=1.0
  - Leeroy the Reckless [BG23_318] 6/2
  - Cataclysmic Harbinger [BG35_123] 13/21
  - Leeroy the Reckless [BG23_318] 13/9
  - Sewer Lord [BG35_604] 70/6
  - Sewer Lord [BG35_604] 32/7
  - Lurking Leviathan [BG35_602] 59/20

### thin#99 comp=8 placement=1.0
  - Trench Fighter [BG34_684] 4/4
  - Decoy Conjurer [BG36_354] 8/11
  - Glambot [BG36_853] 18/18
  - Geomagus Roogug [BG28_583] 4/6
  - Fire Baller [BG31_816] 4/3
  - Snow Baller [BG31_818] 3/4
  - Maritime Extortionist [BG36_524] 7/7

### thin#101 comp=8 placement=1.0
  - Glambot [BG36_853] 302/362
  - Utility Drone [BG26_152] 7136/8818
  - Utility Drone [BG26_152] 320/394
  - Balinda Stonehearth [BG35_883] 12/12
  - Drakkari Enchanter [BG26_ICC_901] 2/10
  - Elemental of Surprise [BG26_175] 8/8

### thin#102 comp=47 placement=1.0
  - Clever Castaway [BG36_342] 4/6
  - Risen Rider [BG25_001] 2/1

### thin#103 comp=47 placement=1.0
  - Risen Rider [BG25_001] 8/7
  - Snazzy Phantom [BG36_515] 6/8
  - Snazzy Phantom [BG36_515] 12/16
  - Snazzy Phantom [BG36_515] 4552/4556
  - Drustfallen Butcher [BG32_324] 7429/7439

### thin#106 comp=20 placement=1.0
  - Bronze Timewalker [BG36_242] 42/33
  - Scarlet Survivor [BG35_814] 39/37
  - Gearfin [BG36_764] 6/5
  - Fearless Foodie [BG30_123] 2/4
  - Sly Infiltrator [BG36_330] 4/5
  - Amber Guardian [BG24_500] 18/11

### thin#109 comp=39 placement=1.0
  - Air Revenant [BG34_858] 375/145
  - Tavern Tempest [BGS_123] 300/100
  - Unbound Tempest [BG36_352] 1433/1179
  - Unbound Tempest [BG36_352] 4928/3788
  - Unbound Tempest [BG36_352] 4930/3791
  - Brann Bronzebeard [BG_LOE_077] 50/21

### thin#110 comp=2 placement=1.0
  - Scarlet Survivor [BG35_814] 6/6
  - Suspicious Prisonguard [BG36_345] 3/3
  - Ominous Seer [BG31_330] 2/1

### thin#112 comp=36 placement=1.0
  - Blade Collector [BG26_817] 2994/3005
  - Enterprising Escapee [BG36_523] 3255/3261
  - Enterprising Escapee [BG36_523] 4518/4525
  - Hooktusk, Master Marauder [BG36_344] 1322/1326
  - Sky Admiral Rogers [BG33_823] 1380/1385
  - Hooktusk, Master Marauder [BG36_344] 1377/1371
  - Leeroy the Reckless [BG23_318] 6/2

### thin#114 comp=39 placement=1.0
  - Crackling Cyclone [BGS_119] 66/65
  - Unbound Tempest [BG36_352] 305/317
  - Flourishing Frostling [BG26_537] 60/30
  - Air Revenant [BG34_858] 10/14
  - Kelp Keeper [BG36_701] 12/14
  - Tavern Tempest [BGS_123] 11/13
  - Brann Bronzebeard [BG_LOE_077] 9/13

### thin#116 comp=3 placement=1.0
  - Wolf Pup [BG36_207] 3/6
  - Headhunter Gryphon [BG36_204] 4/6
  - Hoarding Hyena [BG36_210] 4/6
  - Forest Rover [BG31_801] 12/12
  - Forest Rover [BG31_801] 1/1
  - Cage Gnawer [BG36_211] 2/7
  - Banana Slamma [BG26_802] 6/9

### thin#117 comp=47 placement=1.0
  - Barrier Banshee [BG36_514] 245/245
  - Cadaver Caretaker [BG30_125] 3/3
  - Harmless Bonehead [BG28_300] 1/1
  - Snazzy Phantom [BG36_515] 2620/2505
  - Drustfallen Butcher [BG32_324] 5961/5971

### thin#119 comp=3 placement=1.0
  - Headhunter Gryphon [BG36_204] 109/124
  - Lurking Lionfish [BG36_201] 16/13
  - Lurking Lionfish [BG36_201] 19/18
  - Lurking Lionfish [BG36_201] 3/4
  - Ravaging Scorpid [BG36_209] 22/21
  - Ravaging Scorpid [BG36_209] 31/18
  - Banana Slamma [BG26_802] 6/18

### thin#121 comp=9 placement=1.0
  - Wrath Weaver [BGS_004] 10/12
  - Soul Rewinder [BG26_174] 5/7
  - Wrath Weaver [BGS_004] 3/5
  - Ominous Seer [BG31_330] 2/1

### thin#122 comp=47 placement=1.0
  - Titus Rivendare [BG25_354] 1/7
  - Handless Forsaken [BG25_010] 2/1
  - Snazzy Phantom [BG36_515] 1380/1384
  - Drustfallen Butcher [BG32_324] 1586/1555

### thin#123 comp=47 placement=1.0
  - Titus Rivendare [BG25_354] 1/7
  - Snazzy Phantom [BG36_515] 10/16
  - Drustfallen Butcher [BG32_324] 1756/1761
  - Drustfallen Butcher [BG32_324] 14/19

### thin#124 comp=47 placement=1.0
  - Handless Forsaken [BG25_010] 2/1
  - Snazzy Phantom [BG36_515] 974/980
  - Drustfallen Butcher [BG32_324] 1756/1761
  - Drustfallen Butcher [BG32_324] 2530/2535

### thin#125 comp=47 placement=1.0
  - Cord Puller [BG29_611] 7/10
  - Suspicious Prisonguard [BG36_345] 3/3

### thin#127 comp=9 placement=1.0
  - Highkeeper Ra [BG34_319] 14/18
  - Champion of Sargeras [BG27_016] 8/8
  - Soul Rewinder [BG26_174] 24/42
  - Malchezaar, Prince of Dance [BG26_524] 15/18
  - Devilish Distractor [BG36_762] 12/19
  - Brann Bronzebeard [BG_LOE_077] 4/8
  - Twisted Wrathguard [BG35_155] 8/8

### thin#129 comp=20 placement=1.0
  - Suspicious Prisonguard [BG36_345] 6/6
  - Suspicious Prisonguard [BG36_345] 3/3

### thin#130 comp=36 placement=1.0
  - Crimson Vindicator [BG36_241] 26/15
  - Brann Bronzebeard [BG_LOE_077] 32/14
  - Locked-up Mutineer [BG36_521] 101/78
  - Enterprising Escapee [BG36_523] 96/70
  - Clever Castaway [BG36_342] 104/86
  - Sky Admiral Rogers [BG33_823] 47/28
  - Hooktusk, Master Marauder [BG36_344] 34/14

### thin#132 comp=36 placement=1.0
  - Bigwig Bandit [BG33_822] 17/21
  - Bigwig Bandit [BG33_822] 131/135
  - Enterprising Escapee [BG36_523] 39/39
  - Proud Privateer [BG33_825] 37/33
  - Enterprising Escapee [BG36_523] 39/31
  - Hooktusk, Master Marauder [BG36_344] 12/8

### thin#134 comp=3 placement=1.0
  - Wolf Pup [BG36_207] 3/6
  - Headhunter Gryphon [BG36_204] 18/16
  - Tasty Lobster [BG36_202] 2/1
  - Sky-hatch Runaway [BG36_243] 4/7
  - Buzzing Vermin [BG31_803] 1/1
  - Sprightly Scarab [BG27_084] 3/1
  - Forest Rover [BG31_801] 1/1

### thin#138 comp=2 placement=1.0
  - Expert Aviator [BG34_140] 5/6
  - Tad [BG22_202] 2/2
  - Tad [BG22_202] 2/2

### thin#139 comp=20 placement=1.0
  - Blade Collector [BG26_817] 41/40
  - Sky-hatch Runaway [BG36_243] 6/9
  - Persistent Poet [BG29_813] 2/3
  - Persistent Poet [BG29_813] 2/3
  - Runic Arcanist [BG36_245] 4/6
  - Azsharan Cutlassier [BG33_830] 6/4
  - Amber Guardian [BG24_500] 3/2

### thin#144 comp=47 placement=1.0
  - Eternal Knight [BG25_008] 4/2
  - Roadboar [BG20_101] 5/7
  - Razorfen Geomancer [BG20_100] 4/3
  - Prodigious Tusker [BG33_430] 2/5
  - Amber Guardian [BG24_500] 3/2

### thin#145 comp=3 placement=1.0
  - Headhunter Gryphon [BG36_204] 3684/1888
  - Wolf Pup [BG36_207] 3/6
  - Hoarding Hyena [BG36_210] 5339/2681
  - Tasty Lobster [BG36_202] 3972/2013
  - Deathstrider [BG36_208] 4216/2120
  - Deathstrider [BG36_208] 5201/2613
  - Titus Rivendare [BG25_354] 1/7

### thin#147 comp=47 placement=1.0
  - Scarlet Skull [BG25_022] 2/1
  - Eternal Knight [BG25_008] 8/6
  - Harmless Bonehead [BG28_300] 1/1
