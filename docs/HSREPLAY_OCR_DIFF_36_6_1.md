# HSReplay OCR diff — patch 36.6.1

**When:** 2026-09-22 ~8:15 PM ET  
**Screenshots:** 29 (`/workspace/screenshots/shot-call_*.png`) from https://hsreplay.net/battlegrounds/minions/  
**OCR:** tesseract `eng` (psm 6/11 + contrast preprocess), then fuzzy-match to `data/cards/bg_live_pool_36_6_1.json`.

## Counts

| Metric | Count |
|---|---|
| Screenshots OCRed | 29 |
| Live JSON pool (`bg_live_pool_36_6_1`) | 252 |
| OCR∪vision names matched into live pool | 44 |
| Pure tesseract→pool hits | 10 |
| Prior `key_minions` + aberration soft keys | 70 |
| Prior keys **not** in OCR∪JSON pool | 0 |
| Screenshot-derived names **not** in prior keys | 33 |
| Prior keys scrubbed (not in live JSON pool) | 0 |

## Flags

### Prior key_minions NOT in OCR∪JSON pool

_None._ All prior keys ⊆ live JSON pool (OCR∪JSON).

### OCR / screenshot names that look real but missing from prior `key_minions`

These are in the live shop pool (or confidently read from screenshots and confirmed in-pool) but are **not** currently listed as prior key minions. Informational only — not auto-added.

<details><summary>33 names</summary>

```
Air Revenant
Ancestral Automaton
Aureate Laureate
Barrier Banshee
Buzzing Vermin
Cadaver Caretaker
Cataclysmic Harbinger
Champion of Sargeras
Charging Czarina
Cord Puller
Costume Enthusiast
Crackling Cyclone
Crater Miner
Decoy Conjurer
Deft Deserter
Electric Synthesizer
Eternal Knight
Fire Baller
Flighty Scout
Flittering Bat
Forest Rover
Glim Guardian
Harmless Bonehead
Holy Vanguard
Humming Bird
Humon'gozz
Imposing Percussionist
Intrepid Botanist
Living Prison
Lovesick Balladist
Maritime Extortionist
Stone Age Slab
The Last One Standing
```

</details>

## Scrub of `playstyle_prior.json`

**No changes.** `tests/test_playstyle_prior_live_pool.py` already passes (5/5): every `key_minion` and `aberration_soft_key_minions` entry is in the live non-Naga pool. No comps removed.

Removals this pass: _(none)_

## Artifacts

- `data/cards/hsreplay_ocr_minions.txt`
- `data/cards/hsreplay_ocr_minions.json`
- Live pool authority remains `data/cards/bg_live_pool_36_6_1.json` + `bg_cards.json` (HSJSON, Naga hard-excluded).

## Notes

Tesseract alone recovers card-name banners poorly (ornate gold text on art). Matching uses normalized substring + unambiguous fuzzy against the 252-name live pool, plus a small vision assist list from the same screenshots (rejected if not in-pool).
