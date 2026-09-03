#!/usr/bin/env node
/*
 * Firestone combat-sim sidecar.
 *
 * Reads a BgsBattleInfo JSON object from stdin, runs Firestone's
 * `@firestone-hs/simulate-bgs-battle`, and writes a compact result JSON to
 * stdout:  { "wonPercent": .., "tiedPercent": .., "lostPercent": ..,
 *            "averageDamageWon": .., "averageDamageLost": .. }
 *
 * The Python side (hsbg_coach/firestone_bridge.py) builds the input and parses
 * this output. If anything here fails (package not installed, card data not
 * loadable), it exits non-zero and Python falls back to its pure sim.
 *
 * TEMPLATE NOTE: the exact init of AllCardsService / CardsData and a few
 * BoardEntity field names can shift between package versions. Validate against
 * node_modules/@firestone-hs/simulate-bgs-battle/dist/*.d.ts after install.
 * `npm run check` (see package.json) prints the installed exports to help.
 */
'use strict';

async function main() {
  const input = await readStdin();
  if (!input.trim()) {
    fail('no input on stdin');
  }
  const battleInput = JSON.parse(input);

  let sim, refData;
  try {
    sim = require('@firestone-hs/simulate-bgs-battle');
    refData = require('@firestone-hs/reference-data');
  } catch (e) {
    fail('packages not installed — run `npm install` in bridge/: ' + e.message);
  }

  // Load card reference data. AllCardsService can initialise from the bundled
  // cards json; network fetch is avoided when a local cards file is present.
  const cards = new refData.AllCardsService();
  if (typeof cards.initializeCardsDb === 'function') {
    await cards.initializeCardsDb();
  }
  const cardsData = new sim.CardsData(cards, false);
  if (typeof cardsData.inititialize === 'function') {
    cardsData.inititialize(); // note: package historically spells it this way
  }

  // simulateBattle is a generator that yields progressively refined results;
  // the last yielded value is the final SimulationResult.
  const gen = sim.simulateBattle(battleInput, cards, cardsData);
  let result = null;
  for (const r of gen) {
    result = r;
  }
  if (!result) {
    fail('simulator returned no result');
  }

  process.stdout.write(JSON.stringify({
    wonPercent: result.wonPercent,
    tiedPercent: result.tiedPercent,
    lostPercent: result.lostPercent,
    averageDamageWon: result.averageDamageWon,
    averageDamageLost: result.averageDamageLost,
  }));
}

function readStdin() {
  return new Promise((resolve) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', (c) => (data += c));
    process.stdin.on('end', () => resolve(data));
  });
}

function fail(msg) {
  process.stderr.write('firestone_sim: ' + msg + '\n');
  process.exit(1);
}

main().catch((e) => fail(e && e.stack ? e.stack : String(e)));
