/* Capture reference games using the pinned, already-built local simulator.
 * Usage: node tools/record_showdown.cjs /path/to/pokemon-showdown [output-dir]
 * No network server or dependency installation is required.
 */
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(process.argv[2]);
const out = path.resolve(process.argv[3] || 'unit_tests/fixtures/showdown');
const {Battle, Teams, PRNG, Dex} = require(path.join(root, 'dist/sim'));
const revision = process.env.SHOWDOWN_REVISION;
if (!revision) throw new Error('Set SHOWDOWN_REVISION to the checkout commit');
const format = 'gen9randombattle';
fs.mkdirSync(out, {recursive: true});
function snapshot(battle) {
  return battle.sides.map(side => ({active: side.active.map(p => p && ({
    name: p.name, species: p.species.name, types: p.getTypes(), hp: p.hp, maxhp: p.maxhp,
    ability: p.ability, item: p.item, status: p.status, boosts: {...p.boosts},
    pp: p.moveSlots.map(m => ({id: m.id, pp: m.pp})),
  })), hp: side.pokemon.map(p => p.hp)}));
}
function exactLog(lines) {
  const result = [];
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].startsWith('|split|')) { result.push(lines[i+1]); i += 2; }
    else result.push(lines[i]);
  }
  return result;
}
const summary = [];
for (let game = 0; game < 10; game++) {
  const seed = [111, 222, 333, game + 1];
  const policySeed = [444, 555, 666, game + 1];
  const rng = new PRNG(policySeed);
  const sets = [0, 1].map(player => Teams.generate(format, {seed: [777, 888, game+1, player+1]}));
  const battle = new Battle({formatid: format, seed});
  battle.setPlayer('p1', {name: 'Policy1', team: Teams.pack(sets[0])});
  battle.setPlayer('p2', {name: 'Policy2', team: Teams.pack(sets[1])});
  const initial = battle.sides.map(side => side.pokemon.map(p => ({set: p.set,
    baseStats: p.species.baseStats, stats: {...p.baseStoredStats}, maxhp: p.maxhp,
    types: p.species.types, ability: p.ability, item: p.item,
    nature: {plus: Dex.natures.get(p.set.nature).plus || null, minus: Dex.natures.get(p.set.nature).minus || null}})));
  const steps = [];
  let cursor = 0;
  while (!battle.ended) {
    if (steps.length >= 5000) throw new Error(`Decision cap reached in game ${game}`);
    const before = snapshot(battle);
    const decisions = battle.sides.map(side => {
      const r = side.activeRequest;
      if (!r || r.wait) return {choice: '', reason: 'wait'};
      if (r.teamPreview) return {choice: 'team 123456', reason: 'preview'};
      if (r.side.pokemon.some(p => p.reviving)) {
        const targets = r.side.pokemon.flatMap((p, i) => p.condition.endsWith(' fnt') ? [i+1] : []);
        return {choice: `switch ${rng.sample(targets)}`, reason: 'revival', switches: targets};
      }
      const switches = r.side.pokemon.flatMap((p, i) => !p.active && !p.condition.endsWith(' fnt') ? [i+1] : []);
      if (r.forceSwitch) return {choice: `switch ${rng.sample(switches)}`, reason: 'forced-switch', switches};
      const moves = r.active[0].moves.flatMap((m, i) => !m.disabled && m.pp !== 0 ? [i+1] : []);
      const legalSwitches = r.active[0].trapped ? [] : switches;
      const roll = rng.random();
      const switching = legalSwitches.length && (roll < 0.1 || !moves.length);
      const pool = switching ? legalSwitches : moves;
      if (!pool.length) throw new Error('No legal choice advertised');
      return {choice: `${switching ? 'switch' : 'move'} ${rng.sample(pool)}`, reason: switching ? 'switch' : 'move',
              roll, moves, switches: legalSwitches};
    });
    for (let i = 0; i < 2; i++) {
      if (decisions[i].choice && !battle.sides[i].choose(decisions[i].choice)) {
        throw new Error(`Rejected game ${game}: ${JSON.stringify(decisions[i])}: ${battle.sides[i].choice.error}`);
      }
    }
    battle.commitChoices();
    const events = exactLog(battle.log.slice(cursor));
    cursor = battle.log.length;
    steps.push({turn: battle.turn, before, decisions, after: snapshot(battle), events});
  }
  const movesUsed = [...new Set(steps.flatMap(s => s.events).filter(l => l.startsWith('|move|')).map(l => l.split('|')[3]))];
  const moves = Object.fromEntries(movesUsed.map(name => {
    const m = Dex.moves.get(name);
    return [name, {id:m.id, type:m.type, category:m.category, basePower:m.basePower,
      accuracy:m.accuracy, priority:m.priority, flags:m.flags}];
  }));
  // Controlled probes use observed move powers/categories, but deliberately
  // remove special move effects, STAB, typing, items, abilities and randomness.
  // They test damage arithmetic, not replay parity for those complex moves.
  battle.field.weather = '';
  battle.field.terrain = '';
  battle.field.pseudoWeather = {};
  for (const side of battle.sides) {
    side.sideConditions = {};
    for (const pokemon of side.pokemon) {
      pokemon.ability = 'noability'; pokemon.item = ''; pokemon.status = '';
      pokemon.volatiles = {};
      for (const stat of Object.keys(pokemon.boosts)) pokemon.boosts[stat] = 0;
    }
  }
  battle.random = () => 0; // Showdown's maximum damage roll; willCrit=false below.
  const source = battle.sides[0].active[0];
  const target = battle.sides[1].active[0];
  const damageProbes = movesUsed.filter(name => moves[name].basePower > 0 && moves[name].category !== 'Status').map(name => {
    const m = moves[name];
    const plain = Dex.getActiveMove(m.category === 'Physical' ? 'tackle' : 'watergun');
    plain.basePower = m.basePower; plain.type = '???'; plain.willCrit = false;
    return {observedMove:name, category:m.category, power:m.basePower, level:source.level,
      attackStats:{...source.storedStats}, defenseStats:{...target.storedStats},
      expected:battle.actions.getDamage(source, target, plain)};
  });
  const record = {revision, format, seed, policySeed, initial, moves, winner:battle.winner,
                  turns:battle.turn, steps, log:exactLog(battle.log), damageProbes,
    typeProbes: movesUsed.flatMap(name => initial.flat().map(p => ({move:name, attackType:moves[name].type,
      defenderTypes:p.types, multiplier:Dex.getImmunity(moves[name].type, p.types) ? 2 ** Dex.getEffectiveness(moves[name].type, p.types) : 0}))) };
  fs.writeFileSync(path.join(out, `battle-${game+1}.json`), JSON.stringify(record));
  fs.writeFileSync(path.join(out, `battle-${game+1}.log`), record.log.join('\n') + '\n');
  summary.push({game:game+1, turns:battle.turn, winner:battle.winner, decisions:steps.length});
  console.log(JSON.stringify(summary.at(-1)));
  battle.destroy();
}
fs.writeFileSync(path.join(out, 'manifest.json'), JSON.stringify({revision, format,
  policy:'0.9 uniform legal move; 0.1 uniform legal switch; fallback when a class is unavailable; forced switches uniform; no transformations',
  games:summary}, null, 2));
