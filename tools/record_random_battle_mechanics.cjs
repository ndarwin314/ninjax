/* Inventory every pinned Gen 9 random-battle role and record controlled oracles.
 * node tools/record_random_battle_mechanics.cjs /home/mathy/pokemon-showdown
 * Uses the same simulator as the local server; no dependency installation.
 */
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(process.argv[2] || '/home/mathy/pokemon-showdown');
const out = path.resolve(process.argv[3] || 'unit_tests/fixtures/showdown/gen9-random-mechanics.json');
const {Battle, Dex, Teams} = require(path.join(root, 'dist/sim'));
const revision = process.env.SHOWDOWN_REVISION;
if (!revision) throw new Error('Set SHOWDOWN_REVISION to the checkout commit');
const source = JSON.parse(fs.readFileSync(path.join(root, 'data/random-battles/gen9/sets.json')));
const cataloguePath = path.resolve('ninjax/data/prototype-moves.json');
const catalogue = JSON.parse(fs.readFileSync(cataloguePath));
// Reviewed metadata-only additions. Complex callbacks, volatile lifecycles,
// multiple secondaries and conditional powers deliberately remain gaps.
const additions = `Accelerock|Aeroblast|Agility|Air Slash|Apple Acid|Aqua Cutter|Aqua Jet|Armor Cannon|Astral Barrage|Aura Sphere|Behemoth Blade|Bite|Bitter Malice|Blue Flare|Body Slam|Bolt Strike|Boomburst|Brave Bird|Bug Buzz|Bulk Up|Bullet Punch|Coil|Cosmic Power|Crabhammer|Dazzling Gleam|Diamond Storm|Discharge|Double-Edge|Dragon Ascent|Dragon Claw|Drain Punch|Draining Kiss|Drill Run|Dynamic Punch|Energy Ball|Fiery Wrath|Fire Blast|Fire Punch|Fleur Cannon|Focus Blast|Glacial Lance|Head Smash|Headbutt|Headlong Rush|Heat Wave|High Horsepower|Hypnosis|Ice Beam|Ice Hammer|Ice Punch|Ice Shard|Iron Defense|Iron Tail|Jet Punch|Kowtow Cleave|Lava Plume|Leaf Blade|Leaf Storm|Leech Life|Liquidation|Lunge|Luster Purge|Mach Punch|Make It Rain|Malignant Chain|Matcha Gotcha|Megahorn|Milk Drink|Mud Shot|Muddy Water|Mystical Fire|Night Slash|Nuzzle|Origin Pulse|Overdrive|Petal Blizzard|Play Rough|Poison Fang|Poison Jab|Pounce|Power Gem|Power Whip|Precipice Blades|Psychic|Psycho Boost|Psycho Cut|Quiver Dance|Razor Shell|Rock Polish|Rock Tomb|Sacred Fire|Scald|Scorching Sands|Seed Bomb|Seed Flare|Shadow Claw|Shell Smash|Shift Gear|Slack Off|Sleep Powder|Sludge Bomb|Sludge Wave|Soft-Boiled|Spacial Rend|Spirit Break|Steam Eruption|Stone Edge|Strange Steam|Stun Spore|Superpower|Surf|Tail Glow|Thunder Punch|Thunderbolt|Vacuum Wave|Victory Dance|Volt Tackle|Water Pulse|Waterfall|Wave Crash|Wicked Blow|Wild Charge|Wood Hammer|Work Up|X-Scissor|Zen Headbutt|Zing Zap`.split('|');
function metadata(m) {
  return {id:m.id, num:m.num, type:m.type, category:m.category, power:m.basePower,
    accuracy:m.accuracy === true ? 1 : m.accuracy / 100, alwaysHit:m.accuracy === true,
    priority:m.priority, pp:m.pp, contact:!!m.flags.contact, flags:m.flags,
    critRatio:m.critRatio || 1, willCrit:!!m.willCrit,
    ...(m.boosts && {boosts:m.boosts}), ...(m.self && {self:m.self}),
    ...(m.secondary && {secondary:m.secondary}), ...(m.drain && {drain:m.drain}),
    ...(m.recoil && {recoil:m.recoil}), ...(m.heal && {heal:m.heal}),
    ...(m.status && {status:m.status})};
}
for (const name of additions) catalogue[name] = metadata(Dex.moves.get(name));
fs.writeFileSync(cataloguePath, JSON.stringify(catalogue, null, 2) + '\n');
const roles = Object.entries(source).flatMap(([species, data]) => data.sets.map((set, index) =>
  ({species, index, level:data.level, ...set})));
const names = [...new Set(roles.flatMap(r => r.movepool))].sort();
const moves = Object.fromEntries(names.map(name => [name, metadata(Dex.moves.get(name))]));
// Enumerate all 16 damage rolls with real stats; no type, critical, ability or
// item modifiers. This avoids assuming JAX and Showdown share RNG streams.
const damage = [];
const boosts = [];
const secondaries = [];
for (const name of additions) {
  const m = Dex.moves.get(name);
  const battle = new Battle({formatid:'gen9customgame', seed:[1,2,3,4]});
  const set = {species:'Mew', ability:'No Ability', item:'', nature:'Serious', level:100,
    evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:0}, moves:[name]};
  battle.setPlayer('p1', {team:Teams.pack([set])});
  battle.setPlayer('p2', {team:Teams.pack([{...set, moves:['Splash']}])});
  battle.makeChoices('team 1', 'team 1');
  const a = battle.p1.active[0], b = battle.p2.active[0];
  a.setType('???'); b.setType('???');
  if (m.basePower > 0) {
    const rolls = [];
    for (let roll = 0; roll < 16; roll++) {
      battle.random = () => roll;
      const active = Dex.getActiveMove(name); active.willCrit = false;
      rolls.push(battle.actions.getDamage(a, b, active));
    }
    damage.push({name, level:a.level, stats:a.storedStats, hp:a.maxhp, rolls});
  }
  if (m.boosts) {
    battle.actions.useMove(name, a, b);
    boosts.push({name, expected:{...a.boosts}});
  }
  if (m.secondary) {
    // Trigger a real secondary in Showdown without matching PRNG algorithms.
    b.ability = 'battlearmor';
    battle.random = (from, to) => to === undefined ? 0 : from;
    battle.randomChance = () => true;
    battle.actions.useMove(name, a, b);
    secondaries.push({name, status:b.status, boosts:{...b.boosts},
      confusion:!!b.volatiles.confusion, flinch:!!b.volatiles.flinch});
  }
  battle.destroy();
}
// Items are chosen procedurally, not listed exhaustively in sets.json. Keep a
// reproducible sample explicitly distinguished from the exhaustive role list.
const sampledItems = new Set();
for (let i=0; i<100; i++) for (const set of Teams.generate('gen9randombattle', {seed:[7,8,9,i+1]})) sampledItems.add(set.item);
fs.mkdirSync(path.dirname(out), {recursive:true});
fs.writeFileSync(out, JSON.stringify({revision, format:'gen9randombattle',
  roles, moves, abilities:[...new Set(roles.flatMap(r=>r.abilities))].sort(),
  sampledItems:[...sampledItems].sort(), itemSampleTeams:100, additions, damage, boosts, secondaries}, null, 2)+'\n');
console.log(JSON.stringify({roles:roles.length, moves:names.length, additions:additions.length, damage:damage.length, boosts:boosts.length, revision}));
