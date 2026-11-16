/* Capture passive ability/item reference cases from the local Showdown simulator.
 * SHOWDOWN_REVISION=<commit> node tools/record_passive_mechanics.cjs [checkout]
 */
const fs = require('node:fs');
const path = require('node:path');
const root = process.argv[2] || '/home/mathy/pokemon-showdown';
const {Battle, Dex, Teams} = require(path.join(root, 'dist/sim'));
const revision = process.env.SHOWDOWN_REVISION;
if (!revision) throw new Error('Set SHOWDOWN_REVISION');
function create(config={}) {
  const battle = new Battle({formatid:'gen9customgame', seed:[1,2,3,4]});
  function set(species, ability, item) {
    return {species, ability, item, level:100, nature:'Serious',
      evs:{hp:4,atk:4,def:4,spa:4,spd:4,spe:4}, moves:[config.firstMove || 'Splash','Flamethrower','Tackle','Rest']};
  }
  battle.setPlayer('p1', {team:Teams.pack([set(config.species || 'Mew', config.ability || 'No Ability', config.item || '')])});
  battle.setPlayer('p2', {team:Teams.pack([set('Mew', config.defender || 'No Ability', '')])});
  battle.makeChoices('team 1','team 1');
  const a=battle.p1.active[0], b=battle.p2.active[0];
  a.setType(config.sourceType || '???'); b.setType(config.targetType || '???');
  if (config.status) a.setStatus(config.status, a);
  if (config.hp) a.hp=Math.floor(a.maxhp*config.hp);
  if (config.weather) battle.field.setWeather(config.weather,a);
  return {battle,a,b};
}
function record(p) {
  return {species:p.species.num, baseStats:p.species.baseStats, stats:p.storedStats,
    maxhp:p.maxhp, hp:p.hp, types:p.getTypes()};
}
const damageConfigs=[];
for (const [ability,move,extra] of [
 ['Adaptability','Flamethrower',{sourceType:'Fire'}], ['Blaze','Flamethrower',{hp:.25}],
 ['Overgrow','Giga Drain',{hp:.25}], ['Torrent','Waterfall',{hp:.25}], ['Swarm','Bug Buzz',{hp:.25}],
 ['Huge Power','Tackle'],['Pure Power','Tackle'],['Guts','Tackle',{status:'brn'}],
 ['Toxic Boost','Tackle',{status:'psn'}],['Toxic Boost','Tackle',{status:'tox'}],
 ["Dragon's Maw",'Dragon Claw'],['Rocky Payload','Power Gem'],['Transistor','Thunderbolt'],
 ['Strong Jaw','Bite'],['Iron Fist','Fire Punch'],['Tough Claws','Tackle'],['Mega Launcher','Dragon Pulse'],
 ['Punk Rock','Boomburst'],['Sharpness','Leaf Blade'],['Water Bubble','Waterfall'],
 ['Tinted Lens','Giga Drain',{targetType:'Grass'}],
 ['Hustle','Tackle'], ['Reckless','Double-Edge'], ['Reckless','High Jump Kick'],
 ['Sniper','Wicked Blow',{crit:true}],
 ['Power Spot','Tackle'],
]) damageConfigs.push({ability,move,...extra});
for (const [defender,move,extra] of [
 ['Thick Fat','Flamethrower'],['Heatproof','Flamethrower'],['Fur Coat','Tackle'],
 ['Fur Coat','Psyshock'],['Ice Scales','Flamethrower'],['Punk Rock','Boomburst'],
 ['Purifying Salt','Shadow Ball'],['Filter','Flamethrower',{targetType:'Grass'}],
 ['Solid Rock','Flamethrower',{targetType:'Grass'}],['Prism Armor','Flamethrower',{targetType:'Grass'}],
 ['Fluffy','Tackle'],['Fluffy','Fire Punch'],['Multiscale','Flamethrower'],['Shadow Shield','Flamethrower'],
 ['Water Bubble','Flamethrower'],['Dry Skin','Flamethrower'],
 ['Tablets of Ruin','Tackle'],
]) damageConfigs.push({defender,move,...extra});
for(const [item,move,extra] of [
 ['Expert Belt','Flamethrower',{targetType:'Grass'}],['Expert Belt','Flamethrower'],
 ['Magnet','Thunderbolt'],['Magnet','Tackle'],['Silk Scarf','Tackle'],
 ['Light Ball','Tackle',{species:'Pikachu'}],['Light Ball','Thunderbolt',{species:'Pikachu'}],
 ['Light Ball','Tackle'],['Soul Dew','Dragon Pulse',{species:'Latios'}],
 ['Soul Dew','Psychic',{species:'Latias'}],['Soul Dew','Tackle',{species:'Latios'}],
 ['Lustrous Orb','Dragon Pulse',{species:'Palkia'}],['Lustrous Orb','Waterfall',{species:'Palkia'}],
 ['Lustrous Orb','Tackle',{species:'Palkia'}]
]) damageConfigs.push({item,move,...extra});
const damage=damageConfigs.map(config=>{
 const {battle,a,b}=create(config); const rolls=[];
 for(let roll=0;roll<16;roll++) {battle.random=()=>roll;const move=Dex.getActiveMove(config.move);move.willCrit=!!config.crit;rolls.push(battle.actions.getDamage(a,b,move));}
 const result={config,source:record(a),target:record(b),rolls};battle.destroy();return result;
});
const speed=[];
for(const config of [
 {item:'Choice Scarf'}, {item:'Choice Scarf',stage:1}, {item:'Choice Scarf',status:'par'},
 {ability:'Chlorophyll',weather:'sunnyday',stage:1}, {ability:'Swift Swim',weather:'raindance'},
 {ability:'Slush Rush',weather:'snowscape'}, {ability:'Sand Rush',weather:'sandstorm'},
 {ability:'Chlorophyll',weather:'sunnyday',defender:'Cloud Nine'},
 {ability:'Quick Feet',status:'par',stage:1}, {ability:'Quick Feet',status:'brn'},
 {ability:'Surge Surfer',terrain:'electricterrain'}, {item:'Choice Scarf',tailwind:true,stage:1}
]) {
 const {battle,a,b}=create(config);
 if(config.terrain)battle.field.setTerrain(config.terrain,a);
 if(config.tailwind)a.side.addSideCondition('tailwind',a);
 if(config.stage)a.boosts.spe=config.stage;
 speed.push({config,source:record(a),expected:a.getStat('spe')});battle.destroy();
}
const statuses=[];
for(const [ability,status,weather] of [
 ['Limber','par'],['Water Veil','brn'],['Water Bubble','brn'],['Thermal Exchange','brn'],
 ['Insomnia','slp'],['Vital Spirit','slp'],['Purifying Salt','psn'],['Purifying Salt','tox'],
 ['Purifying Salt','par'],['Purifying Salt','brn'],['Purifying Salt','slp'],['Purifying Salt','frz'],
 ['Leaf Guard','brn','sunnyday'],['Leaf Guard','brn','raindance']
]) {
 const config={defender:ability,weather};const {battle,a,b}=create(config);
 b.trySetStatus(status,a,Dex.moves.get('toxic'));
 statuses.push({ability,status,weather:weather || '',expected:b.status});battle.destroy();
}
const berries=[];
for(const [item,status,confusion,ability=''] of [['Chesto Berry','slp',false],['Chesto Berry','brn',false],
 ['Lum Berry','tox',false],['Lum Berry','par',false],['Lum Berry','',true],['Lum Berry','brn',true],
 ['Sitrus Berry','',false,'Cheek Pouch'],['Lum Berry','brn',false,'Cheek Pouch']]) {
 const {battle,a}=create({item,ability}); a.status=status;if(confusion)a.addVolatile('confusion');
 if(ability)a.hp=100;const initialHP=a.hp;const source=record(a);battle.eachEvent('Update');
 berries.push({item,status,confusion,ability,source,initialHP,expectedHP:a.hp,expectedStatus:a.status,expectedItem:a.item,expectedConfusion:!!a.volatiles.confusion});battle.destroy();
}
const residuals=[];
for(const config of [
 {item:'Flame Orb'}, {item:'Toxic Orb'},
 {ability:'Poison Heal',status:'psn',hp:.5}, {ability:'Poison Heal',status:'tox',hp:.5},
 {ability:'Hydration',status:'brn',weather:'raindance'}, {ability:'Speed Boost'},
 {ability:'Ice Body',weather:'snowscape',hp:.5},
 {item:'White Herb',firstMove:'Shell Smash'}
 ,{defender:'Bad Dreams',status:'slp'}
]) {
 const {battle,a,b}=create(config); const source=record(a);
 if(config.status==='slp')a.statusState.time=3;
 battle.makeChoices('move 1','move 1');
 residuals.push({config,source,expectedHP:a.hp,expectedStatus:a.status,
   expectedItem:a.item,expectedBoosts:{...a.boosts}});battle.destroy();
}
const reactions=[];
for(const config of [
 {defender:'Aftermath',move:'Tackle',targetHP:1},
 {defender:'Aftermath',move:'Tackle',targetHP:1,item:'Protective Pads'},
 {defender:'Aftermath',move:'Tackle',targetHP:1,ability:'Damp'},
 {defender:'Aftermath',move:'Shadow Ball',targetHP:1},
 {defender:'Wind Rider',move:'Hurricane'}, {defender:'Wind Rider',move:'Heat Wave'},
 {ability:'Wind Rider',move:'Tailwind'}
]) {
 const {battle,a,b}=create(config);if(config.targetHP)b.hp=config.targetHP;
 const source=record(a),target=record(b);
 const move=Dex.getActiveMove(config.move);move.accuracy=true;move.willCrit=false;
 battle.actions.useMove(move,a,b);
 reactions.push({config,source,target,expectedSourceHP:a.hp,expectedTargetHP:b.hp,
   expectedSourceBoosts:{...a.boosts},expectedTargetBoosts:{...b.boosts}});battle.destroy();
}
fs.writeFileSync('unit_tests/fixtures/showdown/passive-mechanics.json',JSON.stringify({revision,damage,speed,statuses,berries,residuals,reactions},null,2)+'\n');
console.log(JSON.stringify({damage:damage.length,speed:speed.length,statuses:statuses.length,berries:berries.length,residuals:residuals.length,reactions:reactions.length}));
