# Move implementation coverage and remaining architecture

Scarlet/Violet is the default ruleset. After implementing Disable, Encore and
Taunt, Worry Seed, Ivy Cudgel, Blizzard, Hurricane, Substitute, Electro Shot,
Hyper Beam, Destiny Bond, Knock Off, Trick, Switcheroo, Roost, High Jump Kick,
Explosion, Memento, Spiky Shield, Fake Out, Sucker Punch, Stomping Tantrum and
Foul Play, Beat Up, Sleep Talk and Whirlwind, all move slots in the original
curated sets have represented singles implementations. This does not cover the
expanded `moves.json` catalogue. The restriction design discussion below records
the chosen dedicated-field approach.

## Current coverage contract

The catalogue contains **685 moves**, including **619** with represented singles
effects and **66** unavailable or unreviewed. The pinned random-battle inventory
has **330 of 348** moves supported, leaving **18** gaps.
These are implementation counts, not exhaustive interaction-parity counts.

The current inventory includes Judgment, Freeze-Dry and Flying Press, as well as
Teleport, Chilly Reception, Baton Pass, Shed Tail, Raging Bull, six ordinary
charge moves and six semi-invulnerable charge moves. The curated item loader
supports all 17 typed Plates.
All catalogue moves with `selfSwitch` or `forceSwitch` are supported for singles.
Healing Wish, Lunar Dance and Revival Blessing are also supported. Sky Drop
still needs a charge continuation that carries another Pokémon.

### Review of partially wired moves

| Moves | Current singles behavior |
| --- | --- |
| Flail, Reversal | Both use the six-bracket low-HP power table at the hit boundary. |
| Terrain Pulse | The grounded user's terrain sets the type and doubles base power during move preparation. The resolved type reaches immunity and damage, and ordinary terrain bonuses apply in the modifier chain. |
| Photon Geyser, Shell Side Arm | The shared category resolver is active. Photon Geyser uses staged Attack and Special Attack, stays Special on a tie, and bypasses breakable abilities unless Ability Shield is active. |
| Grass Knot, Heat Crash, Power Trip, Temper Flare | Existing weight, stage and previous-failure rules are available through the catalogue. |
| Thunder, Bleakwind Storm, Wildbolt Storm, Sandsear Storm, Gust, Twister | Weather accuracy, represented secondary effects and the existing airborne-hit rules are wired. Gust and Twister double damage against Fly and base power against Bounce. |
| Seismic Toss, Supercell Slam, Axe Kick, Self-Destruct, Misty Explosion | Use the shared level-damage, crash or pre-hit self-faint rules. Damp also covers the new explosion entries. |
| Dream Eater, Life Dew, Snore, Growth, Grav Apple, First Impression | Reuse the drain, healing, sleep, stat-stage, field-power and first-action paths with move-specific conditions. |
| Psychic Fangs, Mortal Spin, Steel Roller, Hyper Drill, Mighty Cleave | Reuse screen clearing, hazard and binding cleanup, terrain removal, or protection bypass. Hyper Drill and Mighty Cleave leave the guard active. |
| Bone Rush, Freezing Glare, Inferno, Smart Strike | Use existing multi-hit, status-secondary or sure-hit metadata. |

Focused checks are in `unit_tests/test_partial_move_wiring.py`,
`test_move_category.py`, `test_ability_bypassing_moves.py` and
`test_freeze_confusion.py`. Effects attached to unavailable conditions or battle
formats, including Minimize, Sky Drop and Dynamax, remain outside the reviewed
singles behavior.

Rising Voltage is now enabled for singles: Electric Terrain doubles intrinsic
power when the target is grounded, independently of the user's ordinary terrain
bonus. Gravity, Levitate, Flying typing and Air Balloon participate in grounding.
The intrinsic doubling precedes Technician; Expanding Force's separate modifier
lives in the terrain modifier helper after Technician's eligibility check.
Dry Skin's Fire vulnerability is grouped with weather-related ability power
effects but applies in every weather. Solar Power is unchanged by this batch.
Represented BasePower effects now chain integer modifiers in Showdown priority
order, with half-up chain rounding and one half-down application to base power.
Knock Off and type-conversion ability boosts participate in that chain instead
of rounding power during move-property resolution. Focused comparisons use
actual Showdown event results in `fixtures/showdown/power-modifiers.json`;
this does not establish full damage or unrepresented-effect parity.

The explicit support inventory is
[`supported-moves.json`](../../ninjax/data/supported-moves.json), read through
`ninjax.moves.support`. It includes the previous 242-move projection plus reviewed
generic damage, status, stat-change, drain, recoil and multi-hit additions.
Catalogue membership, successful construction, and neutral damage probes alone
do not establish implementation support. New names fail closed until reviewed.

`make_move` retains known metadata for unsupported entries but sets
`implemented=False` and PP to zero. Team construction reports these entries as
omissions even when they occur in `moves.json`. Random-battle metadata checks
skip unsupported entries explicitly. Controlled probes remain limited comparisons
of the particular effects they exercise, not certifications of whole moves.

Regenerate the machine-readable report with the configured project interpreter:

```sh
.venv/bin/python -m tools.audit_random_battle_mechanics > docs/mechanics/gen9-random-coverage.json
```

The [coverage report](gen9-random-coverage.json) lists all unavailable catalogue
moves, supported random-battle names, random-battle gaps and affected roles.
The report uses the same support inventory as construction. Its ability/item
sections retain their separate projection and sampling scopes. Existing
historical batch counts in other design documents describe those earlier runs.

Plate, Fling and represented berry effects respect effective Klutz and the
Magic Room and Embargo timers. The Magic Room and Embargo moves and some
other held-item suppression checks remain unimplemented.

To enable a move, review its complete primary effect and metadata against the
execution path, add any required mechanics and focused regressions, then add its
display name to the support inventory and regenerate the report. Do not infer
support from base power, move flags or the existence of a catalogue record.

## Implementation readiness after the catalogue field expansion

The expanded catalogue now supplies `volatileStatus`, `sideCondition`,
`selfSwitch`, `forceSwitch`, `weather`, `terrain`, `overrideOffensiveStat`,
`overrideDefensiveStat`, `ignoreOffensive`, `ignoreNegativeOffensive` and
`ignorePositiveDefensive`, alongside the existing `ignoreDefensive` field.
These are host-side JSON descriptors. Construction now projects weather, terrain,
primary confusion and stat overrides into move arrays. Reflect, Detect, Flip Turn
and Roar reuse existing execution paths; some dispatch still selects by move
name or ID. Other condition strings remain descriptors without lifecycle support.

### Implemented batch using existing singles state

These **17 moves are now enabled** using existing singles state and shared
mechanics. The table records the wiring for this batch. Primary confusion adds
one boolean move-metadata leaf, not a new battle-state condition. Focused behavior
checks are in `unit_tests/test_wired_moves.py`; support remains scoped to the
represented singles effects and the existing interaction limitations.

| Catalogue information | Enabled moves | Wiring and scope |
| --- | --- | --- |
| `weather` | Sandstorm, Snowscape | Normalize weather names into `WeatherEnum`, project `weather_effect`, and reuse `set_weather` and existing weather residuals. Ensure field targeting avoids opponent-only immunity and Substitute checks. |
| `terrain` | Electric Terrain, Psychic Terrain, Misty Terrain | Map into `TerrainEnum`, project `terrain_effect`, and reuse `set_terrain` with field targeting. Grassy Terrain is already enabled. |
| `sideCondition: reflect` | Reflect | Extend screen application to the existing `BattleState.reflect` timer, including Light Clay, expiry and existing physical-damage reduction. Side targeting must bypass opponent-only hit gates. |
| `overrideDefensiveStat: def` | Psystrike, Secret Sword | Project `defensive_stat` from metadata instead of limiting the override to Psyshock. Body Press's `overrideOffensiveStat: def` likewise describes an already-enabled effect. |
| `volatileStatus: confusion` | Confuse Ray, Supersonic, Sweet Kiss | Dispatch primary confusion through `side.status_effects.apply_confusion` after normal hit checks; retain its existing counter, immunity and switch lifecycle. |
| `volatileStatus: confusion` plus `boosts` | Flatter, Swagger | Compose the existing target stat-change and confusion helpers in move order. Determine success for the whole move: a blocked boost must not automatically discard successful confusion, or vice versa. |
| `volatileStatus: confusion`, `target: allAdjacent` | Teeter Dance | Reuse primary confusion for the opposing active in singles. Full doubles execution still needs target enumeration and an attempt for each eligible adjacent Pokémon. |
| `volatileStatus: protect` | Detect | Shares Protect's activation, consecutive-use counter and queued-action timing. |
| `selfSwitch: true` on a damaging move | Flip Turn | Project `switch_out` and reuse the U-turn/Volt Switch pivot continuation. Preserve no-switch behavior on a miss, protection or absorption and when the user cannot switch. |
| `forceSwitch: true` on a status move | Roar | Generalize Whirlwind's ID-specific routing to a forced-switch descriptor. Reuse replacement selection, switch lifecycle and displaced-actor cancellation, while applying Roar's sound flag and keeping Whirlwind-specific wind handling separate. |

Weather strings are not consistently cased: the current values include
`RainDance`, `Sandstorm`, `sunnyday` and `snowscape`. Normalize them through an
explicit mapping; `snowscape` selects `WeatherEnum.SNOW`. Terrain values are
Showdown condition IDs such as `electricterrain`. Do this conversion on the
host and pass numeric, fixed-shape metadata into traced execution.

### Described by the new fields, with additional execution work

| Moves | What the fields provide | Work still required |
| --- | --- | --- |
| Mist, Safeguard | Named `sideCondition` values | Add timers and checks in stat and status routing. |
| Attract, Charge, Defense Curl, Dragon Cheer, Focus Energy, Magnet Rise, Minimize, Smack Down, Stockpile, Tar Shot | Their condition identity | Move-specific condition payloads, application rules, consumers and expiry. The condition string identifies the required handler. |

`selfSwitch` must retain its distinction between boolean `true`, `copyvolatile`
and `shedtail`; reducing all three to a boolean loses transfer semantics.
Compound moves must execute every
required effect before claiming support.

Teleport and Chilly Reception use the replacement continuation; Chilly Reception
sets snow first. Baton Pass transfers represented passable conditions and stat
stages, while Shed Tail pays half maximum HP and transfers only its Substitute.
The transfer is applied before entry hazards and abilities. The policy currently
covers Substitute, confusion, Taunt and Perish Song; new passable conditions
must be added to the transfer helper as their mechanics are implemented.

### Gaps the field expansion does not resolve

The added stat-ignore fields need separate offensive/defensive stage policies
if used; they must not all map to the existing defensive-ignore boolean.
Currently `ignoreOffensive`, `ignoreNegativeOffensive` and
`ignorePositiveDefensive` are false for every catalogue entry. The true
`ignoreDefensive` entries are Darkest Lariat and Sacred Sword, both already
enabled, so these additions alone unlock no additional move names.

Move construction projects both `secondary` and `secondaries` into
`Move.secondaries`. Fire Fang, Ice Fang, Thunder Fang and Triple Arrows roll
each slot independently. Tri Attack and Dire Claw use mutually exclusive
slots so a hit chooses at most one status.

Construction projects `thawsTarget` separately from the user's `defrost` flag.
Hydro Steam, Matcha Gotcha, Scald, Scorching Sands and Steam Eruption thaw a
surviving frozen target after secondary effects when the hit reaches it.

Freeze-Dry and Flying Press use shared move-aware effectiveness rules, and
Seismic Toss shares level-based damage with Night Shade. Wish uses a delayed
position effect. Future Sight and Doom Desire need delayed attack payloads.
Transformations, richer battle history and doubles target resolution remain
architectural work as described below.

## Conditional damage/type rules: pipeline placement

This section records implemented pipeline ownership and follow-up plans. The placements
below were checked against the local Showdown reference's `data/moves.ts`,
`data/conditions.ts`, `sim/pokemon.ts` and `sim/battle-actions.ts` at the pinned
revision `23b41aa29e2ff5ff80e03f57b257b595a5852047`.

### Execution stages and current owners

1. **Resolve transient move properties before accuracy and hit routing.**
   `moves.properties.effective_move` / `_override_type` own type changes;
   category-dependent moves also need transient category, offensive-stat,
   defensive-stat and contact updates before their consumers run. Do not mutate
   the stored move slot. Resolve keyed category ties once during execution,
   rather than drawing randomness in a property helper that can be called for
   previews or bookkeeping.
2. **Resolve move-aware type effectiveness.** `side.queries.move_immune_from_fields`
   currently performs early immunity checks; `calculations.damage.hit_type_effectiveness`
   later provides raw matchup and damage effectiveness. Both must use consistent
   move-specific rules. The resulting matchup also feeds conditional power,
   Expert Belt and reactive item checks.
3. **Compute intrinsic base power at the appropriate hit-time snapshot.**
   `PowerRule` / `resolve_hit_base_power` now own all represented hit-time
   callbacks, including Acrobatics, Stored Power, weights, Stomping Tantrum,
   HP/status/speed rules and Rising Voltage. `compute_base_power` invokes this
   single dispatcher before the modifier chain. Weather Ball remains a
   property-time ModifyMove effect; hit-plan contributions remain supplied power.
4. **Chain BasePower event modifiers.** `chain_move_power_modifiers` owns the
   priority-0 move factors. Keep them after Technician, abilities, items and
   ordinary terrain effects, retaining integer chain rounding and one final
   application to intrinsic power.
5. **Select and modify damage stats.** `resolve_damaging_hit` selects the
   stage-adjusted stats and reads their field and item factors via
   `stat_modifier`. `compute_base_damage` adds the move-dependent ability
   factors and applies each stat's chain once. Category selection happens
   before these damage-specific factors enter the calculation.
6. **Apply damage stages through one calculation.** `compute_damage_multipliers`
   owns weather, critical hits, random rolls, STAB, effectiveness, burn and the
   final damage chain. `apply_weather_damage`, `stab_damage_modifier` and
   `apply_burn_penalty` isolate focused responsibilities. `final_damage_modifier`
   chains represented abilities, held items and singles screens in Speed/effect
   order, then applies the result once. Multiscale is part of this chain for
   both Pokémon and Substitute recipients; HP commits do not reapply it.
   Power and damage chains share fixed-point primitives in
   `calculations.modifiers`. Exceptions must retain their stage's rounding.
   Offline comparisons live in `test_damage_modifiers.py`, recorded by
   `tools/record_damage_modifiers.cjs` against the pinned Showdown revision.

### Implemented conditional-power batch

Fifteen additional moves are enabled: Facade, Venoshock, Barb Barrage, Brine,
Psyblade, Collision Course, Hex, Infernal Parade, Eruption, Water Spout,
Dragon Energy, Crush Grip, Hard Press, Electro Ball and Gyro Ball.

The priority-0 factors flow through `chain_conditional_move_power_modifiers`
and `chain_field_matchup_move_modifiers`. `resolve_hit_base_power` calls the
status, user-HP, target-HP and speed helpers before Technician. Effective Speeds
are gathered through `queries.effective_speed` at hit time; these include stages,
items, paralysis, weather, abilities and Tailwind independently of critical hits
and Unaware. Facade also skips the burn damage penalty. Existing property-time
power rules are not reapplied. The remaining table entries describe follow-up
work; functions in this batch are implemented even where the original proposal
below labels them new.

Validation lives in `test_variable_power_validation.py`,
`test_power_rules.py` and `test_conditional_power.py`, with offline fixtures
captured from the pinned Showdown callbacks and BasePower events. Freeze-Dry
and Flying Press were enabled separately; category-changing moves and
transformations remain outside this batch.

### Move-to-stage map

Numeric helpers take scalar or array facts, with state access at the execution
boundary. The transformation row describes work that still needs a consumer.

| Rule family | Moves | Specific owner and call path |
| --- | --- | --- |
| Intrinsic status-dependent power | Hex, Infernal Parade | **Implemented:** `resolve_hit_base_power` dispatches status-dependent intrinsic power before the modifier chain, including effective Comatose. |
| Intrinsic user-HP power | Eruption, Water Spout, Dragon Energy, Flail, Reversal | The hit-time dispatcher uses proportional power for the first three moves and `low_hp_base_power` for Flail and Reversal. |
| Intrinsic target-HP power | Crush Grip, Hard Press | **Implemented:** the shared nested fixed-point calculation uses the move's scale and live target HP. |
| Intrinsic speed-ratio power | Electro Ball, Gyro Ball | **Implemented:** `resolve_hit_base_power` uses hit-time effective Speeds from `side.queries.effective_speed`, independent of action order and crit/Unaware damage-stat adjustment. |
| Priority-0 conditional power factors | Facade; Venoshock, Barb Barrage; Brine | **Implemented:** `chain_conditional_move_power_modifiers` adds eligible factors without an intermediate application to power. |
| Priority-0 terrain/effectiveness factors | Psyblade, Collision Course | **Implemented:** `chain_field_matchup_move_modifiers` handles Electric Terrain's presence and the resolved super-effective matchup. Psyblade has no user-grounding requirement. |
| Type override plus intrinsic change | Terrain Pulse | `move_overrides` prepares the doubled power and calls `_override_type` before accuracy and immunity. Both use the grounded user's current terrain. |
| Form-dependent type override | Ivy Cudgel, Raging Bull, Aura Wheel | **Implemented** `properties.form_move_type(identifier, default_type, form)` resolves the transient type through `_override_type`. Raging Bull shares the screen-breaking hook with Brick Break. Aura Wheel checks Morpeko's species and uses the current Hunger Switch form. |
| Item-dependent type override | Judgment | **Implemented:** shared Plate lookup resolves Judgment's type and the matching-type BasePower modifier. Klutz and the represented Magic Room/Embargo timers suppress these Plate effects. Arceus Plate transfer restrictions are checked. |
| Special effectiveness | Freeze-Dry, Flying Press | **Implemented:** `type_effectiveness_factors` applies the move rule per defending type. Early immunity, hit-time matchup and fallback damage calculation consume the rule; Ground, Gravity and Levitate remain contextual checks. |
| Weather damage exception | Hydro Steam | **Implemented.** `apply_weather_damage` boosts Hydro Steam by 1.5 in sun and applies the ordinary Water boost in rain. Both use weather-stage rounding. |
| Category selection | Photon Geyser, Shell Side Arm | `properties.resolve_move_category` sets category, stat selectors and conditional contact together. Both moves are supported. |
| Ability bypass | Moongeist Beam, Photon Geyser, Sunsteel Strike | Move metadata bypasses represented breakable abilities during accuracy, damage and hit reactions. Ability Shield preserves the holder's ability while the item is active. |
| Transformation-dependent type/category | Tera Blast, Tera Starstorm | **New** `properties.transformed_move_type(type_rule, default_type, tera_type, is_terastallized, form)` integrates through `_override_type`. Extend `resolve_move_category` for Tera Blast and use **new** `properties.transformation_move_properties(move, tera_type, is_terastallized, form)` for power, target and self-effect changes. Implement these only with the transformation lifecycle and eligibility checks; the existing type/category fields do not unlock complete moves. |

### Hit preparation and damage orchestration

Keep the hit-time power boundary explicit:

```text
resolve_damaging_hit
  gathers effective speeds for intrinsic power and final modifier ordering
  -> compute_base_power(..., user_speed=..., target_speed=...)
     -> resolve_hit_base_power(base_power, power_rule, user_hp, user_max_hp,
                               target_hp, target_max_hp, target_status,
                               target_ability, user_speed, target_speed,
                               terrain_type, grounded_target)
     -> ordered integer modifier chain
     -> one final application to intrinsic power
```

`resolve_hit_base_power` dispatches to the formula helpers using `PowerRule`.
`compute_base_power` receives effective Pokémon snapshots, speeds and the field
and move-history facts needed by those formulas. Power evaluation consumes no
PRNG keys.

`move_overrides` prepares Weather Ball and Terrain Pulse's intrinsic power and
type. `resolve_hit_base_power` handles live weights, held-item presence, positive
stages and previous-turn failure. Stored Power and Power Trip add 20 power per
positive stage, including accuracy and evasion. Prepared weather and terrain
power and fixed hit-plan contributions pass through the intrinsic dispatcher.

Both hit entry points use
`routing.prepare_hit_move(key, state, player, move) -> (key, resolved_move)`.
It calls `effective_move` and resolves the category once before accuracy and hit
routing. Damage and contact checks consume the same transient move. The resolver
uses the ordinary staged stats from `BattleState.raw_boosted_stats`. Photon
Geyser keeps Special on a tie. Shell Side Arm compares integer damage estimates
and draws from the supplied key only on ties. Damage calculation applies its
normal ability, item and critical-hit rules after category selection.

Construction sets `Move.effectiveness_rule` for Freeze-Dry and Flying Press.
The factor helper uses it alongside the resolved move type in early immunity,
hit resolution and fallback damage. STAB, absorption and type-based abilities
read the resolved type.

Facade has one additional owner: **implemented**
`damage.apply_burn_penalty(damage, status, ability, category, identifier)`, called
from the burn stage in `compute_damage_multipliers`.
It returns damage after burn-stage rounding and skips that penalty
for Facade in the Scarlet/Violet ruleset. Keep the existing position relative to
weather, critical damage, random roll, STAB, effectiveness and later modifiers.
Its priority-0 power boost still belongs to
`chain_conditional_move_power_modifiers`. Showdown's power callback excludes
sleep; use the checked callback rather than inferring eligibility from prose.

Moongeist Beam, Photon Geyser and Sunsteel Strike read a move-scoped defender
ability. Type immunities, Focus Sash and unbreakable abilities such as Rough
Skin still apply. Aura Wheel uses the user's Morpeko form, which Hunger Switch
updates at the residual checkpoint.

Raging Bull, Brick Break and Psychic Fangs share the `apply_pre_damage_move_effects` hook.
The pinned Showdown callbacks run after the accuracy, protection and immunity
gates and before Substitute interception. Hit resolution reads the resulting
screen state when calculating damage.

### Data boundaries and next implementation batches

Keep `PowerRule` for power computations. Use explicit metadata/rule selectors
for effectiveness and category changes instead of encoding them as fake move
types or power multipliers. Host construction should assign rule IDs; traced
helpers should consume arrays and the minimal required inputs. A compact
execution-local context can carry effective speeds or selected category when
the existing Pokémon snapshot does not contain them. No generic full-state
parameter is needed for a pure numeric formula.

The implemented rules cover priority-zero factors, Facade's burn exception,
HP and Speed formulas, special effectiveness and the listed property and
category changes. Party and history mechanics such as Last Respects and Rage
Fist still need durable counters.

Validation should compare the relevant stage and full damage against Showdown:
Technician threshold probes for intrinsic versus modifier doubling; odd powers
combined with items/auras/terrain; HP and speed boundaries; Comatose and status
cases; dual-type effectiveness and early immunity consistency; Hydro Steam with
sun/rain and weather suppression; and category ties, stat stages, modifiers and
contact behavior. Use explicit keys and test JIT; reserve vmap/scan acceptance
for the batched checkpoint suite. Coverage stays unchanged until a complete move
has its mechanics, focused tests and support-inventory entry.

### Implemented architecture families

| Family | Represented singles behavior |
| --- | --- |
| [Protection and survival](#protection-and-survival) | Personal shields, side guards, Endure and protection-breaking hits share activation and hit routing. |
| [Multiple and custom secondaries](#multiple-and-custom-secondaries) | Fixed secondary slots support independent and mutually exclusive rolls, including user boosts. |
| Target thawing | Moves marked `thawsTarget` thaw a surviving frozen target after secondaries when the hit reaches it. |
| Move-specific effectiveness | Freeze-Dry and Flying Press use the shared type rules in early immunity, hit resolution and damage. |
| [Switching composition and state transfer](#switching-composition-and-state-transfer) | Pivots, forced switches, switch-entry recovery and party revival are implemented for singles. |
| [Item lifecycle](#item-lifecycle) | Catalogue moves for item swap, theft, destruction, forced consumption, throwing and recovery are implemented for represented items. |
| [Ability, type, form and moveset mutation](#ability-type-form-and-moveset-mutation) | Switch-cleared identity changes and battle-persistent form changes use shared readers, effect handlers and restoration. |
| [Residual conditions and trapping](#residual-conditions-and-trapping) | Passive healing, repeated damage and stat drops use the end-of-turn scheduler. Field, source-linked and self-imposed traps share switch eligibility. |
| [Special damage outcomes](#special-damage-outcomes) | Final Gambit, the four OHKO moves, False Swipe, Present, Steel Beam and Chloroblast share the existing hit and attempt pipeline. |
| [Restrictions and prerequisites](#restrictions-and-prerequisites) | Belch uses persistent Berry history. Last Resort uses the current entry's move-use history. |
| [Locked sequences](#charge-recharge-and-locked-sequences) | Rampage, Rollout and Uproar share the forced continuation payload with separate interruption policies. |
| [Preparation](#preparation-and-semi-invulnerability) | Focus Punch and Beak Blast bind their preparation to the original queued actor. |
| [Damage, action and team history](#damage-action-and-team-history) | Retaliation, per-turn damage and stat loss, cumulative received hits and cumulative faints have separate records. |
| [Custom stats, healing and costs](#custom-stats-healing-and-costs) | Belly Drum, Stockpile and its consumers, Strength Sap and weather-based healing use shared stage and HP helpers. |
| [Field and accuracy conditions](#field-and-accuracy-conditions) | Gravity, Magic Room, Wonder Room, Mist, Safeguard, Magnet Rise, Lock-On and Focus Energy attach to the field, side or active Pokémon. |
| [Ordinary-type Terastallization](#transformations) | Typed singles actions can declare Tera. The engine tracks the side allowance, defensive type, original offensive types and Tera Blast. |

### Remaining architecture families

The examples below are unavailable or partial; the report is the full inventory.
Some moves need several of these capabilities.

| Missing support | Example moves | Required extension |
| --- | --- | --- |
| [Delayed effects](#delayed-effects) | Future Sight, Doom Desire | Position-owned attack payloads with the original caster's attribution. |
| [Carried-target continuations](#preparation-and-semi-invulnerability) | Sky Drop | Bind the carrier and carried Pokémon for a historical ruleset. |
| [Presentation and form interactions](#ability-type-form-and-moveset-mutation) | Illusion, Terastallized forms | Add disguise-aware opponent observations and connect form interactions to the transformation lifecycle. |
| [Calls and queue manipulation](#calls-and-queue-manipulation) | Metronome, Copycat, Instruct, After You, Quash | Complete caller-specific eligibility, PP and continuation handling. After You and Quash require the doubles queue. |
| [Doubles targeting/combinations](#doubles-targeting-and-combinations) | Follow Me, Rage Powder, Ally Switch, Helping Hand, Dragon Darts, pledges | Target coordinates, redirection, spread plans, per-active queues and combinations. |
| [Advanced transformations](#transformations) | Tera Starstorm | Add Stellar and species-specific Tera form lifecycles. |

Use the outlines below as implementation slices. Each slice identifies where to
store state, when to apply the effect, and what to verify before updating the
support inventory. Project catalogue descriptors to numeric rules on the host.
Keep battle data in fixed-shape arrays and advance PRNG keys only for rules
that require a draw. Use singles as the first checkpoint wherever the full
effect can be represented there.

#### Multiple and custom secondaries

- `Move.secondaries` stores three effect slots and aligned probabilities.
  The independent sampler rolls each slot and applies successful effects in
  catalogue order. The exclusive sampler uses disjoint probability ranges,
  leaving an unassigned range for hits that apply no effect.
- Both samplers share status, confusion, stage, flinch and restriction
  handlers. Serene Grace scales the probabilities. Shield Dust and Substitute
  block target effects while user boosts can still apply. An immune exclusive
  outcome ends the secondary attempt.

#### Residual conditions and trapping

- Binding moves store the opposing source slot, move ID, duration and
  residual damage divisor. They tick after poison and burn and expire on
  source departure. Grip Claw extends the duration and Binding Band increases
  the residual damage.
- Shadow Tag, Arena Trap, Magnet Pull and active bindings share the voluntary
  switch predicate. Ghost typing and active Shed Shell permit escape. Check
  the predicate when committing a new switch. Resolve committed switches,
  pivots and forced replacements via the switch pipeline.
- Mean Look, Block, Spirit Shackle and Jaw Lock link their targets to an
  opposing active party slot. No Retreat self-traps and boosts all five stats.
  Fairy Lock uses a two-count field timer that covers the following turn.
- Aqua Ring, Ingrain, Leech Seed and Ghost Curse use indefinite active
  lifetimes and can transfer through Baton Pass. Ingrain also grounds its
  holder and prevents forced drag. Rapid Spin and Mortal Spin clear Leech Seed
  and ordinary binding effects from the user.
- Salt Cure uses live Water or Steel typing for its damage fraction. It
  survives the source leaving and clears when its target switches. Syrup Bomb
  ticks up to three Speed drops and clears when either participant leaves.
  Both go through the secondary-effect gate, including Substitute and Shield
  Dust.
- The relevant checkpoints are Wish at 4, Aqua Ring at 6, Ingrain at 7,
  Leech Seed at 8, Ghost Curse at 12, binding and Salt Cure at 13, Syrup Bomb
  at 14 and Yawn at 23. The HP checkpoints use the shared commit and faint
  path. Leech Seed transfers capped HP loss to the active Pokémon across the
  field and reverses healing with Liquid Ooze. Big Root boosts applicable recovery.
  When binding and Salt Cure affect the same holder, binding currently ticks first.
- Active lifetimes use zero for inactive, a negative value for indefinite
  effects and positive counts for expiring effects. Wishes store duration
  and the original heal amount by position. The observation exposes active
  conditions and turns, while keeping the exact Wish amount private.

#### Delayed effects

- Wish follows the active position and heals the next turn using half the
  caster's maximum HP at the time of the move. Yawn stays on the target and
  checks live sleep immunity when its timer expires.
- Future Sight and Doom Desire still need a target position, a source
  reference and a delayed attack resolver that reuses hit and faint handling.
  Verify switches on either side, duplicate scheduling and source replacement.

#### Restrictions and prerequisites

- Throat Chop and Psychic Noise apply two-turn restrictions through the
  secondary hit path. Selection and execution consult the same move flags.
  Heal Block prevents indirect recovery and automatic Sitrus consumption,
  including when Psychic Noise itself crosses the Berry threshold.
- Torment, Gigaton Hammer and Blood Moon restrict repeat selection based on
  the last move. Imprison checks the current move bank of the opposing source
  at selection and before executing an ordinary move.
- The Berry-consumption flag follows its party slot through switches. Last
  Resort checks the other move slots' identities and usage since entry.
  Interrupted actions and calls that spend only their caller's PP do not
  count as uses of the affected move slot.

#### Charge, recharge and locked sequences

- Catalogue recharge moves use the existing `CONTINUED` action. The repeat
  payload carries policy, move ID, actor slot, target, remaining turns and
  successful hit count.
- Outrage, Thrash, Petal Dance, Rollout and Uproar use separate repeat
  progression, interruption and completion policies. Rampage samples its
  duration once. Rollout counts consecutive hits and reads Defense Curl's
  switch-cleared power effect. Uproar prevents sleep while active.
- Verify PP payment, target selection on continuation, and which outcomes
  start, advance or end the sequence. Cover protection, misses, fainting,
  forced switches, and the final confusion or power reset. Sky Drop needs the
  paired state described in the next outline and a ruleset that includes it.

#### Preparation and semi-invulnerability

- The scheduler prepares both choices before move execution and binds
  preparation to each original actor slot. An opposing attack that reaches a
  prepared Pokémon can stop Focus Punch before PP is spent. Beak Blast burns
  an eligible contact attacker while the preparation is active.
- Reuse the charge lifecycle for existing concealment. For Sky Drop in a
  historical ruleset, bind carrier and carried target, check each one's action
  and target eligibility, and release both from every cleanup path. Test
  Substitute interception, priority, interrupted execution and cleanup when
  either participant leaves.

#### Damage, action and team history

- Counter and Mirror Coat prepare a category filter at queue time and retain
  the latest eligible hit for their retaliation. A hit absorbed by Substitute
  can count for Rage Fist but does not supply retaliation damage.
- Rage Fist counts received hits, and Last Respects counts team faints.
  Assurance reads any actual loss of target HP this turn. Lash Out reads
  actual stage reductions, and the Fusion moves read the preceding successful
  move. The field boundary clears turn-local records.

#### Protection and survival

- **Implemented for represented singles.** Protect, Detect, Spiky Shield,
  Baneful Bunker, Burning Bulwark, Silk Trap and Endure share the stall
  counter and fail when no action remains. The counter caps the denominator at
  729. Quick Guard filters positive-priority hits, and Wide Guard filters moves
  with spread targets, including Expanding Force when terrain changes its
  target. Both side guards refresh stall without rolling its odds.
- Burning Bulwark and Silk Trap allow status moves. Spiky Shield, Baneful
  Bunker, Burning Bulwark and Silk Trap retaliate on blocked contact. Unseen
  Fist bypasses applicable guards. Long Reach and active Protective Pads
  prevent contact retaliation, and Magic Guard prevents Spiky Shield damage.
- Feint, Hyperspace Hole, Phantom Force and Shadow Force clear personal and
  side guards after accuracy and immunity checks. Endure survives each
  lethal direct hit that turn and takes priority over Sturdy and Focus Sash.
  Indirect HP effects can still faint the holder.
- Hyper Drill and Mighty Cleave hit through applicable guards and leave them
  active for later moves.
- King's Shield and Mat Block retain historical mode handling for supplied move
  data. The Scarlet/Violet catalogue omits their records. Doubles still needs
  target enumeration for side protection across multiple active positions.

#### Switching composition and state transfer

- Dragon Tail and Circle Throw reuse the Roar and Whirlwind forced-switch
  continuation after a hit reaches a surviving target directly. The continuation
  honors forced-switch immunity and replaces any queued action of the dragged Pokémon.
  Magic Bounce redirects Roar and Whirlwind toward their original user.
- Parting Shot requests a pivot for its resolved user after the stat drops.
  Magic Bounce changes who pivots, and Mirror Armor redirects the drops.
  Baton Pass transfers eligible volatile state and Shed Tail transfers its Substitute.
- Healing Wish and Lunar Dance leave recovery at the vacated position. A living
  entrant that needs recovery uses it before hazards. Lunar Dance also restores PP.
- Revival Blessing pauses the turn for the user to select a fainted party slot
  through the existing party action columns. The selected Pokémon returns at half
  HP and the queued actions resume from the saved cursor.

#### Item lifecycle

- Trick and Switcheroo swap items. Thief and Covet transfer a removable item to
  an empty slot. Knock Off and Incinerate remove eligible items. Transfer and
  destruction leave the Pokémon's Recycle history intact. Item removal on a
  damaging move runs before the target checks whether to eat a healing berry.
- Bug Bite and Pluck eat the target's berry without changing the user's held
  item. Stuff Cheeks boosts Defense and then eats the user's berry. Teatime
  checks each active Pokémon separately and forces eligible holders to eat.
- Fling derives power and its payload from the active item and records the spent
  item. Recycle restores the Pokémon's most recent recoverable item when its
  item slot is empty. Recovery history follows the party slot across switches.
- Item effects are limited to represented items and abilities. Additional berry
  effects and Belch's prior-berry prerequisite remain separate work.

#### Ability, type, form and moveset mutation

- Skill Swap, Role Play, Entrainment, Simple Beam, Worry Seed and Gastro Acid
  share ability identity, eligibility and lifecycle hooks. Trace, Imposter,
  ability-changing contact reactions and Neutralizing Gas use those same
  effective-ability readers. Ability Shield protects its holder from ability
  replacement and suppression, including with Klutz.
- Soak, Magic Powder, Reflect Type, Conversion and Conversion 2 replace or copy
  typing. Forest's Curse contributes an added type to matchups. Double Shock
  removes Electric after a successful hit. Color Change applies after the attack.
- Power Trick, Guard Split, Power Split and Speed Swap change raw stats.
  Psych Up and the stage-swap moves work on stages independently. Baton Pass
  transfers Power Trick to the incoming Pokémon's own stats.
- Transform and Imposter copy combat identity, stages and move slots. Transform
  caps copied PP at five, Mimic uses the copied move's base PP, and Sketch keeps
  the copy for the rest of the battle. Temporary copies restore their original
  slots when the user leaves.
- Hunger Switch, Relic Song and Gulp Missile use switch-cleared forms. Disguise,
  Ice Face, Shields Down, Zero to Hero and Tera Shift use the lifetime of their
  own form rules. Plates set Multitype typing. Griseous Core, Adamant Crystal,
  Lustrous Globe, Rusted Sword and Rusted Shield apply the matching bound forms.
- Illusion and player-selected transformations still need their own lifecycle
  and disclosure rules. Move and ability metadata outside the current catalogue
  may need additional rules for other generations.

#### Custom stats, healing and costs

- Belly Drum pays its HP cost before setting Attack. Stockpile tracks its
  actual Defense and Special Defense changes separately from the count.
  Swallow and Spit Up consume those layers and remove their recorded stages.
- Strength Sap captures staged target Attack and separately applies the Attack
  drop. Synthesis, Moonlight, Morning Sun and Shore Up read effective weather
  at execution.

#### Special damage outcomes

Team construction projects the catalogue names into scalar `DamageRule` and
`MoveHpCost` metadata. The move attempt carries those rules through the existing
accuracy, protection, hit, and fainting paths. The rules add no persistent battle
state.

| Moves | Represented singles behavior |
| --- | --- |
| Final Gambit | After a hit passes its gates, snapshot the user's current HP and record the user's faint before applying that fixed damage to the target or its Substitute. |
| Fissure, Guillotine, Horn Drill, Sheer Cold | Use level-based accuracy independent of accuracy and evasion stages, then commit max-HP damage through normal recipient and survival handling. Higher-level targets, ordinary immunities, and Sturdy block the hit. Sturdy is revealed when it blocks the move. Sheer Cold also rejects Ice targets and starts at 20% accuracy for a non-Ice user. |
| False Swipe | Cap damage against the Pokémon at its current HP minus one. Contact effects can still trigger at one HP, and a Substitute takes the uncapped hit. |
| Present | Sample one keyed outcome during move preparation. Heal the target by one quarter of its max HP with 20% probability, or use power 40, 80, or 120 with 40%, 30%, and 10% probability respectively. The heal can reach the Pokémon behind a Substitute. |
| Steel Beam and Chloroblast | Pay half maximum HP rounded up after dealing damage, including damage to a Substitute. Steel Beam also pays after a failed hit. An interruption before the attempt pays neither cost. Magic Guard prevents both costs, and Rock Head prevents Chloroblast's recoil. |

#### Field and accuracy conditions

- Gravity, Magic Room and Wonder Room use field timers. Safeguard and Mist
  belong to the side. Magnet Rise and Focus Energy belong to the active
  Pokémon. Lock-On stores the target's team-slot identity and expires when
  either Pokémon leaves.
- Pass each live condition into the relevant grounding, item, defense, status,
  stat, accuracy or critical-hit query. Verify starting and ending mid-turn,
  reapplication, switch cleanup and interaction with already-queued actions.
  Include Gravity canceling airborne continuations and Magic Room restoring
  previously suppressed item effects when it ends.

#### Calls and queue manipulation

- Build called moves on a shared execution entry point with explicit caller,
  selected move, PP policy, targeting and recursion limits. Resolve Metronome
  from a reviewed eligibility pool, Copycat from move history, and Instruct
  from the affected actor's eligible prior action.
- Keep queue edits in `battle.scheduler`. After You and Quash should change
  the order of pending actions while preserving actor identity and the
  execution cursor. Upper Hand and Thunderclap share a queued-attack
  predicate with Sucker Punch. Verify ties, acted and replaced targets, called
  moves that fail, PP use and the ordering of subsequently queued actions.

#### Doubles targeting and combinations

- Extend active indices and queued actors to include a fixed active-position
  axis. Add target coordinates and player-relative action masks with one
  choice per active position. Keep independent battles on the external
  `vmap` axis.
- Resolve retargeting and redirection before generating the hit plan. Keep
  per-target hit results and apply spread reduction and Wide Guard where
  they belong. Add Helping Hand, Ally Switch, Dragon Darts and coordinated
  pledges after targeting works. Verify side symmetry, fainted or switched
  targets, ally protection and singles behavior with one active position.

#### Transformations

- Ordinary-type Tera shares eligibility between typed action validation and
  masks. The scheduler resolves it after switches and before move actions.
  The transformed Pokémon retains its original offensive typing for STAB
  and uses its Tera type for defense. Tera Blast uses the Tera type and the
  better staged attacking stat after transformation.
- Stellar typing, species-specific Tera forms and Tera Starstorm remain open.
  Mega, Z-Move and Dynamax need their own ruleset support.

Supported means the represented singles effect exists. Known cross-cutting
gaps include status-failure classification, residual ordering, ability/item
interactions and full-environment batching acceptance.

Actions now use `BattleAction(index, move_type)` with `ActionType.ATTACK`,
`SWITCH`, `CONTINUED`, and `NO_OP`. Queued actions retain this PyTree. The
continued kind forces a charging move's release or a recharge turn;
inactive no-op remains separate. Attack index 4 represents
forced Struggle. Legacy integer inputs and the existing flat legal-action mask
are retained as compatibility adapters.
An independent `transformation` field uses `Transformation.NONE`, `MEGA`,
`Z_MOVE`, `DYNAMAX`, or `TERA`. The typed Tera action is available where
the ordinary-type singles eligibility mask permits it.

Roost uses a turn-scoped, switch-cleared volatile flag. The effective active
snapshot removes Flying without mutating stored types; pure Flying becomes
Normal, while Terastallized users retain their type. The flag expires after
end-of-turn effects and does not override abilities such as Levitate.

Explosion and Memento use a shared direct-faint helper. Explosion saves the
user's pre-faint HP for damage calculation and commits its faint before target
resolution; Damp prevents it. Memento uses `HitResult` to require reaching the
target, then attempts its stat drops and directly faints even if they fail.

Detect and Spiky Shield share Protect's consecutive-use counter. The scheduler
provides the remaining-action check, and hit routing applies each active
protection mode's blocking and contact rules.

Fake Out counts move-action attempts in switch-cleared state. The scheduler
captures the queued actor's team slot and the pending move category for Sucker
Punch, including Struggle and charging continuations. Stomping Tantrum reads
the previous turn's `MoveResult`; at the turn boundary the current result is
copied and reset. `HitOutcome.PROTECTED` is distinct from a failed or immune
attempt. Status callbacks report represented failure conditions, though not all
status move failures are classified yet.

HP commits queue a faint on the first transition to zero HP. The singles
resolver consumes pending records in order at action, switch, and residual
checkpoints. Records capture the source side and team slot, target slot, cause,
and Destiny Bond state independently of the public event buffer. Destiny Bond
can append a direct faint for a surviving opposing attacker. Moxie and Beast
Boost use the recorded move source after faint resolution. Innards Out and
Aftermath run as hit reactions. Winner selection uses the last recorded faint
when both sides run out of Pokémon at the same checkpoint.

Substitute uses switch-cleared `VolatileStatus.substitute_hp`. A move's most
recent attempts are retained in a fixed ten-entry `HitResults` buffer, with
intended and actual side/active-position coordinates, outcome, recipient and
actual HP removed. Singles always uses active position zero and has no
redirection yet. Doubles target selection should resolve Follow Me and Rage
Powder before accuracy and carry both target coordinates into each attempt;
spread moves will need capacity beyond the present ten-hit singles buffer.
The buffer is internal and is replaced by the next executed move, rather than
being an observation or a complete turn history.

## Latest shared execution support

Solar Beam, Solar Blade, Meteor Beam, Sky Attack, Freeze Shock and Ice Burn
share Electro Shot's charge and forced-release continuation. Sun skips the
solar charge; rain skips Electro Shot's. Meteor Beam and Electro Shot raise
Special Attack before charging or attacking. Power Herb skips a charge when
weather has not already done so. Rain, sand and snow halve solar power at hit
time. Fly, Bounce, Dig, Dive, Phantom Force and Shadow Force use the same
continuation with their respective semi-invulnerable locations.
The switch-cleared semi-invulnerability enum distinguishes airborne, underground,
underwater and vanished users. Incoming attacks use location-specific hit
exceptions, and No Guard bypasses the location check. Phantom Force and Shadow
Force remove represented protection on a reaching hit. The represented
vulnerable-move damage modifiers, Dig/Dive sandstorm shelter, and Fly/Bounce
Gravity restriction are implemented. Poison-type Toxic bypasses the location
and accuracy checks. Additional interactions such as Cramorant's Gulp Missile
and Sky Drop's carried target remain outside this coverage.

Beat Up constructs a fixed-shape hit plan from participating party members in
team order. The user contributes even when statused; other contributors must
be alive and status-free. Each hit's power is 5 plus its contributor's species
base Attack divided by ten, rounded down. The actual user supplies offensive
stats and modifiers.

Sleep Talk uses the shared selected-move executor without spending the called
move's PP or repeating the user's before-action checks. Eligible selection
ignores called-move PP and Disable, but excludes Sleep Talk, charge moves and
metadata marked `nosleeptalk`. Sleep or effective Comatose permits execution;
waking before using it makes it fail. Called moves are publicly revealed and
multi-hit calls can continue while asleep.

Whirlwind selects a living participating bench member with an explicit PRNG
key and runs the normal switch lifecycle. Queued actions retain their original
actor slot, so the displaced Pokémon's action is cancelled. The represented
Wind Rider, Suction Cups and Guard Dog checks prevent the forced switch.

No original curated move remains unavailable, but additional forced-switch interactions,
status-failure classification, ability/item interactions
and new charge/semi-invulnerable variants still need work. The focused tests do
not establish full parity or a completed compiled acceptance pass. See
[mechanics progress](curated-mechanics-progress.md).

## Disable, Encore and Taunt: expanded options

The implemented choice is Option A in `VolatileStatus`, with shared helpers in
`restrictions.py`. The following options preserve the original design rationale.
Both options need the same selection-time and execution-time behavior. Neither
requires a new scheduler.

### Option A: dedicated typed fields

Put the following in switch-cleared `VolatileStatus`, using `(2, 1)` leaves:

```text
last_move_id          # 0 means no applicable move used since entry
entry_turn            # set when this Pokémon becomes active
disabled_move_id
disable_turns
encore_move_id
encore_turns
taunt_turns
```

All identifiers are engine move IDs; counters use zero for inactive. IDs are preferable to slots for future
move replacement/Transform support; resolve an ID to a current move slot when
needed. Recording `last_move_id` must occur when a move is actually used, not
when an action is merely selected or stopped before execution. A used move that
misses can still become the relevant last move.

Shared pure helpers apply restrictions, construct masks, resolve Encore
overrides, check restrictions before PP consumption, and advance
or clear timers. Existing switch-reset handling clears these fields. The current
Encore lock is held in volatile state rather than a second lock on the Pokémon.

Advantages: named, documented fields; straightforward tracing and debugging;
small implementation; easy focused tests. Cost: more named fields and explicit
lifecycle code as condition coverage grows. These fields remain JAX-compatible;
a small number of extra PyTree leaves does not require a new representation.

### Option B: an indexed condition table

Use fixed-shape arrays such as:

```text
condition_turns       (2 players, N condition types)
condition_move_id     (2 players, N condition types)
```

An enum assigns columns for Disable, Encore and Taunt. Zero turns means
inactive. Taunt's move-ID column is unused. Shared functions set, clear, expire,
and inspect a condition; move-specific rules still interpret its parameters.
Arrays stay fixed-size while tracing; adding a condition type changes their
static shape and requires recompilation, not dynamic insertion.

Advantages: consistent lifecycle operations, convenient bulk clearing, and a
natural representation for future condition observations. Cost: unused payload
cells, weaker semantic names, and more generic infrastructure. A table does not
remove the need for special Encore action overrides or different duration rules.
Do not use dynamic Python dictionaries of active conditions inside the engine.
Last-move history and entry timing are still separate fields/context under
this option; they are not active conditions themselves.

### Integration required by either option

1. **Application and history.** Disable and Encore refer to an eligible last
   move, verify its continued availability/PP, and apply their failure rules.
   Taunt applies a category restriction. Move eligibility belongs in metadata
   or shared rule helpers, not inferred solely from damage dealt.
2. **Selection-time mask.** Disable removes the matching move; Encore removes
   the other moves; Taunt removes Status moves. Intersect these with PP, Choice
   locks, Assault Vest, and other restrictions. If no moves remain, advertise
   Struggle while keeping any otherwise legal switches available.
3. **Execution-time check.** After an Encore override, check the resolved move
   against current Disable/Taunt state before consuming PP. A move blocked by
   a newly applied restriction fails; do not silently pick another move. Keep
   the turn's existing action order when Encore changes the move.
4. **Queue context and expiry.** Derive a small context from `EpisodeState`:
   whether the target still has a pending move and whether it entered this
   turn. Use that context to initialize durations with Scarlet/Violet timing.
   Blindly starting identical counters and decrementing them each turn would
   produce off-by-one errors. Clear conditions on switching, expire them in
   the appropriate residual order, and end Encore early when its move runs
   out of PP or becomes unavailable.
5. **Items, abilities and observations.** Support Mental Herb's immediate cure
   of these represented conditions, and applicable immunity checks such as
   Oblivious/Aroma Veil for Taunt. Expose public active-condition information;
   keep the opponent's queued choice private. Preserve any caller-specified
   static `Move.disabled` data rather than overwriting it with temporary flags.

### Recommendation

Use Option A behind a shared restriction interface. It matches the current
`VolatileStatus` design and implements these three moves without introducing a
condition framework. The pure helper interface keeps a later table migration
local: masks and action execution would not need to know how effects are stored.

Focused restriction tests cover application timing, switching, PP exhaustion,
Mental Herb, locks and queued overrides. Full acceptance of the latest engine
under `jit`/`vmap`/`scan` remains outstanding.

## Item mutation helpers

`side.item_effects.remove_item(state, holder, source, enabled=True)` and
`swap_items(state, source, target, enabled=True)` return `(state, success)`.
Failures leave all state unchanged; callers handle move failure messages and
revealing an ability that blocked the effect. `can_remove_item` checks actual
removal including effective Sticky Hold. `item_transfer_allowed` checks only
item identity and holder binding: a damage-power rule can use it independently
of Sticky Hold. Unknown item IDs fail closed.

Swaps validate both outgoing and incoming bindings before either assignment,
then update public item knowledge and events, clean up Choice locks, and run
represented received-item effects (Sitrus Berry, Mental Herb and Booster Energy).
Empty-handed exchanges are valid when at least one item exists. Same-item swaps
are valid and reveal both items. Mask IDs enforce Ogerpon transfer restrictions
and matching-form power modifiers; Wellspring Mask is enabled in the curated
item projection. Absorb Bulb joins Sitrus Berry, Mental Herb and Booster Energy
among the represented received-item effects, activating after a subsequent
damaging Water hit rather than on receipt. Consumption remains a separate
operation.
