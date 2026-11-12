"""Validation of the remaining curated abilities against pinned Showdown rules.

Expected modifiers and exceptions are taken from data/abilities.ts and the
4096-based battle.modify rule at revision 23b41aa29e2ff5ff80e03f57b257b595a5852047.
This is controlled mechanics validation, not a claim of whole-battle parity.
"""
import jax
import jax.numpy as jnp
import pytest
from unit_tests.test_battle import neutral_type
from ninjax.enum_types import AbilityEnum as A, ItemEnum as I, Type, Status, StatEnum as S, MoveType, Weather, WeatherEnum as W, Terrain, TerrainEnum as T
from ninjax.held_effects import modify_stats, refresh_paradox
from ninjax.game_logic import swap_is_alive, swap_out, effective_move, step_moody, move_hits, end_turn_damage
from ninjax.ability_effects import prepare_type_change
from ninjax.battle import Battle, Phase
from ninjax.damage import compute_base_damage, compute_base_power
from ninjax.prototype import make_move


def ability_state(state, ability, player=0):
    """Assign an ability to the selected side's lead."""
    return state.replace(team=state.team.replace(ability=state.team.ability.at[player, 0, 0].set(ability)))


@pytest.mark.parametrize('ability,stat,weather,terrain', [
    (A.ORICHALCUM_PULSE, S.ATTACK, W.SUN, T.NONE),
    (A.HADRON_ENGINE, S.SPECIAL_ATTACK, W.NONE, T.ELECTRIC)])
def test_field_stat_abilities(neutral_type, ability, stat, weather, terrain):
    """Field-setting abilities boost the proper stat by 5461/4096."""
    _, state = neutral_type
    state = ability_state(state, ability)
    active = swap_is_alive(state, 0)
    assert active.weather.weather == weather and active.terrain.terrain == terrain
    stats = modify_stats(active, 0, jnp.full(6, 100))
    assert stats[stat] == 133
    assert jnp.sum(stats != 100) == 1
    if ability == A.HADRON_ENGINE:
        specs = active.replace(team=active.team.replace(item=active.team.item.at[0,0,0].set(I.CHOICE_SPECS)))
        assert modify_stats(specs, 0, jnp.full(6, 101))[stat] == 202
    clear = state.replace(weather=Weather(W.NONE, 0), terrain=Terrain(T.NONE, 0))
    assert jnp.all(modify_stats(clear, 0, jnp.full(6, 100)) == 100)
    if ability == A.ORICHALCUM_PULSE:
        suppressed = ability_state(active, A.CLOUD_NINE, 1)
        assert modify_stats(suppressed, 0, jnp.full(6, 100))[stat] == 100


@pytest.mark.parametrize('ability,weather,terrain', [
    (A.PROTOSYNTHESIS, W.SUN, T.NONE), (A.QUARK_DRIVE, W.NONE, T.ELECTRIC)])
def test_paradox_lifecycle_and_speed(neutral_type, ability, weather, terrain):
    """Best-stat selection is locked per activation and Booster survives field loss."""
    _, state = neutral_type
    state = ability_state(state, ability)
    state = state.replace(weather=Weather(weather, 2), terrain=Terrain(terrain, 2),
        team=state.team.replace(item=state.team.item.at[0, 0, 0].set(I.BOOSTER_ENERGY)))
    # Stages participate in best-stat choice; HP never does.
    state = state.replace(boosts=state.boosts.replace(normal_boosts=state.boosts.normal_boosts.at[0, S.SPEED].set(2)))
    active = refresh_paradox(state)
    assert active.volatile_status.paradox_stat[0, 0] == S.SPEED
    assert active.active.item[0, 0] == I.BOOSTER_ENERGY
    assert modify_stats(active, 0, jnp.full(6, 100))[S.SPEED] == 150
    changed = active.replace(boosts=active.boosts.replace(normal_boosts=active.boosts.normal_boosts.at[0, S.ATTACK].set(6)))
    assert refresh_paradox(changed).volatile_status.paradox_stat[0, 0] == S.SPEED
    nofield = changed.replace(weather=Weather(W.NONE, 0), terrain=Terrain(T.NONE, 0))
    booster = refresh_paradox(nofield)
    assert booster.active.item[0, 0] == I.NONE
    assert booster.volatile_status.paradox_stat[0, 0] == S.ATTACK
    assert modify_stats(booster, 0, jnp.full(6, 100))[S.ATTACK] == 130
    assert refresh_paradox(booster).volatile_status.paradox_stat[0, 0] == S.ATTACK
    ended = refresh_paradox(active.replace(weather=Weather(W.NONE, 0), terrain=Terrain(T.NONE, 0),
        team=active.team.replace(item=active.team.item.at[0, 0, 0].set(I.NONE))))
    assert ended.volatile_status.paradox_stat[0, 0] == -1


@pytest.mark.parametrize('attacker,defender,category,offense,defense,expected', [
    (A.NONE,A.NONE,0,1,2,69), (A.SWORD_OF_RUIN,A.NONE,0,1,2,91),
    (A.SWORD_OF_RUIN,A.SWORD_OF_RUIN,0,1,2,69),
    (A.BEADS_OF_RUIN,A.NONE,1,3,4,91), (A.BEADS_OF_RUIN,A.BEADS_OF_RUIN,1,3,4,69),
    (A.NONE,A.VESSEL_OF_RUIN,1,3,4,52), (A.VESSEL_OF_RUIN,A.VESSEL_OF_RUIN,1,3,4,69),
    (A.NONE,A.VESSEL_OF_RUIN,0,1,2,69), (A.NONE,A.SWORD_OF_RUIN,0,2,2,52),
    (A.SWORD_OF_RUIN,A.SWORD_OF_RUIN,0,2,2,69),
    (A.SWORD_OF_RUIN,A.NONE,1,3,2,91),  # Psyshock targets Defense.
])
def test_ruin_stat_selection_and_rounding(neutral_type, attacker, defender, category, offense, defense, expected):
    """Ruin uses fixed-point rounding, holder exemptions and selected stats."""
    _, state = neutral_type
    move = state.active[0].moves[0].replace(move_type=jnp.array([category]),
        offensive_stat=jnp.array([offense]), defensive_stat=jnp.array([defense]))
    stats = jnp.full(6, 101)
    damage = compute_base_damage(attacker, defender, 1., 100, stats, stats, move, 80, Status.NONE, Status.NONE)
    assert damage[0] == expected


@pytest.mark.parametrize('ability', [A.LIBERO, A.PROTEAN])
def test_type_change_once_only_on_actual_change(neutral_type, ability):
    """Same-type uses preserve activation; switching resets type and the latch."""
    key, state = neutral_type
    state = ability_state(state, ability)
    mon = state.active[0].replace(type_list=jnp.array([Type.FIRE, Type.NONE]), base_types=jnp.array([Type.FIRE, Type.NONE]))
    from ninjax.side import update_active
    state = update_active(state, 0, mon)
    move = state.active[0].moves[0].replace(type=jnp.array([Type.FIRE]))
    same = prepare_type_change(state, 0, move)
    assert not same.volatile_status.libero_used[0, 0]
    changed = prepare_type_change(same, 0, move.replace(type=jnp.array([Type.GRASS])))
    assert changed.active.type_list[0, 0] == Type.GRASS
    assert changed.volatile_status.libero_used[0, 0]
    again = prepare_type_change(changed, 0, move)
    assert again.active.type_list[0, 0] == Type.GRASS
    switched = swap_out(key, changed, 0, 1)
    assert jnp.array_equal(switched.team.type_list[0, 0], jnp.array([Type.FIRE, Type.NONE]))
    assert not switched.volatile_status.libero_used[0, 0]
    tera = update_active(state, 0, mon.replace(is_terastallized=jnp.array([True])))
    assert prepare_type_change(tera, 0, move.replace(type=jnp.array([Type.GRASS]))).active.type_list[0, 0] == Type.FIRE


def test_pixilate_exceptions_and_slot_integrity(neutral_type):
    """Pixilate converts Normal attacks, except special type-determining moves."""
    _, state = neutral_type
    state = ability_state(state, A.PIXILATE)
    original = state.active[0].moves[0]
    converted = effective_move(state, 0, original)
    assert converted.type[0] == Type.FAIRY and converted.base_power[0] == 96
    assert original.type[0] == Type.NORMAL
    for identifier in (449,718,363,686,546,805,311):
        excluded = effective_move(state, 0, original.replace(identifier=jnp.array([identifier])))
        assert excluded.type[0] == Type.NORMAL and excluded.base_power[0] == 80


def test_moody_normal_caps_and_inactive(neutral_type):
    """Moody changes distinct eligible stats and never boosts HP/accuracy/evasion."""
    key, state = neutral_type
    state = ability_state(state, A.MOODY)
    _, out = step_moody(key, state)
    stages = out.boosts.normal_boosts[0]
    assert jnp.sum(stages == 2) == 1 and jnp.sum(stages == -1) == 1
    assert stages[0] == 0 and jnp.all(out.boosts.acc_boosts == 0)
    low = state.replace(boosts=state.boosts.replace(normal_boosts=state.boosts.normal_boosts.at[0,1:].set(-6)))
    _, out = step_moody(key, low)
    assert jnp.sum(out.boosts.normal_boosts[0,1:] == -4) == 1
    assert jnp.sum(out.boosts.normal_boosts[0,1:] == -6) == 4
    dead = state.replace(team=state.team.replace(current_hp=state.team.current_hp.at[0,0,0].set(0)))
    _, out = step_moody(key, dead)
    assert jnp.all(out.boosts.normal_boosts == 0)


def test_sand_force_power_and_immunity(neutral_type):
    """Sand Force boosts Rock/Ground/Steel and prevents weather chip itself."""
    _, state = neutral_type
    state = ability_state(state, A.SAND_FORCE)
    move = state.active[0].moves[0]
    for typ in (Type.ROCK, Type.GROUND, Type.STEEL, Type.WATER):
        powered = compute_base_power(state.active[0], state.active[1], move.replace(type=jnp.array([typ])), state.terrain, W.SANDSTORM)
        assert powered[0] == (80 if typ == Type.WATER else 104)
    out = end_turn_damage(state.replace(weather=Weather(W.SANDSTORM, 2)))
    assert out.active.current_hp[0,0] == state.active.current_hp[0,0]
    assert out.active.current_hp[1,0] < state.active.current_hp[1,0]


def test_shadow_tag_exemptions_and_replacements(neutral_type):
    """Shadow Tag prevents ordinary switches, not Ghost/holder/pivot/replacement actions."""
    key, battle = neutral_type
    env = Battle(battle.team)
    state, _ = env.reset(key)
    state = state.replace(battle=ability_state(state.battle, A.SHADOW_TAG, 1))
    mask = env.legal_action_mask(state)
    assert not jnp.any(mask[0,8:14]) and jnp.any(mask[0,:4])
    rejected, ts = env.step(state, jnp.array([9,0]))
    assert ts.extras['invalid_action']
    assert jnp.array_equal(rejected.battle.active_index, state.battle.active_index)
    pivot = make_move('U-turn')
    moves = jax.tree.map(lambda x,y: x.at[0,0,0].set(y), state.battle.team.moves, pivot)
    _, pivoted = move_hits(key, state.battle.replace(team=state.battle.team.replace(moves=moves)), 0, 0)
    assert pivoted.switch_requested[0]
    ghost = state.battle.replace(team=state.battle.team.replace(type_list=state.battle.team.type_list.at[0,0].set(jnp.array([Type.GHOST,Type.NONE]))))
    assert jnp.any(env.legal_action_mask(state.replace(battle=ghost))[0,8:14])
    assert jnp.any(env.legal_action_mask(state.replace(battle=ability_state(state.battle,A.SHADOW_TAG)))[0,8:14])
    assert jnp.any(env.legal_action_mask(state.replace(phase=jnp.int32(Phase.REPLACEMENT)))[0,8:14])
    dead = state.battle.replace(team=state.battle.team.replace(current_hp=state.battle.team.current_hp.at[1,0,0].set(0)))
    assert jnp.any(env.legal_action_mask(state.replace(battle=dead))[0,8:14])


@pytest.mark.parametrize('ability', [A.STORM_DRAIN, A.WATER_ABSORB])
def test_water_absorption_and_multi_hit(neutral_type, ability):
    """Water absorption cancels a whole multi-hit move and activates only once."""
    key, state = neutral_type
    state = ability_state(state, ability, 1)
    source = make_move('Surging Strikes')
    moves = jax.tree.map(lambda x,y: x.at[0,0,0].set(y), state.team.moves, source)
    state = state.replace(team=state.team.replace(moves=moves, current_hp=state.team.current_hp.at[1,0,0].set(100)))
    _, out = move_hits(key, state, 0, 0)
    assert out.active.current_hp[0,0] == state.active.current_hp[0,0]
    if ability == A.STORM_DRAIN:
        assert out.active.current_hp[1,0] == 100 and out.boosts.normal_boosts[1,S.SPECIAL_ATTACK] == 1
    else:
        assert out.active.current_hp[1,0] == 100 + state.active.max_hp[1] // 4
    assert jnp.all(out.active.moves.current_pp == state.active.moves.current_pp)
    from ninjax.game_logic import move_used
    from ninjax.events import EventKind
    inaccurate = state.replace(team=state.team.replace(moves=state.team.moves.replace(
        accuracy=state.team.moves.accuracy.at[0,0,0,0].set(0.))))
    _, intercepted = move_used(key, inaccurate, 0, 0)
    assert jnp.array_equal(intercepted.active.current_hp, out.active.current_hp)
    assert not jnp.any(intercepted.events.kind == EventKind.MISS)
    assert intercepted.active.moves.current_pp[0,0,0] == source.current_pp[0] - 1
    heal = source.replace(move_type=jnp.array([MoveType.STATUS]), target_self=jnp.array([True]), healing_fraction=jnp.array([.5]))
    moves = jax.tree.map(lambda x,y: x.at[0,0,0].set(y), state.team.moves, heal)
    healing = state.replace(team=state.team.replace(moves=moves, current_hp=state.team.current_hp.at[0,0,0].set(100)))
    _, healed = move_hits(key, healing, 0, 0)
    assert healed.active.current_hp[0,0] == 100 + state.active.max_hp[0] // 2
    assert healed.active.current_hp[1,0] == 100
    assert jnp.all(healed.boosts.normal_boosts[1] == 0)


def test_friend_guard_has_no_singles_effect(neutral_type):
    """Friend Guard affects allies other than its holder; singles has none."""
    from ninjax.game_logic import do_damaging_move
    key, state = neutral_type
    move = state.active[0].moves[0]
    _, ordinary = do_damaging_move(key, state, 0, move, 0, 0)
    _, guarded = do_damaging_move(key, ability_state(state, A.FRIEND_GUARD,1), 0, move, 0, 0)
    assert jnp.array_equal(ordinary.active.current_hp, guarded.active.current_hp)


@pytest.mark.skip(reason='Unseen Fist cannot be validated end-to-end: Protect/protection state is not implemented')
def test_unseen_fist_requires_protection_integration():
    """Explicitly track the missing mechanic instead of claiming ability parity."""
    pass


@pytest.mark.jit
def test_remaining_ability_hooks_jit_vmap_scan(neutral_type):
    """Exercise newly validated hooks with dynamic abilities and explicit keys."""
    key, state = neutral_type
    abilities = jnp.array([A.ORICHALCUM_PULSE,A.HADRON_ENGINE,A.PROTOSYNTHESIS,A.QUARK_DRIVE,
        A.BEADS_OF_RUIN,A.SWORD_OF_RUIN,A.VESSEL_OF_RUIN,A.LIBERO,A.PROTEAN,A.PIXILATE,
        A.MOODY,A.SAND_FORCE,A.SHADOW_TAG,A.STORM_DRAIN,A.WATER_ABSORB,A.FRIEND_GUARD])
    move = state.active[0].moves[0]
    def activate(key, state, ability):
        """Apply entry, type-change, residual and stat hooks for a selected ability."""
        state = ability_state(state, ability)
        state = refresh_paradox(swap_is_alive(state, 0))
        state = prepare_type_change(state, 0, move)
        key, state = step_moody(key, state)
        return key, state, modify_stats(state, 0, jnp.full(6, 101))
    keys = jax.random.split(key, len(abilities))
    _, batch, stats = jax.jit(jax.vmap(activate, in_axes=(0,None,0)))(keys, state, abilities)
    assert stats.shape == (16,6) and jnp.all(jnp.isfinite(stats))
    assert not jnp.any(batch.events.overflow)
    def transition(carry, ability):
        """Carry an immutable battle through a sequence of ability activations."""
        key, state, stats = activate(carry[0], carry[1], ability)
        return (key, state), stats
    (_, final), history = jax.jit(lambda k,s: jax.lax.scan(transition, (k,s), abilities))(key,state)
    assert history.shape == (16,6) and not final.events.overflow
