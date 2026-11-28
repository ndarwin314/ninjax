"""Move-created residual conditions, durations and active-position lifetimes."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.battle.environment import Battle
from ninjax.battle.field import step_field
from ninjax.battle.residuals import ResidualHPPhase as HP, run_hp_residuals
from ninjax.battle.switching import prepare_forced_switch, reset_switch_effects
from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.items import ItemEnum as I
from ninjax.enums.pokemon import StatEnum as S, Status, Type
from ninjax.moves.routing import move_not_drawn_in
from ninjax.side.residual_conditions import apply_residual_status, tick_syrup_bomb, tick_yawn
from ninjax.side.trapping import clear_source_linked, voluntary_switch_blocked
from ninjax.side.updates import update_active_fields
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


@pytest.fixture(scope='module')
def hit():
    """Share an ordinary compiled move route across residual effects."""
    return jax.jit(move_not_drawn_in)


def test_healing_residuals_independent_of_source_and_passable(neutral_type):
    """Aqua Ring and Ingrain have separate indefinite durations and pass together."""
    _, state = neutral_type
    state = update_active_fields(state, 0, current_hp=jnp.array([100]),
        type_list=jnp.array([Type.FLYING, Type.NONE]))
    for name in ('Aqua Ring', 'Ingrain'):
        state, applied = apply_residual_status(state, 0, make_move(name))
        assert applied
        _, repeated = apply_residual_status(state, 0, make_move(name))
        assert not repeated

    assert state.is_grounded(0)[0] and voluntary_switch_blocked(state)[0]
    passed = reset_switch_effects(state, 0, 1)
    shed = reset_switch_effects(state, 0, 2)
    assert passed.volatile_status.residuals.ingrain[0, 0] == -1
    assert passed.volatile_status.residuals.aqua_ring[0, 0] == -1
    assert shed.volatile_status.residuals.ingrain[0, 0] == 0
    assert shed.volatile_status.residuals.aqua_ring[0, 0] == 0
    out = run_hp_residuals(passed, (HP.AQUA_RING, HP.INGRAIN), jnp.arange(2))
    assert out.active.current_hp[0, 0] == 100 + 2 * (state.active.max_hp[0] // 16)


def test_ingrain_blocks_drag_and_applies_ground_type_matchup(neutral_type):
    """Rooted targets stay in place and lose their airborne Ground immunity."""
    key, state = neutral_type
    state = update_active_fields(state, 1, type_list=jnp.array([Type.FLYING, Type.NONE]))
    ground_move = make_move('Earthquake')
    assert state.is_immune_to_move(1, ground_move)
    rooted, _ = apply_residual_status(state, 1, make_move('Ingrain'))
    _, _, drag = prepare_forced_switch(key, rooted, 0)
    assert not drag.enabled and not rooted.is_immune_to_move(1, ground_move)


def test_leech_seed_uses_the_live_position_and_capped_hp(neutral_type, hit):
    """Seed follows the source position and transfers only the victim's lost HP."""
    key, state = neutral_type
    state = equip_move(state, 'Leech Seed')
    state = update_active_fields(state, 1, current_hp=jnp.array([9]))
    # Leech Seed can miss. Make the focused lifecycle case deterministic.
    moves = state.team.moves
    state = state.replace(team=state.team.replace(moves=moves.replace(
        accuracy=moves.accuracy.at[0, 0, 0, 0].set(1.))))
    _, seeded = hit(key, state, 0, 0, 0)
    assert seeded.volatile_status.residuals.leech_seed[1, 0] == -1
    changed = clear_source_linked(seeded, 0).replace(active_index=seeded.active_index.at[0, 0].set(1))
    changed = update_active_fields(changed, 0, current_hp=jnp.array([100]))
    drained = run_hp_residuals(changed, (HP.LEECH_SEED,), jnp.arange(2))
    assert drained.active.current_hp[1, 0] == 0 and drained.active.current_hp[0, 0] == 109


def test_leech_seed_blocks_grass_and_reverses_with_liquid_ooze(neutral_type):
    """Reject a Grass target and redirect seeded healing against Liquid Ooze."""
    _, state = neutral_type
    move = make_move('Leech Seed')
    grass = update_active_fields(state, 1, type_list=jnp.array([Type.GRASS, Type.NONE]))
    _, success = apply_residual_status(grass, 0, move)
    assert not success
    state = update_active_fields(state, 1, ability=jnp.array([A.LIQUID_OOZE]))
    state = update_active_fields(state, 0, item=jnp.array([I.BIG_ROOT]))
    state, success = apply_residual_status(state, 0, move)
    assert success
    out = run_hp_residuals(state, (HP.LEECH_SEED,), jnp.arange(2))
    drained = state.active.max_hp[1] // 8
    expected = (5324 * drained + 2047) // 4096
    assert out.active.current_hp[0, 0] == state.active.max_hp[0] - expected


@pytest.mark.parametrize('ability, substitute, enabled', [(A.NONE, 0, True), (A.SHIELD_DUST, 0, False), (A.NONE, 50, False)])
def test_salt_cure_secondary_gate(neutral_type, hit, ability, substitute, enabled):
    """Apply Salt Cure only when a secondary reaches the real target."""
    key, state = neutral_type
    state = equip_move(state, 'Salt Cure')
    state = update_active_fields(state, 1, ability=jnp.array([ability]))
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(substitute_hp=v.substitute_hp.at[1, 0].set(substitute)))
    _, out = hit(key, state, 0, 0, 0)
    assert bool(out.volatile_status.residuals.salt_cure[1, 0] != 0) == enabled


@pytest.mark.parametrize('kind, divisor', [(Type.NORMAL, 8), (Type.WATER, 4), (Type.STEEL, 4)])
def test_salt_cure_live_type_and_switch_policy(neutral_type, hit, kind, divisor):
    """Salt Cure uses current typing, survives its source leaving, and cannot pass."""
    key, state = neutral_type
    _, applied = hit(key, equip_move(state, 'Salt Cure'), 0, 0, 0)
    state = update_active_fields(clear_source_linked(applied, 0), 1, type_list=jnp.array([kind, Type.NONE]))
    out = run_hp_residuals(state, (HP.BINDING_AND_SALT,), jnp.arange(2))
    assert out.active.current_hp[1, 0] == state.active.current_hp[1, 0] - state.active.max_hp[1] // divisor
    assert reset_switch_effects(out, 1, 1).volatile_status.residuals.salt_cure[1, 0] == 0


def test_syrup_bomb_three_ticks_and_source_cleanup(neutral_type, hit):
    """Syrup drops Speed on three turns and is cleared if its source departs."""
    key, state = neutral_type
    state = update_active_fields(equip_move(state, 'Syrup Bomb'), 0, ability=jnp.array([A.NO_GUARD]))
    _, applied = hit(key, state, 0, 0, 0)
    assert applied.volatile_status.residuals.syrup_turns[1, 0] == 4
    dropped = applied
    for turn in range(1, 5):
        dropped = tick_syrup_bomb(dropped, jnp.arange(2))
        assert dropped.boosts.normal_boosts[1, S.SPEED] == -min(turn, 3)
    assert dropped.volatile_status.residuals.syrup_turns[1, 0] == 0
    departed = clear_source_linked(applied, 0)
    assert departed.volatile_status.residuals.syrup_turns[1, 0] == 0
    assert departed.boosts.normal_boosts[1, S.SPEED] == 0


def test_curse_builds_two_different_effects(neutral_type, hit):
    """Non-Ghost Curse changes stages, and Ghost Curse bypasses Substitute."""
    key, state = neutral_type
    state = equip_move(state, 'Curse')
    _, ordinary = hit(key, state, 0, 0, 0)
    assert list(ordinary.boosts.normal_boosts[0, 1:]) == [1, 1, 0, 0, -1]
    ghost = update_active_fields(state, 0, type_list=jnp.array([Type.GHOST, Type.NONE]))
    v = ghost.volatile_status
    ghost = ghost.replace(volatile_status=v.replace(substitute_hp=v.substitute_hp.at[1, 0].set(20)))
    _, cursed = hit(key, ghost, 0, 0, 0)
    assert cursed.volatile_status.residuals.curse[1, 0] == -1
    assert cursed.active.current_hp[0, 0] == ghost.active.max_hp[0] - ghost.active.max_hp[0] // 2
    out = run_hp_residuals(cursed, (HP.CURSE,), jnp.arange(2))
    assert out.active.current_hp[1, 0] == ghost.active.max_hp[1] - ghost.active.max_hp[1] // 4


@pytest.mark.parametrize('name, self_trapped', [('Mean Look', False), ('Block', False), ('Spirit Shackle', False), ('Jaw Lock', True)])
def test_linked_traps_block_choices_until_source_leaves(neutral_type, hit, name, self_trapped):
    """Non-damaging and damaging locks share the same source-bound lifetime."""
    key, state = neutral_type
    _, locked = hit(key, equip_move(state, name), 0, 0, 0)
    assert list(voluntary_switch_blocked(locked)) == [self_trapped, True]
    assert not voluntary_switch_blocked(clear_source_linked(locked, 0))[1]


def test_yawn_rechecks_immunity_and_clears_on_switch(neutral_type):
    """Yawn expires after two checkpoints and uses live sleep immunity."""
    key, state = neutral_type
    drowsy, success = apply_residual_status(state, 0, make_move('Yawn'))
    assert success
    key, once = tick_yawn(key, drowsy, jnp.arange(2))
    assert once.active.status[1, 0] == Status.NONE
    _, slept = tick_yawn(key, once, jnp.arange(2))
    assert slept.active.status[1, 0] == Status.SLEEP
    immune = update_active_fields(once, 1, ability=jnp.array([A.INSOMNIA]))
    _, out = tick_yawn(key, immune, jnp.arange(2))
    assert out.active.status[1, 0] == Status.NONE
    assert reset_switch_effects(once, 1, 1).volatile_status.residuals.yawn_turns[1, 0] == 0


def test_field_ticks_wish_and_fairy_lock(neutral_type):
    """Wish follows a replacement and Fairy Lock lasts through the next turn."""
    key, state = neutral_type
    state, wish = apply_residual_status(state, 0, make_move('Wish'))
    state, lock = apply_residual_status(state, 1, make_move('Fairy Lock'))
    assert wish and lock
    step = jax.jit(step_field)
    key, once = step(key, state)
    assert once.slot_residuals.wish_turns[0, 0] == 1
    assert once.fairy_lock_turns == 1 and jnp.all(voluntary_switch_blocked(once))
    entrant = once.replace(active_index=once.active_index.at[0, 0].set(1))
    entrant = update_active_fields(entrant, 0, current_hp=jnp.array([1]))
    _, resolved = step(key, entrant)
    assert resolved.active.current_hp[0, 0] == 1 + state.active.max_hp[0] // 2
    assert resolved.slot_residuals.wish_turns[0, 0] == 0 and resolved.fairy_lock_turns == 0


def test_no_retreat_traps_and_cannot_repeat(neutral_type):
    """No Retreat raises all five stats and blocks another use while trapped."""
    _, state = neutral_type
    move = make_move('No Retreat')
    trapped, applied = apply_residual_status(state, 0, move)
    assert applied and voluntary_switch_blocked(trapped)[0]
    assert list(trapped.boosts.normal_boosts[0, 1:]) == [1, 1, 1, 1, 1]
    _, repeated = apply_residual_status(trapped, 0, move)
    assert not repeated
    ghost = update_active_fields(trapped, 0, type_list=jnp.array([Type.GHOST, Type.NONE]))
    assert not voluntary_switch_blocked(ghost)[0]


def test_residual_observation_spec(neutral_type):
    """Public timers follow the player-relative observation contract."""
    key, state = neutral_type
    env = Battle(state.team)
    episode, _ = env.reset(key)
    battle, _ = apply_residual_status(episode.battle, 0, make_move('Aqua Ring'))
    battle, _ = apply_residual_status(battle, 1, make_move('Yawn'))
    obs = env.observe(episode.replace(battle=battle))
    env.observation_spec.validate(obs)
    assert list(obs.active.aqua_ring[:, 0]) == [True, False]
    assert list(obs.active.yawn_turns[:, 0]) == [2, 0]


@pytest.mark.jit
def test_yawn_through_public_step_and_replacement(neutral_type):
    """The scheduler ticks a selected Yawn and clears it on a legal switch."""
    key, state = neutral_type
    state = equip_move(equip_move(state, 'Yawn'), 'Growl', 1)
    env = Battle(state.team)
    episode, _ = env.reset(key)
    step = jax.jit(env.step)
    drowsy, _ = step(episode, jnp.array([0, 0]))
    assert drowsy.battle.volatile_status.residuals.yawn_turns[1, 0] == 1
    switched, _ = step(drowsy, jnp.array([1, 9]))
    assert switched.battle.active_index[1, 0] == 1
    assert switched.battle.volatile_status.residuals.yawn_turns[1, 0] == 0
    assert switched.battle.active.status[1, 0] == Status.NONE
