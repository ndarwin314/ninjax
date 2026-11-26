"""Move restriction state, hit effects, and queued-action behavior."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.battle.actions import step_action, step_move
from ninjax.battle.environment import Battle
from ninjax.battle.residuals import step_side_conditions
from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.field import Terrain, TerrainEnum
from ninjax.enums.items import ItemEnum as I
from ninjax.enums.pokemon import Status
from ninjax.moves.routing import move_not_drawn_in
from ninjax.moves.status import do_status_move
from ninjax.restrictions import allowed_moves, resolve_move, tick_hit_restrictions
from ninjax.side.hp_effects import take_damage_value
from ninjax.side.updates import clear_volatile_status, update_active
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


@pytest.fixture(scope='module')
def route_restriction_hit():
    """Compile the shared hit path for both restriction moves."""
    return jax.jit(move_not_drawn_in)


@pytest.mark.parametrize('name,blocked_move,timer', [
    ('Throat Chop', 'Hyper Voice', 'throat_chop_turns'),
    ('Psychic Noise', 'Recover', 'heal_block_turns'),
])
def test_damaging_restriction_is_visible_and_blocks_selection(
        neutral_type, route_restriction_hit, name, blocked_move, timer):
    """A successful hit sets the timer and disables just the affected move slots."""
    key, state = neutral_type
    state = equip_move(equip_move(state, name), blocked_move, 1)
    env = Battle(teams=state.team)
    episode, _ = env.reset(key)
    assert env.legal_action_mask(episode)[1, 0]
    _, out = route_restriction_hit(key, episode.battle, 0, 0, 0)
    assert getattr(out.volatile_status, timer)[1, 0] == 2
    restricted = episode.replace(battle=out)
    mask = env.legal_action_mask(restricted)
    assert not mask[1, 0] and mask[1, 1]
    assert getattr(env.observe(restricted).active, timer)[1, 0] == 2
    before_pp = out.active.moves.current_pp[1, 0, 0]
    _, skipped = step_move(key, out, 1, 0, False)
    assert skipped.active.moves.current_pp[1, 0, 0] == before_pp
    assert getattr(clear_volatile_status(out, 1).volatile_status, timer)[1, 0] == 0


@pytest.mark.parametrize('name,ability,item,sub_hp,expected', [
    ('Psychic Noise', A.NONE, I.SITRUS_BERRY, 100, 2),
    ('Psychic Noise', A.SHIELD_DUST, I.SITRUS_BERRY, 0, 0),
    ('Psychic Noise', A.AROMA_VEIL, I.SITRUS_BERRY, 0, 0),
    ('Psychic Noise', A.NONE, I.MENTAL_HERB, 0, 0),
    ('Throat Chop', A.NONE, I.NONE, 100, 0),
    ('Throat Chop', A.AROMA_VEIL, I.MENTAL_HERB, 0, 2),
])
def test_damaging_restriction_hit_gates(neutral_type, route_restriction_hit, name, ability, item, sub_hp, expected):
    """Apply the secondary only when the hit and the target allow it."""
    key, state = neutral_type
    state = equip_move(state, name)
    mon = state.active[1]
    state = update_active(state, 1, mon.replace(
        current_hp=jnp.asarray([mon.max_hp * 55 // 100]), ability=jnp.asarray([ability]), item=jnp.asarray([item])))
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(substitute_hp=v.substitute_hp.at[1, 0].set(sub_hp)))
    _, out = route_restriction_hit(key, state, 0, 0, 0)
    timer = out.volatile_status.heal_block_turns if name == 'Psychic Noise' else out.volatile_status.throat_chop_turns
    assert timer[1, 0] == expected
    if item == I.SITRUS_BERRY:
        assert (out.active.item[1, 0] == I.SITRUS_BERRY) == bool(expected)
        if expected:
            hp = out.active.current_hp[1, 0]
            assert hp * 2 <= mon.max_hp
            assert take_damage_value(out, 1, jnp.array([-10]), False).active.current_hp[1, 0] == hp
            order = jnp.arange(2)
            out = tick_hit_restrictions(out, order)
            assert out.volatile_status.heal_block_turns[1, 0] == 1
            assert out.active.item[1, 0] == I.SITRUS_BERRY
            out = tick_hit_restrictions(out, order)
            assert out.active.item[1, 0] == I.NONE
            assert out.active.current_hp[1, 0] > hp
    if item == I.MENTAL_HERB:
        assert (out.active.item[1, 0] == I.MENTAL_HERB) == (name == 'Throat Chop')


def test_taunt_blocks_queued_status_without_pp(neutral_type):
    """A newly applied Taunt blocks the selected action before PP is spent."""
    key, state = neutral_type
    state = equip_move(state, 'Recover', 1)
    _, state = do_status_move(key, state, 0, make_move('Taunt'), 0, 0)
    assert not allowed_moves(state.active.moves.identifier[..., 0], state.active.moves.move_type[..., 0],
        disabled_move_id=state.volatile_status.disabled_move_id,
        disable_turns=state.volatile_status.disable_turns,
        taunt_turns=state.volatile_status.taunt_turns,
        encore_move_id=state.volatile_status.encore_move_id,
        encore_turns=state.volatile_status.encore_turns,
        gravity_duration=state.gravity_duration)[1, 0]
    before = state.active.moves.current_pp[1, 0, 0]
    _, out = step_move(key, state, 1, 0, False)
    assert out.active.moves.current_pp[1, 0, 0] == before
    assert out.volatile_status.taunt_turns[1, 0] == 3
    assert clear_volatile_status(out, 1).volatile_status.taunt_turns[1, 0] == 0


def test_encore_override_and_disable_conflict(neutral_type):
    """Encore redirects an already chosen move; Disable can block the result."""
    key, state = neutral_type
    state = equip_move(state, 'Recover', 1)
    state = state.replace(volatile_status=state.volatile_status.replace(last_move_id=state.volatile_status.last_move_id.at[1, 0].set(105),
        pending_move=state.volatile_status.pending_move.at[1, 0].set(True)))
    _, state = do_status_move(key, state, 0, make_move('Encore'), 0, 0)
    assert resolve_move(state.active.moves.identifier[1, :, 0], state.active.moves.move_type[1, :, 0], state.active.moves.current_pp[1, :, 0], 1,
        disabled_move_id=state.volatile_status.disabled_move_id[1, 0],
        disable_turns=state.volatile_status.disable_turns[1, 0],
        taunt_turns=state.volatile_status.taunt_turns[1, 0],
        encore_move_id=state.volatile_status.encore_move_id[1, 0],
        encore_turns=state.volatile_status.encore_turns[1, 0],
        gravity_duration=state.gravity_duration)[0] == 0
    assert jnp.array_equal(allowed_moves(state.active.moves.identifier[..., 0], state.active.moves.move_type[..., 0],
        disabled_move_id=state.volatile_status.disabled_move_id,
        disable_turns=state.volatile_status.disable_turns,
        taunt_turns=state.volatile_status.taunt_turns,
        encore_move_id=state.volatile_status.encore_move_id,
        encore_turns=state.volatile_status.encore_turns,
        gravity_duration=state.gravity_duration)[1], jnp.array([True, False, False, False]))
    state = state.replace(volatile_status=state.volatile_status.replace(disabled_move_id=state.volatile_status.disabled_move_id.at[1, 0].set(105),
        disable_turns=state.volatile_status.disable_turns.at[1, 0].set(3)))
    assert resolve_move(state.active.moves.identifier[1, :, 0], state.active.moves.move_type[1, :, 0], state.active.moves.current_pp[1, :, 0], 1,
        disabled_move_id=state.volatile_status.disabled_move_id[1, 0],
        disable_turns=state.volatile_status.disable_turns[1, 0],
        taunt_turns=state.volatile_status.taunt_turns[1, 0],
        encore_move_id=state.volatile_status.encore_move_id[1, 0],
        encore_turns=state.volatile_status.encore_turns[1, 0],
        gravity_duration=state.gravity_duration)[1]
    assert not jnp.any(allowed_moves(state.active.moves.identifier[..., 0], state.active.moves.move_type[..., 0],
        disabled_move_id=state.volatile_status.disabled_move_id,
        disable_turns=state.volatile_status.disable_turns,
        taunt_turns=state.volatile_status.taunt_turns,
        encore_move_id=state.volatile_status.encore_move_id,
        encore_turns=state.volatile_status.encore_turns,
        gravity_duration=state.gravity_duration)[1])
    before = state.active.moves.current_pp[1, 0, 0]
    _, out = step_move(key, state, 1, 1, False)
    assert out.active.moves.current_pp[1, 0, 0] == before


def test_disable_history_and_mental_herb(neutral_type):
    """Disable needs a usable last move and Mental Herb cures all restrictions."""
    key, state = neutral_type
    _, failed = do_status_move(key, state, 0, make_move('Disable'), 0, 0)
    assert failed.volatile_status.disable_turns[1, 0] == 0
    state = equip_move(state, 'Recover', 1)
    state = state.replace(volatile_status=state.volatile_status.replace(last_move_id=state.volatile_status.last_move_id.at[1, 0].set(105)))
    _, disabled = do_status_move(key, state, 0, make_move('Disable'), 0, 0)
    assert disabled.volatile_status.disabled_move_id[1, 0] == 105
    assert not allowed_moves(disabled.active.moves.identifier[..., 0], disabled.active.moves.move_type[..., 0],
        disabled_move_id=disabled.volatile_status.disabled_move_id,
        disable_turns=disabled.volatile_status.disable_turns,
        taunt_turns=disabled.volatile_status.taunt_turns,
        encore_move_id=disabled.volatile_status.encore_move_id,
        encore_turns=disabled.volatile_status.encore_turns,
        gravity_duration=disabled.gravity_duration)[1, 0]
    state = update_active(state, 1, state.active[1].replace(item=jnp.array([I.MENTAL_HERB])))
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(taunt_turns=v.taunt_turns.at[1, 0].set(2),
        encore_turns=v.encore_turns.at[1, 0].set(2), encore_move_id=v.encore_move_id.at[1, 0].set(105)))
    _, cured = do_status_move(key, state, 0, make_move('Disable'), 0, 0)
    assert cured.volatile_status.disable_turns[1, 0] == 0
    assert cured.volatile_status.taunt_turns[1, 0] == cured.volatile_status.encore_turns[1, 0] == 0
    assert cured.active.item[1, 0] == I.NONE
    assert cured.known_items[1, 0] == I.NONE


def test_duration_and_encore_pp_expiry(neutral_type):
    """Timers account for a pending action and Encore ends when PP is gone."""
    key, state = neutral_type
    state = equip_move(state, 'Recover', 1)
    v = state.volatile_status.replace(last_move_id=state.volatile_status.last_move_id.at[1, 0].set(105), pending_move=state.volatile_status.pending_move.at[1, 0].set(True))
    state = state.replace(volatile_status=v)
    _, state = do_status_move(key, state, 0, make_move('Encore'), 0, 0)
    assert state.volatile_status.encore_turns[1, 0] == 3
    mon = state.active[1]
    state = update_active(state, 1, mon.replace(moves=mon.moves.replace(current_pp=mon.moves.current_pp.at[0, 0].set(0))))
    _, state = step_side_conditions(key, state)
    assert state.volatile_status.encore_turns[1, 0] == 0


def test_environment_masks_and_public_observation(neutral_type):
    """Masks combine restrictions with switch availability and expose effects."""

    key, state = neutral_type
    state = equip_move(state, 'Recover', 0)
    env = Battle(teams=state.team)
    episode, _ = env.reset(key)
    v = episode.battle.volatile_status
    v = v.replace(taunt_turns=v.taunt_turns.at[0, 0].set(2))
    episode = episode.replace(battle=episode.battle.replace(volatile_status=v))
    mask = env.legal_action_mask(episode)
    assert not mask[0, 0] and jnp.any(mask[0, 1:4]) and jnp.any(mask[0, 8:14])
    observation = env.observe(episode)
    assert observation.active.taunt_turns[0, 0] == 2
    assert observation.active.taunt_turns[1, 1] == 2


@pytest.mark.parametrize('restriction,chosen', [('Taunt', 0), ('Disable', 0), ('Encore', 1)])
def test_faster_restriction_changes_pending_action(neutral_type, restriction, chosen):
    """The second actor uses its current restrictions, not just the old mask."""

    key, state = neutral_type
    state = equip_move(state, restriction, 0)
    state = equip_move(state, 'Recover', 1)
    table = state.team.stat_table.replace(base_stats=state.team.stat_table.base_stats.at[0, 0, 5].set(150))
    state = state.replace(team=state.team.replace(stat_table=table))
    env = Battle(teams=state.team)
    episode, _ = env.reset(key)
    v = episode.battle.volatile_status
    episode = episode.replace(battle=episode.battle.replace(volatile_status=v.replace(last_move_id=v.last_move_id.at[1, 0].set(105))))
    before = episode.battle.active.moves.current_pp[1, :, 0]
    episode, _ = env.step(episode, jnp.array([0, chosen], jnp.int32))
    after = episode.battle.active.moves.current_pp[1, :, 0]
    if restriction == 'Taunt':
        assert jnp.array_equal(before, after)
        assert episode.battle.volatile_status.taunt_turns[1, 0] == 2
    elif restriction == 'Disable':
        assert jnp.array_equal(before, after)
        assert episode.battle.volatile_status.disable_turns[1, 0] == 3
    else:
        assert after[0] == before[0] - 1 and after[1] == before[1]
        assert episode.revealed_moves[1, 0, 0]


def test_restriction_application_timing_and_immunities(neutral_type):
    """Durations account for pending actions; immunity prevents application."""
    key, state = neutral_type
    state = equip_move(state, 'Recover', 1)
    v = state.volatile_status.replace(last_move_id=state.volatile_status.last_move_id.at[1, 0].set(105))
    state = state.replace(volatile_status=v, turn_number=2)
    for name, field, fast, slow in [('Disable', 'disable_turns', 4, 5), ('Encore', 'encore_turns', 3, 4), ('Taunt', 'taunt_turns', 3, 4)]:
        with_pending = state.replace(volatile_status=v.replace(pending_move=v.pending_move.at[1, 0].set(True)))
        _, first = do_status_move(key, with_pending, 0, make_move(name), 0, 0)
        _, second = do_status_move(key, state, 0, make_move(name), 0, 0)
        assert getattr(first.volatile_status, field)[1, 0] == fast
        assert getattr(second.volatile_status, field)[1, 0] == slow
    mon = state.active[1].replace(ability=jnp.array([A.OBLIVIOUS]))
    immune = update_active(state, 1, mon)
    _, immune = do_status_move(key, immune, 0, make_move('Taunt'), 0, 0)
    assert immune.volatile_status.taunt_turns[1, 0] == 0


def test_encore_preserves_original_priority_for_psychic_terrain(neutral_type):
    """A redirected priority move remains blocked by Psychic Terrain."""

    key, state = neutral_type
    state = equip_move(state, 'Dragon Pulse', 0)
    mon = state.active[0]
    quick = make_move('Quick Attack')
    state = update_active(state, 0, mon.replace(moves=jax.tree.map(lambda bank, move: bank.at[1].set(move), mon.moves, quick)))
    v = state.volatile_status
    state = state.replace(terrain=Terrain(TerrainEnum.PSYCHIC, 5), volatile_status=v.replace(
        encore_turns=v.encore_turns.at[0, 0].set(3), encore_move_id=v.encore_move_id.at[0, 0].set(406)))
    hp = state.active.current_hp[1, 0]
    pp = state.active.moves.current_pp[0, 0, 0]
    _, out = step_move(key, state, 0, 1, False)
    assert out.active.current_hp[1, 0] == hp
    assert out.active.moves.current_pp[0, 0, 0] == pp - 1


def test_encore_rejects_ineligible_last_move(neutral_type):
    """Encore cannot lock itself and Disable cannot target an exhausted move."""
    key, state = neutral_type
    state = equip_move(state, 'Encore', 1)
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(last_move_id=v.last_move_id.at[1, 0].set(227)))
    _, out = do_status_move(key, state, 0, make_move('Encore'), 0, 0)
    assert out.volatile_status.encore_turns[1, 0] == 0
    mon = state.active[1]
    state = update_active(state, 1, mon.replace(moves=mon.moves.replace(current_pp=mon.moves.current_pp.at[0, 0].set(0))))
    _, out = do_status_move(key, state, 0, make_move('Disable'), 0, 0)
    assert out.volatile_status.disable_turns[1, 0] == 0


def test_forced_struggle_respects_flinch_and_history(neutral_type):
    """An interrupted Struggle neither damages nor overwrites the last move."""

    key, state = neutral_type
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(flinched=v.flinched.at[0, 0].set(True),
        last_move_id=v.last_move_id.at[0, 0].set(105)))
    _, out = step_action(key, state, 4, 0)
    assert jnp.array_equal(out.active.current_hp, state.active.current_hp)
    assert out.volatile_status.last_move_id[0, 0] == 105

    state = state.replace(volatile_status=state.volatile_status.replace(flinched=state.volatile_status.flinched.at[0, 0].set(False)))
    state = update_active(state, 0, state.active[0].replace(status=jnp.array([Status.SLEEP]), sleep_counter=jnp.array([2])))
    _, out = step_action(key, state, 4, 0)
    assert out.active.sleep_counter[0, 0] == 1
    assert jnp.array_equal(out.active.current_hp, state.active.current_hp)
