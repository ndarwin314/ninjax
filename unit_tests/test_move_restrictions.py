"""Disable, Encore and Taunt state and queued-action behavior."""
import jax
import jax.numpy as jnp
import pytest
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move
from ninjax.prototype import make_move
from ninjax.game_logic import do_status_move, step_side_conditions
from ninjax.battle import step_move
from ninjax.restrictions import allowed_moves, resolve_move, apply_restriction
from ninjax.enum_types import ItemEnum as I, AbilityEnum as A
from ninjax.side import update_active, clear_volatile_status


def test_taunt_blocks_queued_status_without_pp(neutral_type):
    """A newly applied Taunt blocks the selected action before PP is spent."""
    key, state = neutral_type
    state = equip_move(state, 'Recover', 1)
    _, state = do_status_move(key, state, 0, make_move('Taunt'), 0, 0)
    assert not allowed_moves(state)[1, 0]
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
    assert resolve_move(state, 1, 1)[0] == 0
    assert jnp.array_equal(allowed_moves(state)[1], jnp.array([True, False, False, False]))
    state = state.replace(volatile_status=state.volatile_status.replace(disabled_move_id=state.volatile_status.disabled_move_id.at[1, 0].set(105),
        disable_turns=state.volatile_status.disable_turns.at[1, 0].set(3)))
    assert resolve_move(state, 1, 1)[1]
    assert not jnp.any(allowed_moves(state)[1])
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
    assert not allowed_moves(disabled)[1, 0]
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


@pytest.mark.jit
def test_restrictions_transform(neutral_type):
    """Restriction state is compatible with jit, vmap, and scan."""
    key, state = neutral_type
    move = make_move('Taunt')
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    out = jax.jit(jax.vmap(lambda s: apply_restriction(s, 0, move)))(batch)
    assert jnp.all(out.volatile_status.taunt_turns[:, 1, 0] == 3)
    def body(s, _):
        """Carry the restriction through two residual phases."""
        return step_side_conditions(key, s)[1], None
    final, _ = jax.jit(lambda s: jax.lax.scan(body, s, None, length=2))(jax.tree.map(lambda x: x[0], out))
    assert final.volatile_status.taunt_turns[1, 0] == 1


def test_environment_masks_and_public_observation(neutral_type):
    """Masks combine restrictions with switch availability and expose effects."""
    from ninjax.battle import Battle
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
    from ninjax.battle import Battle
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
    from ninjax.enum_types import Terrain, TerrainEnum
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
    from ninjax.battle import step_action
    key, state = neutral_type
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(flinched=v.flinched.at[0, 0].set(True),
        last_move_id=v.last_move_id.at[0, 0].set(105)))
    _, out = step_action(key, state, 4, 0)
    assert jnp.array_equal(out.active.current_hp, state.active.current_hp)
    assert out.volatile_status.last_move_id[0, 0] == 105
    from ninjax.enum_types import Status
    state = state.replace(volatile_status=state.volatile_status.replace(flinched=state.volatile_status.flinched.at[0, 0].set(False)))
    state = update_active(state, 0, state.active[0].replace(status=jnp.array([Status.SLEEP]), sleep_counter=jnp.array([2])))
    _, out = step_action(key, state, 4, 0)
    assert out.active.sleep_counter[0, 0] == 1
    assert jnp.array_equal(out.active.current_hp, state.active.current_hp)
