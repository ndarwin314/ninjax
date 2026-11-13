"""Action timing and previous-turn results for curated attacks."""
import jax.numpy as jnp
import pytest
from ninjax.battle import step_field, step_move
from ninjax.enum_types import MoveResult, MoveType
from ninjax.game_logic import effective_move
from ninjax.hit import HitOutcome
from ninjax.prototype import make_move
from ninjax.side import clear_volatile_status
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


def test_fake_out_first_action_and_reset(neutral_type):
    """Fake Out hits once per entry; an interrupted attempt still counts."""
    key, state = neutral_type
    state = equip_move(state, 'Fake Out')
    key, first = step_move(key, state, 0, 0, False)
    assert first.active.current_hp[1, 0] < state.active.current_hp[1, 0]
    assert first.volatile_status.flinched[1, 0]
    _, second = step_move(key, first, 0, 0, False)
    assert second.hit_results[0].outcome == HitOutcome.FAILED
    assert second.active.current_hp[1, 0] == first.active.current_hp[1, 0]
    assert clear_volatile_status(second, 0).volatile_status.active_move_actions[0, 0] == 0
    v = state.volatile_status
    blocked = state.replace(volatile_status=v.replace(flinched=v.flinched.at[0, 0].set(True)))
    key, blocked = step_move(key, blocked, 0, 0, False)
    v = blocked.volatile_status
    blocked = blocked.replace(volatile_status=v.replace(flinched=v.flinched.at[0, 0].set(False)))
    _, out = step_move(key, blocked, 0, 0, False)
    assert out.hit_results[0].outcome == HitOutcome.FAILED


@pytest.mark.parametrize('pending,category,slot,works', [(True, MoveType.PHYSICAL, 0, True), (True, MoveType.SPECIAL, 0, True), (True, MoveType.STATUS, 0, False), (False, MoveType.PHYSICAL, 0, False), (True, MoveType.PHYSICAL, 1, False)])
def test_sucker_punch_queued_intent(neutral_type, pending, category, slot, works):
    """A pending damaging action must belong to the current opposing Pokémon."""
    key, state = neutral_type
    state = equip_move(state, 'Sucker Punch')
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(pending_move=v.pending_move.at[1, 0].set(pending),
        pending_move_type=v.pending_move_type.at[1, 0].set(category), pending_actor_slot=v.pending_actor_slot.at[1, 0].set(slot)))
    _, out = step_move(key, state, 0, 0, False)
    assert bool(out.active.current_hp[1, 0] < state.active.current_hp[1, 0]) == works
    assert bool(out.hit_results[0].outcome == HitOutcome.FAILED) != works


def test_previous_result_rollover_and_protect(neutral_type):
    """Miss and interruption boost next-turn power; Protect is not a failure."""
    key, state = neutral_type
    state = equip_move(state, 'Fake Out')
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(active_move_actions=v.active_move_actions.at[0, 0].set(1)))
    key, failed = step_move(key, state, 0, 0, False)
    assert failed.volatile_status.move_result[0, 0] == MoveResult.FAILED
    key, next_turn = step_field(key, failed)
    assert next_turn.volatile_status.previous_move_result[0, 0] == MoveResult.FAILED
    assert effective_move(next_turn, 0, make_move('Stomping Tantrum')).base_power[0] == 150
    state = equip_move(state, 'Dragon Pulse')
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(protected=v.protected.at[1, 0].set(True)))
    key, protected = step_move(key, state, 0, 0, False)
    assert protected.volatile_status.move_result[0, 0] == MoveResult.SUCCEEDED
    _, next_turn = step_field(key, protected)
    assert effective_move(next_turn, 0, make_move('Stomping Tantrum')).base_power[0] == 75


def test_scheduler_supplies_sucker_punch_intent(neutral_type):
    """A queued damaging move allows priority Sucker Punch in an environment turn."""
    from ninjax.battle import ActionType, Battle, BattleAction
    key, state = neutral_type
    state = equip_move(state, 'Sucker Punch', player=0)
    state = equip_move(state, 'Dragon Pulse', player=1)
    env = Battle(state.team)
    episode, _ = env.reset(key)
    actions = BattleAction(jnp.zeros(2, jnp.int32), jnp.full(2, ActionType.ATTACK))
    out, ts = env.step(episode, actions)
    assert not ts.extras['invalid_action']
    assert out.battle.active.current_hp[1, 0] < episode.battle.active.current_hp[1, 0]
    assert out.battle.volatile_status.previous_move_result[0, 0] == MoveResult.SUCCEEDED


def test_full_hp_healing_records_failure(neutral_type):
    """A failed Recover at full HP marks the move as failed for the next turn."""
    key, state = neutral_type
    state = equip_move(state, 'Recover')
    key, out = step_move(key, state, 0, 0, False)
    assert out.volatile_status.move_result[0, 0] == MoveResult.FAILED
    _, out = step_field(key, out)
    assert effective_move(out, 0, make_move('Stomping Tantrum')).base_power[0] == 150


def test_observation_exposes_history_not_pending_intent(neutral_type):
    """Move-history leaves are public while pending move categories stay internal."""
    from ninjax.battle import Battle
    key, state = neutral_type
    env = Battle(state.team)
    episode, _ = env.reset(key)
    v = episode.battle.volatile_status
    v = v.replace(previous_move_result=v.previous_move_result.at[0, 0].set(MoveResult.FAILED),
                  active_move_actions=v.active_move_actions.at[0, 0].set(2))
    obs = env.observe(episode.replace(battle=episode.battle.replace(volatile_status=v)))
    env.observation_spec.validate(obs)
    assert obs.active.previous_move_result[0, 0] == MoveResult.FAILED
    assert obs.active.active_move_actions[0, 0] == 2
    assert not hasattr(obs.active, 'pending_move_type')


def test_stomping_result_transform_shapes(neutral_type):
    """Previous-result power selection traces under vmap and scan."""
    import jax
    _, state = neutral_type
    move = make_move('Stomping Tantrum')
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    assert jax.eval_shape(jax.vmap(lambda s: effective_move(s, 0, move).base_power), batch).shape == (2, 1)
    def scan(s):
        """Carry move-result state while observing power over two turns."""
        return jax.lax.scan(lambda s, _: (s, effective_move(s, 0, move).base_power), s, None, length=2)[1]
    assert jax.eval_shape(scan, state).shape == (2, 1)
