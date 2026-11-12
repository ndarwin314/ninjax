"""Substitute routing and immutable per-attempt hit records."""
import jax.numpy as jnp
import pytest

from ninjax.battle import step_move
from ninjax.enum_types import AbilityEnum as A, ItemEnum as I, Status
from ninjax.game_logic import create_substitute, move_hits, move_misses, resolve_damaging_hit, resolve_multi_hit
from ninjax.hit import HitOutcome, HitRecipient
from ninjax.prototype import make_move
from ninjax.side import clear_volatile_status, update_active
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


def with_substitute(state, player, hp):
    """Set a known Substitute HP for an isolated hit-routing test."""
    v = state.volatile_status
    return state.replace(volatile_status=v.replace(substitute_hp=v.substitute_hp.at[player, 0].set(hp)))


def test_substitute_cost_failures_and_switch_reset(neutral_type):
    """Substitute uses quarter HP, cannot be stacked, and clears with volatiles."""
    _, state = neutral_type
    hp = state.active.current_hp[0, 0]
    cost = state.active.max_hp[0] // 4
    out = create_substitute(state, 0)
    assert out.active.current_hp[0, 0] == hp - cost
    assert out.volatile_status.substitute_hp[0, 0] == cost
    assert create_substitute(out, 0).active.current_hp[0, 0] == hp - cost
    assert clear_volatile_status(out, 0).volatile_status.substitute_hp[0, 0] == 0
    low = update_active(state, 0, state.active[0].replace(current_hp=jnp.array([cost])))
    assert create_substitute(low, 0).volatile_status.substitute_hp[0, 0] == 0


def test_substitute_absorbs_damage_and_blocks_reactions(neutral_type):
    """A breaking hit has no spillover, contact reaction or secondary status."""
    key, state = neutral_type
    state = with_substitute(state, 1, 1)
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.ROUGH_SKIN]), item=jnp.array([I.ROCKY_HELMET])))
    move = make_move('Flare Blitz').replace(secondary_chance=jnp.array([1.]))
    _, out, _, _ = resolve_damaging_hit(key, state, 0, move)
    assert out.active.current_hp[1, 0] == state.active.current_hp[1, 0]
    assert out.active.status[1, 0] == Status.NONE
    assert out.volatile_status.substitute_hp[1, 0] == 0
    result = out.hit_results[0]
    assert result.outcome == HitOutcome.HIT and result.recipient == HitRecipient.SUBSTITUTE
    assert result.damage == 1 and result.actual_side == result.intended_side == 1
    assert not result.redirected
    # Recoil uses the one point removed from the Substitute, not computed damage.
    assert out.active.current_hp[0, 0] == state.active.current_hp[0, 0] - 1


@pytest.mark.parametrize('bypass', ['sound', 'infiltrator'])
def test_bypass_substitute(neutral_type, bypass):
    """Sound moves and Infiltrator can damage the Pokémon behind a Substitute."""
    key, state = neutral_type
    state = with_substitute(state, 1, 100)
    move = make_move('Hyper Voice') if bypass == 'sound' else make_move('Flamethrower')
    if bypass == 'infiltrator':
        state = update_active(state, 0, state.active[0].replace(ability=jnp.array([A.INFILTRATOR])))
    _, out, _, _ = resolve_damaging_hit(key, state, 0, move)
    assert out.volatile_status.substitute_hp[1, 0] == 100
    assert out.active.current_hp[1, 0] < state.active.current_hp[1, 0]
    assert out.hit_results[0].recipient == HitRecipient.POKEMON


def test_status_and_bypasssub(neutral_type):
    """Targeted status is blocked; bypass moves and side hazards work."""
    key, state = neutral_type
    state = with_substitute(equip_move(state, 'Spore'), 1, 30)
    _, out = move_hits(key, state, 0, 0)
    assert out.active.status[1, 0] == Status.NONE
    assert out.hit_results[0].outcome == HitOutcome.BLOCKED
    state = equip_move(state, 'Encore')
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(last_move_id=v.last_move_id.at[1, 0].set(state.active.moves.identifier[1, 0, 0])))
    _, out = move_hits(key, state, 0, 0)
    assert out.hit_results[0].outcome == HitOutcome.HIT
    state = equip_move(state, 'Stealth Rock')
    _, out = move_hits(key, state, 0, 0)
    assert out.stealth_rocks[1, 0]


def test_multihit_records_each_recipient(neutral_type):
    """Later hits can damage the Pokémon after an earlier hit breaks Substitute."""
    key, state = neutral_type
    state = with_substitute(state, 1, 1)
    move = make_move('Surging Strikes')
    _, out = resolve_multi_hit(key, state, 0, move)
    assert out.hit_results[0].recipient == HitRecipient.SUBSTITUTE
    assert out.hit_results[1].recipient == HitRecipient.POKEMON
    assert out.hit_results[0].damage == 1
    assert out.active.current_hp[1, 0] < state.active.current_hp[1, 0]


def test_miss_result_and_move_execution(neutral_type):
    """Misses have an attempt record and Substitute can be used from a move slot."""
    key, state = neutral_type
    state = equip_move(state, 'Substitute')
    _, missed = move_misses(key, state, 0, 0)
    assert missed.hit_results[0].outcome == HitOutcome.MISS
    assert missed.hit_results[0].actual_side == 0
    _, out = step_move(key, state, 0, 0, False)
    assert out.volatile_status.substitute_hp[0, 0] == state.active.max_hp[0] // 4
    assert out.hit_results[0].outcome == HitOutcome.HIT
    assert out.hit_results[0].actual_side == 0


def test_hit_results_record_immunity_and_failed_creation(neutral_type):
    """Immunity and an already-present Substitute produce non-hit outcomes."""
    from ninjax.enum_types import Type
    key, state = neutral_type
    state = equip_move(state, 'Super Fang')
    state = update_active(state, 1, state.active[1].replace(type_list=jnp.array([Type.GHOST, Type.NONE])))
    _, out = move_hits(key, state, 0, 0)
    assert out.hit_results[0].outcome == HitOutcome.IMMUNE
    assert out.hit_results[0].recipient == HitRecipient.NONE
    state = with_substitute(equip_move(state, 'Substitute'), 0, 12)
    _, out = move_hits(key, state, 0, 0)
    assert out.hit_results[0].outcome == HitOutcome.BLOCKED
    assert out.volatile_status.substitute_hp[0, 0] == 12


def test_hit_record_shapes_under_jax_transforms(neutral_type):
    """Tracing batched and scanned hits retains a fixed PyTree shape."""
    import jax
    from ninjax.game_logic import record_hit
    _, state = neutral_type
    def one(s, target):
        """Record one attempted hit for a traced target index."""
        return record_hit(s, 0, target, HitOutcome.HIT, HitRecipient.POKEMON, 3)
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    shaped = jax.eval_shape(jax.vmap(lambda s, target: one(s, target).hit_results.data), batch, jnp.array([0, 1]))
    assert shaped.shape == (2, 10, 7)
    def scan(s):
        """Trace a fixed scan without compiling a full battle."""
        return jax.lax.scan(lambda carry, i: (one(carry, i), None), s, jnp.array([0, 1]))[0].hit_results.data
    assert jax.eval_shape(scan, state).shape == (10, 7)


def test_drain_uses_actual_substitute_damage(neutral_type):
    """Drain is based on the Substitute HP removed, capped when it breaks."""
    key, state = neutral_type
    state = update_active(state, 0, state.active[0].replace(current_hp=jnp.array([10])))
    state = with_substitute(state, 1, 5)
    _, out, _, _ = resolve_damaging_hit(key, state, 0, make_move('Giga Drain'))
    assert out.hit_results[0].damage == 5
    assert out.active.current_hp[0, 0] == 13
    assert out.active.current_hp[1, 0] == state.active.current_hp[1, 0]


def test_hit_result_can_represent_future_redirection():
    """Target positions distinguish a redirected attempt on the same side."""
    from ninjax.hit import HitResult, HitResults
    result = HitResult(1, 0, 1, 1, HitOutcome.MISS, HitRecipient.NONE, 0)
    saved = HitResults.empty().set(0, result)[0]
    assert saved.redirected and saved.intended_position == 0 and saved.actual_position == 1


def test_hazard_move_runs_after_substitute_damage(neutral_type):
    """Stone Axe still places rocks after damaging a Substitute."""
    key, state = neutral_type
    state = with_substitute(state, 1, 1)
    _, out, _, _ = resolve_damaging_hit(key, state, 0, make_move('Stone Axe'))
    assert out.hit_results[0].recipient == HitRecipient.SUBSTITUTE
    assert out.stealth_rocks[1, 0]
