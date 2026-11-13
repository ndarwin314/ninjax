"""Effective ability state and switch-reset behavior."""
import jax
import jax.numpy as jnp
import pytest
from unit_tests.test_battle import neutral_type
from ninjax.enum_types import AbilityEnum as A, Status
from ninjax.side import clear_volatile_status, set_status


def test_ability_override_suppression_and_switch_reset(neutral_type):
    """An override is temporary; suppression does not erase either ability."""
    _, state = neutral_type
    state = state.replace(team=state.team.replace(ability=state.team.ability.at[0, 0, 0].set(A.LEVITATE)))
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(ability_override=v.ability_override.at[0, 0].set(A.INSOMNIA)))
    assert state.ability[0, 0] == A.INSOMNIA
    assert state.team.ability[0, 0, 0] == A.LEVITATE
    v = state.volatile_status
    suppressed = state.replace(volatile_status=v.replace(ability_suppressed=v.ability_suppressed.at[0, 0].set(True)))
    assert suppressed.ability[0, 0] == A.NONE
    assert clear_volatile_status(suppressed, 0).ability[0, 0] == A.LEVITATE


@jax.jit
def read_abilities(state):
    """Exercise effective ability resolution as a compiled pure function."""
    return state.ability


@pytest.mark.jit
def test_ability_resolution_batches(neutral_type):
    """The ability property has a stable player axis under vmap and scan."""
    _, state = neutral_type
    batched = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    assert jax.vmap(read_abilities)(batched).shape == (2, 2, 1)
    _, values = jax.lax.scan(lambda s, _: (s, read_abilities(s)), state, None, length=2)
    assert values.shape == (2, 2, 1)


@pytest.mark.jit
def test_dynamic_active_gather_transforms(neutral_type):
    """Shared gathers retain slots and effective abilities under jit/vmap/scan."""
    _, state = neutral_type
    team = state.team.replace(
        current_hp=jnp.arange(100, 112, dtype=jnp.int32).reshape(2, 6, 1),
        ability=jnp.arange(12, dtype=jnp.int32).reshape(2, 6, 1),
        moves=state.team.moves.replace(identifier=jnp.arange(48, dtype=jnp.int32).reshape(2, 6, 4, 1)))
    state = state.replace(team=team)
    slots = jnp.array([[2, 4], [5, 1]], jnp.int32)
    def read(indices):
        """Read only selected leaves from a dynamic active snapshot."""
        current = state.replace(active_index=indices[:, None])
        active = current.active
        return active.current_hp, active.moves.identifier, current.ability
    expected = jax.vmap(lambda indices: (
        team.current_hp[jnp.arange(2), indices],
        team.moves.identifier[jnp.arange(2), indices],
        team.ability[jnp.arange(2), indices]))(slots)
    actual = jax.jit(jax.vmap(read))(slots)
    _, scanned = jax.jit(lambda indices: jax.lax.scan(
        lambda carry, row: (carry, read(row)), None, indices))(slots)
    for wanted, batched, sequence in zip(expected, actual, scanned):
        assert jnp.array_equal(wanted, batched)
        assert jnp.array_equal(wanted, sequence)


def test_status_immunity_uses_override(neutral_type):
    """A temporary Insomnia grants sleep immunity without changing the team."""
    key, state = neutral_type
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(ability_override=v.ability_override.at[0, 0].set(A.INSOMNIA)))
    _, blocked = set_status(key, state, 0, jnp.array([Status.SLEEP]))
    assert blocked.active.status[0, 0] == Status.NONE
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(ability_suppressed=v.ability_suppressed.at[0, 0].set(True)))
    _, asleep = set_status(key, state, 0, jnp.array([Status.SLEEP]))
    assert asleep.active.status[0, 0] == Status.SLEEP


def test_override_is_used_by_move_immunity_and_survives_hp_updates(neutral_type):
    """Levitate and Sturdy use overrides without rewriting original abilities."""
    from ninjax.prototype import make_move
    from ninjax.side import take_damage_value
    _, state = neutral_type
    state = state.replace(team=state.team.replace(ability=state.team.ability.at[1, 0, 0].set(A.STURDY)))
    v = state.volatile_status
    levitating = state.replace(volatile_status=v.replace(ability_override=v.ability_override.at[1, 0].set(A.LEVITATE)))
    assert levitating.is_immune_to_move(1, make_move('Earthquake'))
    assert not levitating.is_grounded(1)[0]
    out = take_damage_value(levitating, 1, jnp.array([10000]), True)
    assert out.active.current_hp[1, 0] == 0
    assert out.team.ability[1, 0, 0] == A.STURDY
    v = state.volatile_status
    suppressed = state.replace(volatile_status=v.replace(ability_suppressed=v.ability_suppressed.at[1, 0].set(True)))
    assert take_damage_value(suppressed, 1, jnp.array([10000]), True).active.current_hp[1, 0] == 0
    assert take_damage_value(state, 1, jnp.array([10000]), True).active.current_hp[1, 0] == 1


def test_effective_ability_drives_weather_and_boosts(neutral_type):
    """Cloud Nine and Contrary use the volatile ability, not the team value."""
    from ninjax.enum_types import Weather, WeatherEnum as W, StatEnum
    from ninjax.held_effects import effective_weather
    from ninjax.side import add_boosts
    _, state = neutral_type
    state = state.replace(weather=Weather(W.SUN, 5))
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(ability_override=v.ability_override.at[0, 0].set(A.CLOUD_NINE)))
    assert effective_weather(state) == W.NONE
    state = state.replace(volatile_status=state.volatile_status.replace(ability_override=state.volatile_status.ability_override.at[0, 0].set(A.CONTRARY)))
    assert add_boosts(state, 0, StatEnum.ATTACK, 1).boosts.normal_boosts[0, 1] == -1


def test_hit_effects_use_override_and_preserve_original(neutral_type):
    """Stamina from an override responds to a hit without changing the team."""
    from ninjax.prototype import make_move
    from ninjax.game_logic import resolve_damaging_hit
    key, state = neutral_type
    original = state.team.ability[1, 0, 0]
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(ability_override=v.ability_override.at[1, 0].set(A.STAMINA)))
    _, out, _, _ = resolve_damaging_hit(key, state, 0, make_move('Dragon Pulse'))
    assert out.boosts.normal_boosts[1, 2] == 1
    assert out.team.ability[1, 0, 0] == original
    assert out.ability[1, 0] == A.STAMINA


@pytest.mark.jit
def test_effective_immunities_transform(neutral_type):
    """Effective status, grounding and move immunities compile under vmap/scan."""
    from ninjax.prototype import make_move
    _, state = neutral_type
    move = make_move('Earthquake')
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(ability_override=v.ability_override.at[1, 0].set(A.LEVITATE)))
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    def check(s):
        """Check Ground immunity from the active effective ability."""
        return s.is_immune_to_move(1, move), s.is_grounded(1)
    immunity, grounded = jax.jit(jax.vmap(check))(batch)
    assert jnp.all(immunity) and not jnp.any(grounded)
    def step(s, _):
        """Carry the same immutable battle through repeated checks."""
        return s, check(s)
    _, (immunity, grounded) = jax.jit(lambda s: jax.lax.scan(step, s, None, length=2))(state)
    assert jnp.all(immunity) and not jnp.any(grounded)


def test_switch_uses_override_then_restores_incoming_ability(neutral_type):
    """Regenerator uses the outgoing override and the incoming ability is original."""
    from ninjax.game_logic import swap_out
    key, state = neutral_type
    team = state.team.replace(current_hp=state.team.current_hp.at[0, 0, 0].set(20),
        ability=state.team.ability.at[0, 1, 0].set(A.STURDY))
    v = state.volatile_status
    state = state.replace(team=team, volatile_status=v.replace(ability_override=v.ability_override.at[0, 0].set(A.REGENERATOR)))
    maximum = state.active.max_hp[0]
    out = swap_out(key, state, 0, 1)
    assert out.team.current_hp[0, 0, 0] == 20 + maximum // 3
    assert out.team.ability[0, 0, 0] == A.NONE
    assert out.ability[0, 0] == A.STURDY
    assert out.volatile_status.ability_override[0, 0] == -1
