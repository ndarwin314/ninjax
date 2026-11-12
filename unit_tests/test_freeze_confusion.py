"""Freeze, confusion, and weather-sensitive curated moves."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.battle import before_move, step_move
from ninjax.curated_move_effects import secondary_effects
from ninjax.enum_types import AbilityEnum as A, Status, Terrain, TerrainEnum as T, Type, Weather, WeatherEnum as W
from ninjax.game_logic import move_accuracy, resolve_damaging_hit
from ninjax.prototype import make_move
from ninjax.side import apply_confusion, update_active
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


@pytest.mark.parametrize('name,weather,expected', [
    ('Blizzard', W.NONE, .7), ('Blizzard', W.SNOW, 1.),
    ('Hurricane', W.NONE, .7), ('Hurricane', W.RAIN, 1.), ('Hurricane', W.SUN, .5),
])
def test_weather_accuracy(neutral_type, name, weather, expected):
    """Weather adjusts accuracy and suppressed weather has no effect."""
    _, state = neutral_type
    state = state.replace(weather=Weather(weather, 5))
    assert float(move_accuracy(state, 0, make_move(name))[0]) == pytest.approx(expected)
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.CLOUD_NINE])))
    assert float(move_accuracy(state, 0, make_move(name))[0]) == pytest.approx(.7)


def test_weather_always_hit_ignores_evasion(neutral_type):
    """Snow and rain bypass evasion for their respective moves."""
    _, state = neutral_type
    state = state.replace(boosts=state.boosts.replace(acc_boosts=state.boosts.acc_boosts.at[1, 1].set(6)))
    for name, weather in [('Blizzard', W.SNOW), ('Hurricane', W.RAIN)]:
        assert float(move_accuracy(state.replace(weather=Weather(weather, 5)), 0, make_move(name))[0]) == 1.


def test_secondary_status_and_immunity(neutral_type):
    """Both moves apply their secondary effect and sunshine prevents freezing."""
    key, state = neutral_type
    blizzard = make_move('Blizzard').replace(secondary_chance=jnp.array([1.]))
    hurricane = make_move('Hurricane').replace(secondary_chance=jnp.array([1.]))
    _, out = secondary_effects(key, state, 0, blizzard)
    assert out.active.status[1, 0] == Status.FREEZE
    _, out = secondary_effects(key, state.replace(weather=Weather(W.SUN, 5)), 0, blizzard)
    assert out.active.status[1, 0] == Status.NONE
    _, out = secondary_effects(key, state, 0, hurricane)
    assert 2 <= out.volatile_status.confusion_turns[1, 0] <= 5
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.SHIELD_DUST])))
    _, out = secondary_effects(key, state, 0, hurricane)
    assert out.volatile_status.confusion_turns[1, 0] == 0


def test_freeze_thaw_and_fire_hit(neutral_type):
    """Frozen users can thaw before acting, and Fire damage thaws its target."""
    key, state = neutral_type
    state = update_active(state, 0, state.active[0].replace(status=jnp.array([Status.FREEZE])))
    _, subkey = jax.random.split(key)
    thaws = bool(jax.random.uniform(subkey) < .2)
    _, out, interrupted = before_move(key, state, 0)
    assert bool(interrupted) != thaws
    assert int(out.active.status[0, 0]) == (Status.NONE if thaws else Status.FREEZE)
    _, out, interrupted = before_move(key, state, 0, defrost_move=True)
    assert not interrupted and out.active.status[0, 0] == Status.FREEZE
    state = update_active(state, 1, state.active[1].replace(status=jnp.array([Status.FREEZE])))
    fire = make_move('Flamethrower').replace(secondary_chance=jnp.array([0.]))
    _, out, _, _ = resolve_damaging_hit(key, state, 0, fire)
    assert out.active.status[1, 0] == Status.NONE


def test_confusion_immunities_and_countdown(neutral_type):
    """Own Tempo and grounded Misty Terrain prevent confusion; flinch pauses it."""
    key, state = neutral_type
    immune = update_active(state, 0, state.active[0].replace(ability=jnp.array([A.OWN_TEMPO])))
    _, out = apply_confusion(key, immune, 0)
    assert out.volatile_status.confusion_turns[0, 0] == 0
    ground = update_active(state, 0, state.active[0].replace(type_list=jnp.array([Type.NORMAL, Type.NONE])))
    ground = ground.replace(terrain=Terrain(T.MISTY, 5))
    _, out = apply_confusion(key, ground, 0)
    assert out.volatile_status.confusion_turns[0, 0] == 0
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(confusion_turns=v.confusion_turns.at[0, 0].set(3), flinched=v.flinched.at[0, 0].set(True)))
    _, out, blocked = before_move(key, state, 0)
    assert blocked and out.volatile_status.confusion_turns[0, 0] == 3
    v = out.volatile_status
    state = out.replace(volatile_status=v.replace(confusion_turns=v.confusion_turns.at[0, 0].set(1), flinched=v.flinched.at[0, 0].set(False)))
    _, out, blocked = before_move(key, state, 0)
    assert not blocked and out.volatile_status.confusion_turns[0, 0] == 0


def test_confusion_self_hit_uses_own_stats(neutral_type):
    """A self-hit consumes the action and uses the user's Attack and Defense."""
    _, state = neutral_type
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(confusion_turns=v.confusion_turns.at[0, 0].set(3)))
    key = next(jax.random.key(i) for i in range(100) if int(jax.random.randint(jax.random.split(jax.random.key(i), 3)[1], (), 0, 100)) < 33)
    _, _, damage_key = jax.random.split(key, 3)
    stats = state.raw_boosted_stats[0]
    level = state.active.level[0, 0]
    base = int((((2 * level // 5 + 2) * 40 * stats[1] // stats[2]) // 50 + 2))
    damage = max(1, (base % 65536) * int(jax.random.randint(damage_key, (), 85, 101)) // 100)
    _, out, blocked = before_move(key, state, 0)
    assert blocked
    assert out.active.current_hp[0, 0] == max(0, int(state.active.current_hp[0, 0]) - damage)
    assert out.active.current_hp[1, 0] == state.active.current_hp[1, 0]
    assert out.volatile_status.confusion_turns[0, 0] == 2


def test_defrost_move_thaws_only_when_executed(neutral_type):
    """A flinched user stays frozen, while executing Flare Blitz thaws it."""
    key, state = neutral_type
    state = equip_move(state, 'Flare Blitz')
    state = update_active(state, 0, state.active[0].replace(status=jnp.array([Status.FREEZE])))
    v = state.volatile_status
    flinched = state.replace(volatile_status=v.replace(flinched=v.flinched.at[0, 0].set(True)))
    _, out = step_move(key, flinched, 0, 0, False)
    assert out.active.status[0, 0] == Status.FREEZE
    assert out.active.moves.current_pp[0, 0, 0] == state.active.moves.current_pp[0, 0, 0]
    _, out = step_move(key, state, 0, 0, False)
    assert out.active.status[0, 0] == Status.NONE
    assert out.active.moves.current_pp[0, 0, 0] == state.active.moves.current_pp[0, 0, 0] - 1
