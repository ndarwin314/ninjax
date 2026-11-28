"""Regression coverage for special damage rules through catalogue moves."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.battle.actions import step_move
from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.items import ItemEnum
from ninjax.enums.pokemon import Status, Type
from ninjax.records.hits import HitOutcome, HitRecipient
from ninjax.side.updates import update_active, update_active_fields
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


@pytest.fixture(scope='module')
def step():
    """Share the compiled selected-move entry point across the cases."""
    return jax.jit(step_move)


def set_level(state, player, level):
    """Change a selected Pokémon's level for OHKO eligibility tests."""
    mon = state.active[player]
    mon = mon.replace(stat_table=mon.stat_table.replace(level=jnp.array([level])))
    return update_active(state, player, mon.replace(current_hp=jnp.reshape(mon.max_hp, (1,))))


def set_accuracy(state, accuracy):
    """Set the selected move's accuracy without changing its catalogue rule."""
    mon = state.active[0]
    moves = mon.moves.replace(accuracy=mon.moves.accuracy.at[0, 0].set(accuracy))
    return update_active(state, 0, mon.replace(moves=moves))


@pytest.mark.parametrize('name', [
    'Final Gambit', 'Fissure', 'Horn Drill', 'Guillotine', 'Sheer Cold',
    'False Swipe', 'Present', 'Steel Beam', 'Chloroblast',
])
def test_special_moves_are_available(name):
    """Reviewed catalogue moves have usable PP."""
    move = make_move(name)
    assert move.implemented[0]
    assert move.current_pp[0] > 0


def test_final_gambit_uses_current_hp_and_faints_user(neutral_type, step):
    """Final Gambit transfers the user's current HP and faints on a hit."""
    key, state = neutral_type
    state = equip_move(state, 'Final Gambit')
    state = update_active_fields(state, 0, current_hp=jnp.array([37]))
    before = state.active.current_hp[1, 0]
    _, out = step(key, state, 0, 0, False)
    assert out.active.current_hp[1, 0] == before - 37
    assert out.active.current_hp[0, 0] == 0


def test_final_gambit_preserves_user_when_immune(neutral_type, step):
    """Ghost immunity prevents both the damage and the self-faint."""
    key, state = neutral_type
    state = equip_move(state, 'Final Gambit')
    state = update_active_fields(state, 1, type_list=jnp.array([Type.GHOST, Type.NONE]))
    _, out = step(key, state, 0, 0, False)
    assert jnp.array_equal(out.active.current_hp, state.active.current_hp)


def test_final_gambit_simultaneous_final_ko(neutral_type, step):
    """The user's faint is recorded first when both final Pokémon faint."""
    key, state = neutral_type
    state = equip_move(state, 'Final Gambit')
    state = update_active_fields(state, 1, current_hp=jnp.array([1]))
    state = state.replace(roster=jnp.zeros((2, 6), bool).at[:, 0].set(True))
    _, out = step(key, state, 0, 0, False)
    assert jnp.all(out.active.current_hp == 0)
    assert out.winner == 1


def test_final_gambit_hits_substitute(neutral_type, step):
    """A substitute receives the fixed damage while the user faints."""
    key, state = neutral_type
    state = equip_move(state, 'Final Gambit')
    state = update_active_fields(state, 0, current_hp=jnp.array([37]))
    volatile = state.volatile_status
    state = state.replace(volatile_status=volatile.replace(
        substitute_hp=volatile.substitute_hp.at[1, 0].set(50)))
    _, out = step(key, state, 0, 0, False)
    assert out.volatile_status.substitute_hp[1, 0] == 13
    assert out.active.current_hp[1, 0] == state.active.current_hp[1, 0]
    assert out.active.current_hp[0, 0] == 0


def test_final_gambit_skips_effectiveness_reactions(neutral_type, step):
    """Fixed damage bypasses Tera Shell and Weakness Policy on a weakness."""
    key, state = neutral_type
    state = equip_move(state, 'Final Gambit')
    state = update_active_fields(state, 0, current_hp=jnp.array([37]))
    state = update_active_fields(state, 1, ability=jnp.array([A.TERA_SHELL]),
                                 type_list=jnp.array([Type.ROCK, Type.NONE]),
                                 item=jnp.array([ItemEnum.WEAKNESS_POLICY]))
    _, out = step(key, state, 0, 0, False)
    assert out.active.current_hp[1, 0] == state.active.current_hp[1, 0] - 37
    assert not out.volatile_status.ability_state.tera_shell[1, 0]
    assert out.active.item[1, 0] == ItemEnum.WEAKNESS_POLICY


def test_nonlethal_hit_leaves_one_hp(neutral_type, step):
    """The survival floor applies to the Pokémon receiving damage."""
    key, state = neutral_type
    state = equip_move(state, 'False Swipe')
    state = update_active_fields(state, 1, current_hp=jnp.array([2]))
    _, out = step(key, state, 0, 0, False)
    assert out.active.current_hp[1, 0] == 1
    assert out.hit_results[0].damage == 1


def test_false_swipe_can_break_substitute(neutral_type, step):
    """The recipient's HP floor does not protect the substitute."""
    key, state = neutral_type
    state = equip_move(state, 'False Swipe')
    volatile = state.volatile_status
    state = state.replace(volatile_status=volatile.replace(
        substitute_hp=volatile.substitute_hp.at[1, 0].set(1)))
    _, out = step(key, state, 0, 0, False)
    assert out.volatile_status.substitute_hp[1, 0] == 0
    assert out.active.current_hp[1, 0] == state.active.current_hp[1, 0]


def test_false_swipe_zero_damage_triggers_mummy(neutral_type, step):
    """A contact hit at one HP still triggers Mummy's ability change."""
    key, state = neutral_type
    state = equip_move(state, 'False Swipe')
    state = update_active_fields(state, 1, current_hp=jnp.array([1]),
                                 ability=jnp.array([A.MUMMY]))
    _, out = step(key, state, 0, 0, False)
    assert out.hit_results[0].damage == 0
    assert out.ability[0, 0] == A.MUMMY


def test_ohko_fails_against_higher_level(neutral_type, step):
    """A higher-level target cannot be hit even with an accuracy bonus."""
    key, state = neutral_type
    state = equip_move(state, 'Fissure')
    state = set_level(state, 0, 1)
    state = set_level(state, 1, 100)
    _, out = step(key, state, 0, 0, False)
    assert out.active.current_hp[1, 0] == state.active.current_hp[1, 0]
    assert out.hit_results[0].outcome == HitOutcome.IMMUNE


def test_ohko_sturdy_blocks_even_with_large_level_advantage(neutral_type, step):
    """Sturdy blocks OHKO moves even when their level-based accuracy reaches 100%."""
    key, state = neutral_type
    state = equip_move(state, 'Fissure')
    state = set_level(state, 0, 100)
    state = set_level(state, 1, 1)
    state = update_active_fields(state, 1, ability=jnp.array([A.STURDY]),
                                 current_hp=state.active.current_hp[1] - 1)
    _, out = step(key, state, 0, 0, False)
    assert out.active.current_hp[1, 0] == state.active.current_hp[1, 0]
    assert out.known_abilities[1, 0] == A.STURDY


def test_ohko_hit_with_large_level_advantage(neutral_type, step):
    """Level advantage makes Fissure certain against a grounded target."""
    key, state = neutral_type
    state = equip_move(state, 'Fissure')
    state = set_level(state, 0, 100)
    state = set_level(state, 1, 1)
    _, out = step(key, state, 0, 0, False)
    assert out.active.current_hp[1, 0] == 0


@pytest.mark.parametrize('name, target_type', [('Fissure', Type.FLYING), ('Sheer Cold', Type.ICE)])
def test_ohko_target_immunity_precedes_hit_roll(neutral_type, step, name, target_type):
    """No Guard does not bypass type immunity or Sheer Cold's Ice check."""
    key, state = neutral_type
    state = equip_move(state, name)
    state = update_active_fields(state, 0, ability=jnp.array([A.NO_GUARD]))
    state = update_active_fields(state, 1, type_list=jnp.array([target_type, Type.NONE]))
    _, out = step(key, state, 0, 0, False)
    assert out.hit_results[0].outcome == HitOutcome.IMMUNE
    assert out.active.current_hp[1, 0] == state.active.current_hp[1, 0]


def test_ohko_breaks_substitute(neutral_type, step):
    """A successful OHKO move can destroy the doll without damaging its owner."""
    key, state = neutral_type
    state = equip_move(state, 'Horn Drill')
    state = update_active_fields(state, 0, ability=jnp.array([A.NO_GUARD]))
    volatile = state.volatile_status
    state = state.replace(volatile_status=volatile.replace(
        substitute_hp=volatile.substitute_hp.at[1, 0].set(30)))
    _, out = step(key, state, 0, 0, False)
    assert out.hit_results[0].recipient == HitRecipient.SUBSTITUTE
    assert out.volatile_status.substitute_hp[1, 0] == 0
    assert out.active.current_hp[1, 0] == state.active.current_hp[1, 0]


@pytest.mark.parametrize('name', ['Steel Beam', 'Chloroblast'])
def test_direct_cost_on_hit(neutral_type, step, name):
    """A successful use pays half maximum HP as direct damage."""
    key, state = neutral_type
    state = equip_move(state, name)
    state = set_accuracy(state, 1.)
    _, out = step(key, state, 0, 0, False)
    cost = (state.active.max_hp[0] + 1) // 2
    assert out.active.current_hp[0, 0] == state.active.current_hp[0, 0] - cost


@pytest.mark.parametrize('name, pays', [('Steel Beam', True), ('Chloroblast', False)])
def test_direct_cost_on_protection(neutral_type, step, name, pays):
    """A protected target distinguishes the two direct-cost triggers."""
    key, state = neutral_type
    state = equip_move(state, name)
    volatile = state.volatile_status
    protection = volatile.protection.replace(
        protected=volatile.protection.protected.at[1, 0].set(True))
    state = state.replace(volatile_status=volatile.replace(protection=protection))
    _, out = step(key, state, 0, 0, False)
    cost = (state.active.max_hp[0] + 1) // 2
    assert out.active.current_hp[0, 0] == state.active.current_hp[0, 0] - cost * pays
    assert out.active.current_hp[1, 0] == state.active.current_hp[1, 0]


@pytest.mark.parametrize('name, pays', [('Steel Beam', True), ('Chloroblast', False)])
def test_direct_cost_on_miss(neutral_type, step, name, pays):
    """Only Steel Beam pays when accuracy fails."""
    key, state = neutral_type
    state = set_accuracy(equip_move(state, name), 0.)
    _, out = step(key, state, 0, 0, False)
    assert out.hit_results[0].outcome == HitOutcome.MISS
    cost = (state.active.max_hp[0] + 1) // 2
    assert out.active.current_hp[0, 0] == state.active.current_hp[0, 0] - cost * pays


@pytest.mark.parametrize('name, ability, pays', [
    ('Steel Beam', A.ROCK_HEAD, True), ('Chloroblast', A.ROCK_HEAD, False),
    ('Steel Beam', A.MAGIC_GUARD, False), ('Chloroblast', A.MAGIC_GUARD, False),
])
def test_direct_cost_damage_abilities(neutral_type, step, name, ability, pays):
    """Magic Guard prevents either cost and Rock Head only prevents recoil."""
    key, state = neutral_type
    state = set_accuracy(equip_move(state, name), 1.)
    state = update_active_fields(state, 0, ability=jnp.array([ability]))
    _, out = step(key, state, 0, 0, False)
    cost = (state.active.max_hp[0] + 1) // 2
    assert out.active.current_hp[0, 0] == state.active.current_hp[0, 0] - cost * pays


@pytest.mark.parametrize('name', ['Steel Beam', 'Chloroblast'])
def test_direct_cost_is_skipped_when_asleep(neutral_type, step, name):
    """A sleep interruption happens before execution and its HP cost."""
    key, state = neutral_type
    state = equip_move(state, name)
    state = update_active_fields(state, 0, status=jnp.array([Status.SLEEP]),
                                 sleep_counter=jnp.array([3]))
    _, out = step(key, state, 0, 0, False)
    assert out.active.current_hp[0, 0] == state.active.current_hp[0, 0]
    assert out.active.moves.current_pp[0, 0, 0] == state.active.moves.current_pp[0, 0, 0]


def test_present_has_keyed_healing_and_damage(neutral_type, step):
    """Different explicit keys select healing and damaging Present outcomes."""
    _, state = neutral_type
    state = set_accuracy(equip_move(state, 'Present'), 1.)
    state = update_active_fields(state, 1, current_hp=jnp.array([40]))
    before = state.active.current_hp[1, 0]
    outcomes = [step(jax.random.key(seed), state, 0, 0, False)[1] for seed in range(16)]
    healing_index = next(index for index, out in enumerate(outcomes) if out.active.current_hp[1, 0] > before)
    heal = outcomes[healing_index]
    max_hp = state.active.max_hp[1]
    assert heal.active.current_hp[1, 0] == jnp.minimum(max_hp, before + (max_hp + 2) // 4)
    assert heal.hit_results[0].outcome == HitOutcome.HIT
    assert heal.hit_results[0].damage == 0
    assert any(out.active.current_hp[1, 0] < before for out in outcomes)

    key = jax.random.key(healing_index)
    full = update_active_fields(state, 1, current_hp=jnp.reshape(max_hp, (1,)))
    _, failed = step(key, full, 0, 0, False)
    assert failed.hit_results[0].outcome == HitOutcome.FAILED

    volatile = state.volatile_status
    behind_sub = state.replace(volatile_status=volatile.replace(
        substitute_hp=volatile.substitute_hp.at[1, 0].set(20)))
    _, healed = step(key, behind_sub, 0, 0, False)
    assert healed.active.current_hp[1, 0] == heal.active.current_hp[1, 0]
    assert healed.volatile_status.substitute_hp[1, 0] == 20
