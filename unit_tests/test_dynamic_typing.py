"""Runtime move types and routing for dynamic move properties."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.battle.environment import Battle
from ninjax.battle.types import BattleParams, InformationMode
from ninjax.calculations.damage import compute_base_power
from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.pokemon import FormEnum as F, Type
from ninjax.moves.execution import move_used
from ninjax.moves.properties import effective_move
from ninjax.moves.routing import move_not_drawn_in
from ninjax.records.hits import HitOutcome, HitRecipient
from ninjax.side.updates import update_active
from ninjax.teams.construction import make_move
from ninjax.teams.curated import load_sets, make_sandbox_pool
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


@pytest.mark.parametrize('form,expected', [(F.DEFAULT, Type.GRASS), (F.OGERPON_WELLSPRING, Type.WATER),
    (F.OGERPON_HEARTHFLAME, Type.FIRE), (F.OGERPON_CORNERSTONE, Type.ROCK)])
def test_ivy_cudgel_uses_form_not_current_types(neutral_type, form, expected):
    """The current form controls typing even after a separate type change."""
    _, state = neutral_type
    mon = state.active[0].replace(form=jnp.array([form]), type_list=jnp.array([Type.ELECTRIC, Type.NONE]))
    state = update_active(state, 0, mon)
    move = make_move('Ivy Cudgel')
    assert effective_move(state, 0, move).type[0] == expected
    assert move.type[0] == Type.GRASS and move.crit_stage[0] == 1


@pytest.fixture(scope='module')
def resolve_form_move():
    """Share one JIT executable across all traced form and move IDs."""
    return jax.jit(effective_move)


@pytest.mark.parametrize('name,form,expected', [
    ('Raging Bull', F.DEFAULT, Type.NORMAL),
    ('Raging Bull', F.TAUROS_PALDEA_COMBAT, Type.FIGHTING),
    ('Raging Bull', F.TAUROS_PALDEA_BLAZE, Type.FIRE),
    ('Raging Bull', F.TAUROS_PALDEA_AQUA, Type.WATER),
    ('Aura Wheel', F.DEFAULT, Type.ELECTRIC),
    ('Aura Wheel', F.MORPEKO_HANGRY, Type.DARK),
    ('Aura Wheel', F.TAUROS_PALDEA_AQUA, Type.ELECTRIC),
])
def test_form_move_types(neutral_type, resolve_form_move, name, form, expected):
    """Resolve form moves from identity, independent of the user's current type."""
    _, state = neutral_type
    mon = state.active[0].replace(form=jnp.array([form]), type_list=jnp.array([Type.GRASS, Type.NONE]))
    move = make_move(name)
    out = resolve_form_move(update_active(state, 0, mon), 0, move)
    assert out.type[0] == expected
    assert move.type[0] == (Type.NORMAL if name == 'Raging Bull' else Type.ELECTRIC)


@pytest.fixture(scope='module')
def route_form_move():
    """Compile the regular accuracy and hit path once for all screen cases."""
    return jax.jit(move_not_drawn_in)


@pytest.mark.parametrize('case,outcome', [
    ('substitute', HitOutcome.HIT),
    ('protected', HitOutcome.PROTECTED),
    ('immune', HitOutcome.IMMUNE),
    ('miss', HitOutcome.MISS),
])
def test_raging_bull_screen_break_follows_hit_gates(neutral_type, route_form_move, case, outcome):
    """A form-resolved hit breaks screens before Substitute, after hit gates."""
    key, state = neutral_type
    state = equip_move(state, 'Raging Bull')
    user = state.active[0].replace(form=jnp.array([F.TAUROS_PALDEA_COMBAT]))
    state = update_active(state, 0, user)
    state = state.replace(
        side_conditions=state.side_conditions.replace(
            reflect=state.side_conditions.reflect.at[1, 0].set(4),
            light_screen=state.side_conditions.light_screen.at[1, 0].set(4),
            aurora_veil=state.side_conditions.aurora_veil.at[1, 0].set(4)))
    if case == 'substitute':
        v = state.volatile_status
        state = state.replace(volatile_status=v.replace(substitute_hp=v.substitute_hp.at[1, 0].set(10)))
    elif case == 'protected':
        v = state.volatile_status
        state = state.replace(volatile_status=v.replace(protection=v.protection.replace(protected=v.protection.protected.at[1, 0].set(True))))
    elif case == 'immune':
        defender = state.active[1].replace(type_list=jnp.array([Type.GHOST, Type.NONE]))
        state = update_active(state, 1, defender)
    else:
        moves = state.team.moves
        state = state.replace(team=state.team.replace(moves=moves.replace(
            accuracy=moves.accuracy.at[0, 0, 0, 0].set(0.))))
    _, out = route_form_move(key, state, 0, 0, 0)
    assert out.hit_results[0].outcome == outcome
    expected = 0 if case == 'substitute' else 4
    assert out.side_conditions.reflect[1, 0] == out.side_conditions.light_screen[1, 0] == out.side_conditions.aurora_veil[1, 0] == expected
    if case == 'substitute':
        assert out.hit_results[0].recipient == HitRecipient.SUBSTITUTE
        assert out.active.current_hp[1, 0] == state.active.current_hp[1, 0]


def test_revelation_dance_uses_current_and_tera_type(neutral_type):
    """The first current type is used, or the ordinary Tera type when active."""
    _, state = neutral_type
    mon = state.active[0].replace(type_list=jnp.array([Type.FIRE, Type.FLYING]), tera_type=jnp.array([Type.WATER]))
    state = update_active(state, 0, mon)
    move = make_move('Revelation Dance')
    assert effective_move(state, 0, move).type[0] == Type.FIRE
    state = update_active(state, 0, mon.replace(type_list=jnp.array([Type.NONE, Type.GHOST])))
    assert effective_move(state, 0, move).type[0] == Type.GHOST
    state = update_active(state, 0, mon.replace(is_terastallized=jnp.array([True])))
    assert effective_move(state, 0, move).type[0] == Type.WATER


def test_dynamic_types_follow_normalize_order(neutral_type):
    """Move callbacks override Normalize; Revelation Dance is excluded entirely."""
    _, state = neutral_type
    mon = state.active[0].replace(form=jnp.array([F.OGERPON_WELLSPRING]),
        type_list=jnp.array([Type.FIRE, Type.NONE]), ability=jnp.array([A.NORMALIZE]))
    state = update_active(state, 0, mon)
    ivy = effective_move(state, 0, make_move('Ivy Cudgel'))
    dance = effective_move(state, 0, make_move('Revelation Dance'))
    assert ivy.type[0] == Type.WATER and ivy.base_power[0] == 100
    assert compute_base_power(state.active[0], state.active[1], ivy, state.terrain)[0] == 120
    assert dance.type[0] == Type.FIRE and dance.base_power[0] == 90
    state = update_active(state, 0, mon.replace(form=jnp.array([F.DEFAULT])))
    copied = effective_move(state, 0, make_move('Ivy Cudgel'))
    assert copied.type[0] == Type.NORMAL and copied.base_power[0] == 100
    assert compute_base_power(state.active[0], state.active[1], copied, state.terrain)[0] == 120
    state = update_active(state, 0, mon.replace(ability=jnp.array([A.PIXILATE]),
        type_list=jnp.array([Type.NORMAL, Type.NONE])))
    dance = effective_move(state, 0, make_move('Revelation Dance'))
    assert dance.type[0] == Type.NORMAL and dance.base_power[0] == 90


def test_protean_uses_resolved_ivy_type(neutral_type):
    """Protean sees Ivy Cudgel's form-resolved type when the move is used."""


    key, state = neutral_type
    state = equip_move(state, 'Ivy Cudgel', 0)
    mon = state.active[0].replace(form=jnp.array([F.OGERPON_WELLSPRING]),
        type_list=jnp.array([Type.GRASS, Type.NONE]), ability=jnp.array([A.PROTEAN]))
    state = update_active(state, 0, mon)
    _, out = move_used(key, state, 0, 0)
    assert jnp.array_equal(out.active.type_list[0], jnp.array([Type.WATER, Type.NONE]))


def test_curated_form_is_preserved():
    """Curated construction preserves the form despite shared species numbers."""

    sets = load_sets()
    pool, reports = make_sandbox_pool(sets)
    index = next(i for i, s in enumerate(sets) if s.species == 'Ogerpon-Wellspring')
    assert pool.form[index, 0] == F.OGERPON_WELLSPRING
    assert 'Ivy Cudgel' not in reports[index]['moves']


def test_form_observation_visibility(neutral_type):
    """Form follows species visibility without exposing the hidden opposing team."""

    key, state = neutral_type
    team = state.team.replace(form=state.team.form.at[1, 0, 0].set(F.OGERPON_WELLSPRING)
                              .at[1, 1, 0].set(F.OGERPON_HEARTHFLAME))
    env = Battle(teams=team, params=BattleParams(information_mode=InformationMode.HIDDEN))
    episode, _ = env.reset(key)
    obs = env.observe(episode)
    env.observation_spec.validate(obs)
    assert obs.team.form[0, 1, 0] == F.OGERPON_WELLSPRING
    assert obs.team.form[0, 1, 1] == F.DEFAULT
    assert obs.team.form[1, 0, 1] == F.OGERPON_HEARTHFLAME
