"""Runtime move types for Ivy Cudgel and Revelation Dance."""
import jax
import jax.numpy as jnp
import pytest
from unit_tests.test_battle import neutral_type
from ninjax.enum_types import AbilityEnum as A, FormEnum as F, Type
from ninjax.game_logic import effective_move
from ninjax.prototype import make_move
from ninjax.side import update_active


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
    assert ivy.type[0] == Type.WATER and ivy.base_power[0] == 120
    assert dance.type[0] == Type.FIRE and dance.base_power[0] == 90
    state = update_active(state, 0, mon.replace(form=jnp.array([F.DEFAULT])))
    copied = effective_move(state, 0, make_move('Ivy Cudgel'))
    assert copied.type[0] == Type.NORMAL and copied.base_power[0] == 120
    state = update_active(state, 0, mon.replace(ability=jnp.array([A.PIXILATE]),
        type_list=jnp.array([Type.NORMAL, Type.NONE])))
    dance = effective_move(state, 0, make_move('Revelation Dance'))
    assert dance.type[0] == Type.NORMAL and dance.base_power[0] == 90


def test_protean_uses_resolved_ivy_type(neutral_type):
    """Protean sees Ivy Cudgel's form-resolved type when the move is used."""
    from unit_tests.test_curated_moves import equip_move
    from ninjax.game_logic import move_used
    key, state = neutral_type
    state = equip_move(state, 'Ivy Cudgel', 0)
    mon = state.active[0].replace(form=jnp.array([F.OGERPON_WELLSPRING]),
        type_list=jnp.array([Type.GRASS, Type.NONE]), ability=jnp.array([A.PROTEAN]))
    state = update_active(state, 0, mon)
    _, out = move_used(key, state, 0, 0)
    assert jnp.array_equal(out.active.type_list[0], jnp.array([Type.WATER, Type.NONE]))


def test_curated_form_is_preserved():
    """Curated construction preserves the form despite shared species numbers."""
    from ninjax.curated import load_sets, make_sandbox_pool
    sets = load_sets()
    pool, reports = make_sandbox_pool(sets)
    index = next(i for i, s in enumerate(sets) if s.species == 'Ogerpon-Wellspring')
    assert pool.form[index, 0] == F.OGERPON_WELLSPRING
    assert 'Ivy Cudgel' not in reports[index]['moves']


def test_form_observation_visibility(neutral_type):
    """Form follows species visibility without exposing the hidden opposing team."""
    from ninjax.battle import Battle, BattleParams, InformationMode
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


@pytest.mark.jit
def test_dynamic_types_jit_vmap_scan(neutral_type):
    """Dynamic typing uses immutable arrays under JAX transformations."""
    _, state = neutral_type
    move = make_move('Ivy Cudgel')
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    forms = batch.team.form.at[0, 0, 0, 0].set(F.OGERPON_WELLSPRING).at[1, 0, 0, 0].set(F.OGERPON_HEARTHFLAME)
    batch = batch.replace(team=batch.team.replace(form=forms))
    types = jax.jit(jax.vmap(lambda s: effective_move(s, 0, move).type[0]))(batch)
    assert jnp.array_equal(types, jnp.array([Type.WATER, Type.FIRE]))
    _, types = jax.jit(lambda s: jax.lax.scan(lambda s, _: (s, effective_move(s, 0, move).type[0]), s, None, length=2))(state)
    assert jnp.all(types == Type.GRASS)
