"""Gen 9 item interactions with per-hit, contact and base-power hooks."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.curated_move_effects import secondary_effects
from ninjax.damage import compute_base_power
from ninjax.enum_types import AbilityEnum as A, ItemEnum as I, FormEnum as F, Status, Type
from ninjax.game_logic import do_status_move, move_hits, resolve_damaging_hit
from ninjax.held_effects import kings_rock_flinch
from ninjax.move_effects import after_every_hit
from ninjax.prototype import make_move
from ninjax.side import update_active
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


def held(state, player, item, ability=A.NONE):
    """Set a holder's item and effective base ability for controlled cases."""
    return update_active(state, player, state.active[player].replace(
        item=jnp.array([item]), ability=jnp.array([ability])))


@pytest.mark.parametrize('ability', [A.NONE, A.SHIELD_DUST, A.INNER_FOCUS])
def test_kings_rock_independent_secondary(neutral_type, ability):
    """The item can flinch alongside a failed native secondary, except immunities."""
    _, state = neutral_type
    state = held(held(state, 0, I.KINGS_ROCK), 1, I.NONE, ability)
    # Select a reproducible successful item roll, accounting for the native roll.
    key = next(jax.random.key(i) for i in range(100)
        if jax.random.uniform(jax.random.split(jax.random.split(jax.random.key(i))[0])[1]) < .1)
    move = make_move('Lumina Crash').replace(secondary_chance=jnp.array([0.]))
    _, out = secondary_effects(key, state, 0, move)
    assert bool(out.volatile_status.flinched[1, 0]) == (ability == A.NONE)
    assert out.boosts.normal_boosts[1, 4] == 0


def test_kings_rock_probability_and_nonstacking(neutral_type):
    """Per-hit independent rolls follow 10%/20% thresholds without extra native flinch."""
    _, state = neutral_type
    state = held(state, 0, I.KINGS_ROCK)
    keys = jax.random.split(jax.random.key(83), 64)
    move = make_move('Population Bomb')
    draws = jax.vmap(lambda k: jax.random.uniform(jax.random.split(k)[1]))(keys)
    for ability, threshold in ((A.NONE, .1), (A.SERENE_GRACE, .2)):
        current = held(state, 0, I.KINGS_ROCK, ability)
        flags = jax.vmap(lambda k: kings_rock_flinch(k, current, 0, move)[1].volatile_status.flinched[1, 0])(keys)
        assert jnp.array_equal(flags, draws < threshold)
    for candidate in (make_move('Fake Out'), make_move('Calm Mind')):
        out_key, out = kings_rock_flinch(keys[0], state, 0, candidate)
        assert not out.volatile_status.flinched[1, 0]
        assert jnp.array_equal(jax.random.key_data(out_key), jax.random.key_data(keys[0]))


def test_kings_rock_does_not_flinch_through_substitute(neutral_type):
    """A real damaging hit on the doll never rolls the item's secondary."""
    key, state = neutral_type
    state = held(state, 0, I.KINGS_ROCK)
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(substitute_hp=v.substitute_hp.at[1, 0].set(100)))
    move = make_move('Dragon Pulse').replace(base_power=jnp.array([10]))
    item_key, out, _, _ = resolve_damaging_hit(key, state, 0, move)
    plain_key, _, _, _ = resolve_damaging_hit(key, held(state, 0, I.NONE), 0, move)
    assert not out.volatile_status.flinched[1, 0]
    assert out.volatile_status.substitute_hp[1, 0] < 100
    assert jnp.array_equal(jax.random.key_data(item_key), jax.random.key_data(plain_key))


@pytest.mark.parametrize('defense', [A.ROUGH_SKIN, A.GOOEY])
def test_pads_block_defender_contact_reactions(neutral_type, defense):
    """Pads prevent Helmet, contact damage and contact stage drops."""
    key, state = neutral_type
    state = held(held(state, 0, I.PROTECTIVE_PADS), 1, I.ROCKY_HELMET, defense)
    move = make_move('U-turn')
    _, safe = after_every_hit(key, state, 0, move, False)
    assert safe.active.current_hp[0, 0] == state.active.current_hp[0, 0]
    assert safe.boosts.normal_boosts[0, 5] == 0
    assert safe.known_items[0, 0] == I.PROTECTIVE_PADS
    assert safe.known_abilities[1, 0] == defense
    _, exposed = after_every_hit(key, held(state, 0, I.NONE), 0, move, False)
    assert exposed.active.current_hp[0, 0] < state.active.current_hp[0, 0]
    if defense == A.GOOEY:
        assert exposed.boosts.normal_boosts[0, 5] == -1


def test_pads_preserve_poison_touch_and_contact_power(neutral_type):
    """Pads do not disable the user's contact ability or Tough Claws bonus."""
    _, state = neutral_type
    state = held(held(state, 0, I.PROTECTIVE_PADS, A.POISON_TOUCH), 1, I.NONE, A.STATIC)
    key = next(jax.random.key(i) for i in range(100)
        if jax.random.uniform(jax.random.split(jax.random.key(i), 3)[2]) < .3)
    move = make_move('U-turn')
    _, out = after_every_hit(key, state, 0, move, False)
    assert out.active.status[0, 0] == Status.NONE
    assert out.active.status[1, 0] == Status.POISON
    mon = state.active[0].replace(ability=jnp.array([A.TOUGH_CLAWS]))
    assert compute_base_power(mon, state.active[1], move, state.terrain)[0] > move.base_power[0]


@pytest.mark.parametrize('ability,bypasses', [(A.NONE, False), (A.UNSEEN_FIST, True)])
def test_pads_spiky_shield_and_unseen_fist(neutral_type, ability, bypasses):
    """Pads prevent shield retaliation while Unseen Fist still bypasses protection."""
    key, state = neutral_type
    state = held(equip_move(state, 'U-turn'), 0, I.PROTECTIVE_PADS, ability)
    key, state = do_status_move(key, state, 1, make_move('Spiky Shield'), 0, 0)
    _, out = move_hits(key, state, 0, 0)
    assert out.active.current_hp[0, 0] == state.active.current_hp[0, 0]
    assert bool(out.active.current_hp[1, 0] < state.active.current_hp[1, 0]) == bypasses


@pytest.mark.parametrize('form,item,expected', [
    (F.OGERPON_WELLSPRING, I.WELLSPRING_MASK, 96),
    (F.OGERPON_WELLSPRING, I.NONE, 80),
    (F.OGERPON_HEARTHFLAME, I.WELLSPRING_MASK, 80),
    (F.OGERPON_HEARTHFLAME, I.HEARTHFLAME_MASK, 96),
    (F.OGERPON_CORNERSTONE, I.CORNERSTONE_MASK, 96),
])
def test_mask_power_is_form_bound_and_all_types(neutral_type, form, item, expected):
    """Matching masks boost arbitrary move types with pinned 4915/4096 rounding."""
    _, state = neutral_type
    mon = state.active[0].replace(item=jnp.array([item]), form=jnp.array([form]))
    move = make_move('Dragon Pulse').replace(base_power=jnp.array([80]))
    assert compute_base_power(mon, state.active[1], move, state.terrain)[0] == expected


@pytest.mark.parametrize('ability,substitute,activates', [
    (A.NONE, False, True), (A.NONE, True, False), (A.WATER_ABSORB, False, False),
])
def test_absorb_bulb_requires_water_hit_reaching_holder(neutral_type, ability, substitute, activates):
    """Substitute and Water absorption prevent consumption; successful hits boost SpA."""
    key, state = neutral_type
    state = held(equip_move(state, 'Water Shuriken'), 1, I.ABSORB_BULB, ability)
    # A low-power Water hit lets the holder survive and trigger its item.
    move = make_move('Dragon Pulse').replace(type=jnp.array([Type.WATER]), base_power=jnp.array([10]))
    mon = state.active[0]
    state = update_active(state, 0, mon.replace(moves=jax.tree.map(lambda bank, value: bank.at[0].set(value), mon.moves, move)))
    if substitute:
        v = state.volatile_status
        state = state.replace(volatile_status=v.replace(substitute_hp=v.substitute_hp.at[1, 0].set(100)))
    _, out = move_hits(key, state, 0, 0)
    assert out.active.item[1, 0] == (I.NONE if activates else I.ABSORB_BULB)
    assert out.boosts.normal_boosts[1, 3] == (1 if activates else 0)
    if activates:
        assert out.known_items[1, 0] == I.NONE
        _, twice, _, _ = resolve_damaging_hit(key, out, 0, move, 1)
        assert twice.boosts.normal_boosts[1, 3] == 1
