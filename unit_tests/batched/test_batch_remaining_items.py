"""Opt-in batched and scanned compatibility checks."""
import jax
import jax.numpy as jnp

from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.items import ItemEnum as I
from ninjax.side.item_effects import kings_rock_flinch
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type
from unit_tests.test_remaining_items import held


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
