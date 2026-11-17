"""Opt-in batched and scanned compatibility checks."""
from unit_tests.test_passive_mechanics import (
    I,
    Status,
    configure,
    jax,
    jnp,
    neutral_type,
    pytest,
    residual_abilities,
    update_items,
)
from ninjax.side.stat_effects import stat_changes


@pytest.mark.jit
def test_passive_helpers_jit_vmap_scan(neutral_type):
    """New passive transitions preserve every state leaf under JAX transforms."""
    key, state = neutral_type
    state = configure(state, {'ability': 'Shed Skin', 'defender': 'Defiant', 'item': 'Lum Berry', 'status': 'tox'})

    def update(carry, unused):
        """Use explicit keys and immutable state through item/ability updates."""
        key, state = carry
        key, state = residual_abilities(key, update_items(state))
        state = stat_changes(state, 1, jnp.array([2]), jnp.array([-1]), True)
        return (key, state), state.active.status
    (_, final), _ = jax.jit(lambda k, s: jax.lax.scan(update, (k, s), None, length=2))(key, state)
    assert final.active.status[0, 0] == Status.NONE
    assert final.active.item[0, 0] == I.NONE
    assert final.boosts.normal_boosts[1, 1] == 4
    batch = jax.tree.map(lambda x: jnp.stack((x, x)), state)
    out = jax.jit(jax.vmap(update_items))(batch)
    assert jnp.all(out.active.status[:, 0, 0] == Status.NONE)
    assert not jnp.any(out.events.overflow)
