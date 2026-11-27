"""Opt-in batched and scanned compatibility checks."""
import jax
import jax.numpy as jnp

from ninjax.battle.environment import Battle
from unit_tests.test_battle import neutral_type


def test_forced_mask_transforms(neutral_type):
    """Charge/recharge masks trace under batching and scanning without compilation."""
    key, state = neutral_type
    env = Battle(state.team)
    episode, _ = env.reset(key)
    v = state.volatile_status
    episode = episode.replace(battle=state.replace(volatile_status=v.replace(continuation=v.continuation.replace(recharging=v.continuation.recharging.at[0, 0].set(True)))))
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), episode)
    shape = jax.eval_shape(jax.vmap(env.legal_action_mask), batch)
    assert shape.shape == (2, 2, 15)
    def scan(s):
        """Keep fixed action-mask output over a scan of forced turns."""
        return jax.lax.scan(lambda s, _: (s, env.legal_action_mask(s)), s, None, length=2)[1]
    assert jax.eval_shape(scan, episode).shape == (2, 2, 15)
