"""Offensive stat owner selection before item and ability modifiers."""
import jax
import jax.numpy as jnp
from ninjax.enum_types import OffensiveStatOwner as Owner
from ninjax.side import raw_boosted_stats, select_offensive_stats
from ninjax.stats import StatBoosts


def test_user_and_target_offensive_stages():
    """Select each owner's own raw stats and stages, with critical/Unaware rules."""
    stats = jnp.array([[100, 120, 100, 100, 100, 100], [100, 200, 100, 100, 100, 100]])
    boosts = StatBoosts(normal_boosts=jnp.zeros((2, 6), jnp.int32).at[0, 1].set(2).at[1, 1].set(-2), acc_boosts=jnp.zeros((2, 2), jnp.int32))
    assert select_offensive_stats(stats, boosts, 0, Owner.USER, False, False)[1] == 240
    assert select_offensive_stats(stats, boosts, 0, Owner.TARGET, False, False)[1] == 100
    assert select_offensive_stats(stats, boosts, 0, Owner.TARGET, False, True)[1] == 200
    assert select_offensive_stats(stats, boosts, 0, Owner.TARGET, True, False)[1] == 200
    assert select_offensive_stats(stats, boosts, 0, Owner.USER, True, False)[1] == 120
    attacker, defender = raw_boosted_stats(stats, boosts, 0, False, False, False, Owner.TARGET)
    assert attacker[1] == 100 and defender[2] == 100


def test_stat_owner_traces_with_dynamic_owner():
    """The owner selector supports traced values under vmap and scan."""
    stats = jnp.full((2, 6), 100, jnp.int32)
    boosts = StatBoosts(normal_boosts=jnp.zeros((2, 6), jnp.int32), acc_boosts=jnp.zeros((2, 2), jnp.int32))
    owners = jnp.array([Owner.USER, Owner.TARGET])
    def select(owner):
        """Select a traced stat owner."""
        return select_offensive_stats(stats, boosts, 0, owner, False, False)
    assert jax.eval_shape(jax.vmap(select), owners).shape == (2, 6)
    assert jax.eval_shape(lambda: jax.lax.scan(lambda _, x: (None, select(x)), None, owners)[1]).shape == (2, 6)
