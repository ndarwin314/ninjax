"""Acceptance checks for the trace-derived, neutral-ability/item sandbox."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ninjax.prototype import make_sandbox_teams, random_actions, run
from ninjax.side import raw_boosted_stats
from ninjax.stats import StatBoosts

REFERENCE = Path(__file__).parent / 'fixtures/showdown/battle-1.json'


def test_reference_projection_is_explicit():
    data = json.loads(REFERENCE.read_text())
    teams, omitted = make_sandbox_teams(data)
    assert teams.current_hp.shape == (2, 6, 1)
    assert jnp.all(teams.ability == 0)
    assert jnp.all(teams.item == 0)
    expected_hp = [[p['maxhp'] for p in side] for side in data['initial']]
    np.testing.assert_array_equal(teams.current_hp[..., 0], expected_hp)
    assert any(p['moves'] for p in omitted)
    assert jnp.all(jnp.where(teams.moves.implemented, True, teams.moves.current_pp == 0))


def test_policy_classes_and_uniform_choices():
    mask = jnp.zeros((2, 15), bool).at[:, :4].set(True).at[:, 8:14].set(True)
    choices = jax.vmap(random_actions, in_axes=(0, None))(jax.random.split(jax.random.key(7), 10000), mask)
    switching = np.asarray(choices >= 8)
    assert .08 < switching.mean() < .12
    counts = np.bincount(np.asarray(choices).ravel(), minlength=15)
    assert np.all(counts[:4] > 4000)
    assert np.all(counts[8:14] > 240)
    forced = random_actions(jax.random.key(2), mask.at[:, :4].set(False))
    assert jnp.all((forced >= 8) & (forced < 14))


def test_damage_boosts_use_each_players_stages():
    stats = jnp.full((2, 6), 100)
    stages = jnp.zeros((2, 6), jnp.int32).at[0, 1].set(2).at[1, 2].set(-2)
    boosts = StatBoosts(normal_boosts=stages, acc_boosts=jnp.zeros((2, 2), jnp.int32))
    attack, defense = jax.jit(raw_boosted_stats)(stats, boosts, 0, False, False, False)
    assert attack[1] == 200
    assert defense[2] == 50
    _, defense = jax.jit(raw_boosted_stats)(stats, boosts, 0, False, False, True)
    assert defense[2] == 50  # Critical hits preserve defensive drops.
    _, defense = jax.jit(raw_boosted_stats)(stats, boosts, 0, True, False, False)
    assert defense[2] == 100


def test_complete_trace_derived_sandbox():
    result = run(REFERENCE, seed=0)
    assert 0 < result['turns'] <= 100
    assert len(result['decisions']) < 1000
    assert result['reward'] in ([1., -1.], [-1., 1.], [0., 0.])
    assert result['omissions']
    assert all(min(d['hp']) >= 0 for d in result['decisions'])
