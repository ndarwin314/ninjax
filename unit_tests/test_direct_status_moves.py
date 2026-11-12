"""Moves that fit existing HP, status, and held-item state."""
import jax
import jax.numpy as jnp
import pytest

from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move
from ninjax.prototype import make_move
from ninjax.game_logic import do_status_move, move_hits
from ninjax.curated_move_effects import direct_status_move
from ninjax.side import update_active
from ninjax.enum_types import AbilityEnum as A, ItemEnum as I, Status, Terrain, TerrainEnum as T


def test_pain_split_sets_hp_and_bypasses_damage_abilities(neutral_type):
    """Pain Split ignores Magic Guard/Multiscale and caps each HP independently."""
    key, state = neutral_type
    state = update_active(state, 0, state.active[0].replace(current_hp=jnp.array([21]), ability=jnp.array([A.MAGIC_GUARD])))
    state = update_active(state, 1, state.active[1].replace(current_hp=jnp.array([100]), ability=jnp.array([A.MULTISCALE])))
    _, out = do_status_move(key, state, 0, make_move('Pain Split'), 0, 0)
    assert jnp.all(out.active.current_hp == 60)
    assert jnp.all(out.volatile_status.last_damage == 0)
    # The mean is computed before either participant's Sitrus Berry activates.
    state = update_active(state, 0, state.active[0].replace(item=jnp.array([I.SITRUS_BERRY])))
    _, out = do_status_move(key, state, 0, make_move('Pain Split'), 0, 0)
    assert out.active.current_hp[1, 0] == 60
    assert out.active.current_hp[0, 0] == 60 + state.active.max_hp[0] // 4


@pytest.mark.parametrize('blocked', ['none', 'full', 'sleep', 'insomnia', 'electric', 'misty'])
def test_rest_replaces_status_only_when_sleep_succeeds(neutral_type, blocked):
    """Rest heals and sleeps for two turns; failed attempts preserve the user."""
    key, state = neutral_type
    state = update_active(state, 0, state.active[0].replace(current_hp=jnp.array([20]), status=jnp.array([Status.TOXIC])))
    if blocked == 'full':
        state = update_active(state, 0, state.active[0].replace(current_hp=jnp.array([state.active.max_hp[0]])))
    if blocked == 'sleep':
        state = update_active(state, 0, state.active[0].replace(status=jnp.array([Status.SLEEP]), sleep_counter=jnp.array([2])))
    if blocked == 'insomnia':
        state = update_active(state, 0, state.active[0].replace(ability=jnp.array([A.INSOMNIA])))
    if blocked in ('electric', 'misty'):
        state = state.replace(terrain=Terrain(T.ELECTRIC if blocked == 'electric' else T.MISTY, 5))
    _, out = do_status_move(key, state, 0, make_move('Rest'), 0, 0)
    if blocked != 'none':
        assert jnp.array_equal(out.active.current_hp, state.active.current_hp)
        assert jnp.array_equal(out.active.status, state.active.status)
        assert jnp.array_equal(out.active.sleep_counter, state.active.sleep_counter)
    else:
        assert out.active.current_hp[0, 0] == state.active.max_hp[0]
        assert out.active.status[0, 0] == Status.SLEEP
        assert out.active.sleep_counter[0, 0] == 3


def test_rest_sleep_duration(neutral_type):
    """Two following actions are interrupted; the third can execute."""
    from ninjax.battle import step_move
    key, state = neutral_type
    state = update_active(state, 0, state.active[0].replace(current_hp=jnp.array([20])))
    key, state = do_status_move(key, state, 0, make_move('Rest'), 0, 0)
    pp = state.active.moves.current_pp[0, 0, 0]
    for _ in range(2):
        key, state = step_move(key, state, 0, 0, False)
        assert state.active.moves.current_pp[0, 0, 0] == pp
    key, state = step_move(key, state, 0, 0, False)
    assert state.active.status[0, 0] == Status.NONE
    assert state.active.moves.current_pp[0, 0, 0] == pp-1


def test_poltergeist_requires_and_reveals_item(neutral_type):
    """An itemless target is unharmed; a held item is publicly remembered."""
    key, state = neutral_type
    state = equip_move(state, 'Poltergeist')
    _, failed = move_hits(key, state, 0, 0)
    assert jnp.array_equal(failed.active.current_hp, state.active.current_hp)
    assert failed.known_items[1, 0] == -1
    state = update_active(state, 1, state.active[1].replace(item=jnp.array([I.LEFTOVERS])))
    _, out = move_hits(key, state, 0, 0)
    assert out.active.current_hp[1, 0] < state.active.current_hp[1, 0]
    assert out.active.item[1, 0] == I.LEFTOVERS
    assert out.known_items[1, 0] == I.LEFTOVERS
    guarded = state.replace(volatile_status=state.volatile_status.replace(protected=state.volatile_status.protected.at[1, 0].set(True)))
    _, out = move_hits(key, guarded, 0, 0)
    assert out.known_items[1, 0] == -1
    assert jnp.array_equal(out.active.current_hp, guarded.active.current_hp)


@pytest.mark.jit
def test_direct_hp_callbacks_transform(neutral_type):
    """Direct HP/status assignments support actual jit, vmap, and scan."""
    key, state = neutral_type
    state = update_active(state, 0, state.active[0].replace(current_hp=jnp.array([20])))
    rest = make_move('Rest')
    split = make_move('Pain Split')
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    keys = jax.random.split(key, 2)
    _, out = jax.jit(jax.vmap(lambda k, s: do_status_move(k, s, 0, rest, 0, 0)))(keys, batch)
    assert jnp.all(out.active.status[:, 0, 0] == Status.SLEEP)
    def step(carry, _):
        """Carry explicit state and key through repeated HP assignments."""
        return direct_status_move(carry[0], carry[1], 0, split), None
    (_, out), _ = jax.jit(lambda carry: jax.lax.scan(step, carry, None, length=2))((key, state))
    assert out.active.current_hp[0, 0] == out.active.current_hp[1, 0]
