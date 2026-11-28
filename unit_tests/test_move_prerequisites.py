"""Belch and Last Resort track their prerequisites per party member."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.battle.actions import step_move
from ninjax.enums.items import ItemEnum as I
from ninjax.records.hits import HitOutcome
from ninjax.restrictions import active_restriction_mask
from ninjax.side.item_effects import consume, release_fling_item
from ninjax.side.updates import update_active, update_active_fields
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


@pytest.fixture(scope='module')
def step():
    """Reuse a compiled action path across prerequisite cases."""
    return jax.jit(step_move)


def set_move(state, index, name):
    """Replace an active move slot with a catalogue move."""
    mon = state.active[0]
    moves = jax.tree.map(lambda bank, move: bank.at[index].set(move), mon.moves, make_move(name))
    return update_active(state, 0, mon.replace(moves=moves))


def test_belch_requires_actual_berry_consumption(neutral_type):
    """Eating a berry unlocks Belch, while throwing one does not."""
    _, state = neutral_type
    state = equip_move(state, 'Belch')
    state = update_active_fields(state, 0, item=jnp.array([I.SITRUS_BERRY]))
    assert not active_restriction_mask(state)[0, 0]
    flung = release_fling_item(state, 0, I.SITRUS_BERRY)
    assert not active_restriction_mask(flung)[0, 0]
    eaten = consume(state, 0)
    assert active_restriction_mask(eaten)[0, 0]
    assert eaten.ate_berry[0, 0]


def test_last_resort_requires_other_move_uses(neutral_type, step):
    """Last Resort fails after spending PP until each other slot was used."""
    key, state = neutral_type
    state = equip_move(state, 'Last Resort')
    for slot, name in enumerate(('Tackle', 'Protect', 'Recover'), 1):
        state = set_move(state, slot, name)
    before_pp = state.active.moves.current_pp[0, 0, 0]
    _, failed = step(key, state, 0, 0, False, False)
    assert failed.hit_results[0].outcome == HitOutcome.FAILED
    assert failed.active.moves.current_pp[0, 0, 0] == before_pp - 1
    ids = failed.active.moves.identifier[0, :, 0]
    failed = failed.replace(used_move_ids=failed.used_move_ids.at[0, 0].set(ids))
    _, succeeded = step(key, failed, 0, 0, False, False)
    assert succeeded.hit_results[0].outcome == HitOutcome.HIT
