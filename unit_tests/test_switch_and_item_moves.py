"""Switch continuations and transactional use of represented held items."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.battle.environment import Battle
from ninjax.battle.switching import EntryRecovery, swap_out
from ninjax.battle.types import Phase
from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.items import ItemEnum as I
from ninjax.enums.pokemon import StatEnum, Status
from ninjax.moves.routing import move_hits, prepare_hit_move
from ninjax.moves.status import do_status_move
from ninjax.records.hits import HitOutcome
from ninjax.side.updates import update_active
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move


@pytest.fixture(scope='module')
def hit():
    """Use a single compiled hit dispatcher for both new move families."""
    return jax.jit(move_hits)


@pytest.fixture(scope='module')
def status():
    """Compile the direct status dispatcher for recovery and forced item use."""
    return jax.jit(do_status_move)


@pytest.fixture(scope='module')
def switch():
    """Use the same compiled entry path for both recovery moves."""
    return jax.jit(swap_out)


@pytest.mark.parametrize('name,ability,substitute,drag', [
    ('Dragon Tail', A.SOUNDPROOF, 0, True), ('Circle Throw', A.NONE, 0, True),
    ('Dragon Tail', A.SUCTION_CUPS, 0, False), ('Circle Throw', A.NONE, 1, False),
])
def test_damaging_drag_requires_a_direct_hit(neutral_type, hit, name, ability, substitute, drag):
    """Drag only a surviving direct recipient, preserving the damage outcome."""
    key, state = neutral_type
    state = equip_move(state, name)
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([ability])))
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(substitute_hp=v.substitute_hp.at[1, 0].set(substitute)))
    _, out = hit(key, state, 0, 0)
    assert out.hit_results[0].outcome == HitOutcome.HIT
    assert bool(out.active_index[1, 0] != 0) == drag


@pytest.mark.parametrize('name', ['Roar', 'Whirlwind'])
def test_magic_bounce_redirects_status_drag(neutral_type, hit, name):
    """A reflected phazing move drags its original user."""
    key, state = neutral_type
    state = equip_move(state, name)
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.MAGIC_BOUNCE])))
    _, out = hit(key, state, 0, 0)
    assert out.active_index[0, 0] != 0 and out.active_index[1, 0] == 0
    assert out.hit_results[0].actual_side == 0


@pytest.mark.parametrize('ability,depart,recipient', [
    (A.NONE, 0, 1), (A.MAGIC_BOUNCE, 1, 0), (A.MIRROR_ARMOR, 0, 0),
    (A.CLEAR_BODY, -1, -1),
])
def test_parting_shot_uses_resolved_source(neutral_type, hit, ability, depart, recipient):
    """Drops, reflection and the resulting pivot follow the resolved move."""
    key, state = neutral_type
    state = equip_move(state, 'Parting Shot')
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([ability])))
    _, out = hit(key, state, 0, 0)
    expected = jnp.arange(2) == depart
    assert jnp.array_equal(out.switch_requested, expected)
    assert jnp.array_equal(out.boosts.normal_boosts[:, 1], -(jnp.arange(2) == recipient).astype(jnp.int32))
    assert jnp.array_equal(out.boosts.normal_boosts[:, 3], -(jnp.arange(2) == recipient).astype(jnp.int32))


@pytest.mark.parametrize('name,attacker_item,defender_item,expected_own,expected_other', [
    ('Thief', I.NONE, I.LEFTOVERS, I.LEFTOVERS, I.NONE),
    ('Thief', I.NONE, I.SITRUS_BERRY, I.SITRUS_BERRY, I.NONE),
    ('Covet', I.LIFE_ORB, I.LEFTOVERS, I.LIFE_ORB, I.LEFTOVERS),
    ('Incinerate', I.NONE, I.SITRUS_BERRY, I.NONE, I.NONE),
    ('Bug Bite', I.LEFTOVERS, I.SITRUS_BERRY, I.LEFTOVERS, I.NONE),
    ('Pluck', I.LEFTOVERS, I.SITRUS_BERRY, I.LEFTOVERS, I.NONE),
])
def test_hit_item_mutations(neutral_type, hit, name, attacker_item, defender_item, expected_own, expected_other):
    """Transfer and off-holder consumption do not overwrite recoverable memory."""
    key, state = neutral_type
    state = equip_move(state, name)
    hp = 100 if name in ('Bug Bite', 'Pluck') else state.active.max_hp[0]
    state = update_active(state, 0, state.active[0].replace(item=jnp.array([attacker_item]), current_hp=jnp.array([hp])))
    target_hp = state.active.max_hp[1] // 2 + 2
    state = update_active(state, 1, state.active[1].replace(
        item=jnp.array([defender_item]), current_hp=jnp.array([target_hp])))
    _, out = hit(key, state, 0, 0)
    assert out.active.item[0, 0] == expected_own
    assert out.active.item[1, 0] == expected_other
    assert out.consumed_items[1, 0] == I.NONE
    if name in ('Bug Bite', 'Pluck'):
        assert out.active.current_hp[0, 0] > 100


def test_fling_spends_a_prepared_item_even_into_protect(neutral_type, hit, status):
    """The spent item is recoverable, and a protected foe receives no effects."""
    key, state = neutral_type
    state = equip_move(state, 'Fling')
    state = update_active(state, 0, state.active[0].replace(item=jnp.array([I.FLAME_ORB])))
    _, move = prepare_hit_move(key, state, 0, make_move('Fling'))
    assert move.base_power[0] == 30 and move.flung_item[0] == I.FLAME_ORB
    _, protected = status(key, state, 1, make_move('Protect'), 0, 0)
    _, out = hit(key, protected, 0, 0)
    assert out.hit_results[0].outcome == HitOutcome.PROTECTED
    assert out.active.item[0, 0] == I.NONE and out.active.status[1, 0] == Status.NONE
    assert out.consumed_items[0, 0] == I.FLAME_ORB
    _, out = status(key, out, 0, make_move('Recycle'), 0, 0)
    assert out.active.item[0, 0] == I.FLAME_ORB and out.consumed_items[0, 0] == I.NONE


def test_fling_uses_the_items_on_hit_payload(neutral_type, hit):
    """A successful Fling burns the direct recipient and remembers its item."""
    key, state = neutral_type
    state = equip_move(state, 'Fling')
    state = update_active(state, 0, state.active[0].replace(item=jnp.array([I.FLAME_ORB])))
    _, out = hit(key, state, 0, 0)
    assert out.active.status[1, 0] == Status.BURN and out.active.item[0, 0] == I.NONE
    assert out.consumed_items[0, 0] == I.FLAME_ORB


def test_forced_berry_use_records_its_owner(neutral_type, status):
    """Teatime and Stuff Cheeks can eat berries above their normal thresholds."""
    key, state = neutral_type
    for player in range(2):
        state = update_active(state, player, state.active[player].replace(item=jnp.array([I.SITRUS_BERRY])))
    _, stuffed = status(key, state, 0, make_move('Stuff Cheeks'), 0, 0)
    assert stuffed.boosts.normal_boosts[0, 2] == 2
    assert stuffed.active.item[0, 0] == I.NONE
    _, out = status(key, state, 0, make_move('Teatime'), 0, 0)
    assert jnp.all(out.active.item == I.NONE)
    assert jnp.all(out.consumed_items[:, 0] == I.SITRUS_BERRY)


def test_teatime_checks_each_holder(neutral_type, hit):
    """Klutz suppresses the healing, while Good as Gold prevents eating."""
    key, state = neutral_type
    state = equip_move(state, 'Teatime')
    state = update_active(state, 0, state.active[0].replace(
        ability=jnp.array([A.KLUTZ]), item=jnp.array([I.SITRUS_BERRY]), current_hp=jnp.array([100])))
    state = update_active(state, 1, state.active[1].replace(
        ability=jnp.array([A.GOOD_AS_GOLD]), item=jnp.array([I.SITRUS_BERRY])))
    _, out = hit(key, state, 0, 0)
    assert out.active.item[0, 0] == I.NONE and out.active.current_hp[0, 0] == 100
    assert out.active.item[1, 0] == I.SITRUS_BERRY


def test_bug_bite_uses_the_eaters_item_suppression(neutral_type, hit):
    """A Klutz user can remove the berry but receives no Sitrus healing."""
    key, state = neutral_type
    state = equip_move(state, 'Bug Bite')
    state = update_active(state, 0, state.active[0].replace(ability=jnp.array([A.KLUTZ]), current_hp=jnp.array([100])))
    state = update_active(state, 1, state.active[1].replace(item=jnp.array([I.SITRUS_BERRY])))
    _, out = hit(key, state, 0, 0)
    assert out.active.item[1, 0] == I.NONE and out.active.current_hp[0, 0] == 100


def test_lunar_dance_spends_entry_recovery_on_pp(neutral_type, status, switch):
    """A healthy entrant with spent PP can consume the Lunar Dance effect."""
    key, state = neutral_type
    moves = state.team.moves
    pp = moves.current_pp.at[0, 1, 0, 0].set(0)
    state = state.replace(team=state.team.replace(moves=moves.replace(current_pp=pp)))
    _, sacrificed = status(key, state, 0, make_move('Lunar Dance'), 0, 0)
    assert sacrificed.active.current_hp[0, 0] == 0
    out = switch(key, sacrificed, 0, 1)
    assert out.active.moves.current_pp[0, 0, 0] == moves.max_pp[0, 1, 0, 0]
    assert out.entry_recovery[0] == 0


def test_entry_recovery_respects_order_and_precedes_hazards(neutral_type, status, switch):
    """A healthy replacement keeps Lunar Dance pending for a later entrant."""
    key, state = neutral_type
    team = state.team.replace(current_hp=state.team.current_hp.at[0, 2, 0].set(1))
    state = state.replace(team=team)
    _, danced = status(key, state, 0, make_move('Lunar Dance'), 0, 0)
    switched = switch(key, danced, 0, 1)
    assert switched.entry_recovery[0] == EntryRecovery.LUNAR_DANCE
    _, wished = status(key, switched, 0, make_move('Healing Wish'), 0, 0)
    field = wished.hazards.replace(stealth_rocks=wished.hazards.stealth_rocks.at[0, 0].set(True))
    out = switch(key, wished.replace(hazards=field), 0, 2)
    maximum = out.active.max_hp[0]
    assert out.active.current_hp[0, 0] == maximum - maximum // 8
    assert out.entry_recovery[0] == EntryRecovery.HEALING_WISH


def test_revival_selects_a_fainted_party_member_without_switching(neutral_type):
    """The revival phase reuses party columns and leaves the active slot intact."""
    key, state = neutral_type
    state = equip_move(equip_move(state, 'Revival Blessing'), 'Growl', player=1)
    lead = state.active[0]
    stats = lead.stat_table.base_stats.at[StatEnum.SPEED].set(500)
    state = update_active(state, 0, lead.replace(stat_table=lead.stat_table.replace(base_stats=stats)))
    team = state.team.replace(current_hp=state.team.current_hp.at[0, 2, 0].set(0),
        is_alive=state.team.is_alive.at[0, 2, 0].set(False))
    env = Battle(team)
    episode, _ = env.reset(key)
    step = jax.jit(env.step)
    episode, _ = step(episode, jnp.array([0, 0]))
    assert episode.phase == Phase.REVIVAL and episode.turn == 0 and episode.cursor == 1
    assert list(episode.acting_player_mask) == [True, False]
    assert list(env.legal_action_mask(episode)[0, 8:14]) == [False, False, True, False, False, False]
    revived, _ = step(episode, jnp.array([10, 14]))
    assert revived.phase == Phase.TURN and revived.turn == 1
    assert revived.battle.boosts.normal_boosts[0, 1] == -1
    assert revived.battle.active_index[0, 0] == 0
    assert revived.battle.team.current_hp[0, 2, 0] == team.max_hp[0, 2] // 2
    assert revived.battle.team.status[0, 2, 0] == Status.NONE
