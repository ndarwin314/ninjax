"""Worry Seed behavior, ability lifecycle, and public knowledge."""
import jax
import jax.numpy as jnp
import pytest
from unit_tests.test_battle import neutral_type
from unit_tests.test_curated_moves import equip_move
from ninjax.ability_effects import worry_seed
from ninjax.battle import Battle, BattleParams, InformationMode
from ninjax.enum_types import AbilityEnum as A, ItemEnum as I, Status, Weather, WeatherEnum as W
from ninjax.game_logic import move_hits, swap_out
from ninjax.held_effects import effective_weather
from ninjax.side import update_active


def test_worry_seed_wakes_and_reveals_until_switch(neutral_type):
    """The override is public and temporary, without revealing the base ability."""
    key, state = neutral_type
    target = state.active[1].replace(ability=jnp.array([A.CLOUD_NINE]), status=jnp.array([Status.SLEEP]), sleep_counter=jnp.array([2]))
    state = update_active(state, 1, target)
    state = state.replace(weather=Weather(W.SUN, 5))
    state = equip_move(state, 'Worry Seed', 0)
    _, out = move_hits(key, state, 0, 0)
    assert out.ability[1, 0] == A.INSOMNIA and out.team.ability[1, 0, 0] == A.CLOUD_NINE
    assert out.active.status[1, 0] == Status.NONE and out.active.sleep_counter[1, 0] == 0
    assert effective_weather(out) == W.SUN
    assert out.known_abilities[1, 0] == A.INSOMNIA
    env = Battle(teams=state.team, params=BattleParams(information_mode=InformationMode.HIDDEN))
    episode, _ = env.reset(key)
    episode = episode.replace(battle=out)
    obs = env.observe(episode)
    assert obs.team.ability_known[0, 1, 0] and obs.team.ability[0, 1, 0] == A.INSOMNIA
    assert not obs.team.sheet_known[0, 1, 0]
    switched = swap_out(key, out, 1, 1)
    assert switched.known_abilities[1, 0] == -1
    assert switched.team.ability[1, 0, 0] == A.CLOUD_NINE
    assert switched.volatile_status.ability_override[1, 0] == -1
    returned = swap_out(key, switched, 1, 0)
    episode = episode.replace(battle=returned)
    obs = env.observe(episode)
    assert not obs.team.ability_known[0, 1, 0]
    assert obs.team.ability[1, 0, 0] == A.CLOUD_NINE


@pytest.mark.parametrize('ability,item', [(A.TRUANT,I.NONE), (A.INSOMNIA,I.NONE), (A.COMATOSE,I.NONE), (A.LEVITATE,I.ABILITY_SHIELD)])
def test_worry_seed_failed_targets(neutral_type, ability, item):
    """Restricted abilities and Ability Shield cannot be replaced."""
    _, state = neutral_type
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([ability]), item=jnp.array([item])))
    out = worry_seed(state, 0)
    assert out.volatile_status.ability_override[1, 0] == -1
    assert out.known_abilities[1, 0] == -1
    if item == I.ABILITY_SHIELD:
        assert out.known_items[1, 0] == I.ABILITY_SHIELD


def test_worry_seed_clears_old_ability_state(neutral_type):
    """Replacing Protosynthesis ends its stat effect and does not restore Booster Energy."""
    _, state = neutral_type
    v = state.volatile_status
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.PROTOSYNTHESIS])))
    state = state.replace(volatile_status=v.replace(paradox_stat=v.paradox_stat.at[1, 0].set(1), paradox_booster=v.paradox_booster.at[1, 0].set(True)))
    out = worry_seed(state, 0)
    assert out.ability[1, 0] == A.INSOMNIA
    assert out.volatile_status.paradox_stat[1, 0] == -1
    assert not out.volatile_status.paradox_booster[1, 0]


def test_worry_seed_preserves_suppression(neutral_type):
    """A suppressed ability is still replaced and Insomnia still wakes on hit."""
    _, state = neutral_type
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.LEVITATE]),
        status=jnp.array([Status.SLEEP]), sleep_counter=jnp.array([2])))
    v = state.volatile_status
    state = state.replace(volatile_status=v.replace(ability_suppressed=v.ability_suppressed.at[1, 0].set(True)))
    out = worry_seed(state, 0)
    assert out.volatile_status.ability_override[1, 0] == A.INSOMNIA
    assert out.volatile_status.ability_suppressed[1, 0]
    assert out.ability[1, 0] == A.NONE and out.active.status[1, 0] == Status.NONE


def test_vital_spirit_is_distinct_but_still_blocks_sleep(neutral_type):
    """Vital Spirit may be replaced even though it has the same sleep immunity."""
    _, state = neutral_type
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.VITAL_SPIRIT])))
    assert state.is_status_immune(1, jnp.array([Status.SLEEP]))[0]
    assert worry_seed(state, 0).ability[1, 0] == A.INSOMNIA


@pytest.mark.parametrize('guard', ['protect', 'good_as_gold'])
def test_worry_seed_respects_existing_hit_blocks(neutral_type, guard):
    """Protection and Good as Gold prevent both the change and its reveal."""
    key, state = neutral_type
    state = equip_move(state, 'Worry Seed', 0)
    if guard == 'protect':
        v = state.volatile_status
        state = state.replace(volatile_status=v.replace(protected=v.protected.at[1, 0].set(True)))
    else:
        state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.GOOD_AS_GOLD])))
    _, out = move_hits(key, state, 0, 0)
    assert out.volatile_status.ability_override[1, 0] == -1
    assert out.known_abilities[1, 0] == -1


@pytest.mark.jit
def test_worry_seed_jit_vmap_scan(neutral_type):
    """Ability changes and repeated failure remain compatible with transforms."""
    _, state = neutral_type
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    out = jax.jit(jax.vmap(lambda s: worry_seed(s, 0)))(batch)
    assert jnp.all(jax.vmap(lambda s: s.ability)(out)[:, 1, 0] == A.INSOMNIA)
    final, _ = jax.jit(lambda s: jax.lax.scan(lambda s, _: (worry_seed(s, 0), None), s, None, length=2))(state)
    assert final.ability[1, 0] == A.INSOMNIA
