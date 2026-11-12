import jax
import jax.numpy as jnp
from unit_tests.test_battle import neutral_type
from ninjax.enum_types import ItemEnum as I, AbilityEnum as A, WeatherEnum as W, Weather, StatEnum as S
from ninjax.side import take_damage_value, clear_volatile_status
from ninjax.held_effects import refresh_paradox, modify_stats, effective_weather, after_move, leftovers


def test_sash_consumed_once_and_berry_heals(neutral_type):
    _, state = neutral_type
    state = state.replace(team=state.team.replace(item=state.team.item.at[1, 0, 0].set(I.FOCUS_SASH)))
    hit = jax.jit(lambda s, damage: take_damage_value(s, 1, damage, True))
    survived = hit(state, 10000)
    assert survived.active.current_hp[1, 0] == 1
    assert survived.active.item[1, 0] == I.NONE
    assert hit(survived, 10000).active.current_hp[1, 0] == 0
    state = state.replace(team=state.team.replace(item=state.team.item.at[1, 0, 0].set(I.SITRUS_BERRY)))
    maxhp = state.active.current_hp[1, 0]
    out = hit(state, maxhp//2+1)
    assert out.active.current_hp[1, 0] == maxhp-(maxhp//2+1)+maxhp//4
    assert out.active.item[1, 0] == I.NONE


def test_booster_weather_and_switch_reset(neutral_type):
    _, state = neutral_type
    state = state.replace(team=state.team.replace(
        ability=state.team.ability.at[0, 0, 0].set(A.PROTOSYNTHESIS),
        item=state.team.item.at[0, 0, 0].set(I.BOOSTER_ENERGY)))
    refresh = jax.jit(refresh_paradox)
    sun = refresh(state.replace(weather=Weather(W.SUN, 5)))
    assert sun.active.item[0, 0] == I.BOOSTER_ENERGY
    assert sun.volatile_status.paradox_stat[0, 0] == 1
    boosted = refresh(sun.replace(weather=Weather(W.NONE, 0)))
    assert boosted.active.item[0, 0] == I.NONE
    assert boosted.volatile_status.paradox_booster[0, 0]
    cleared = clear_volatile_status(boosted, 0)
    assert cleared.volatile_status.paradox_stat[0, 0] == -1
    assert cleared.volatile_status.choice_lock[0, 0] == -1
    assert not cleared.volatile_status.paradox_booster[0, 0]
    stats = jax.jit(modify_stats)(boosted, 0, jnp.full(6, 100))
    assert stats[S.ATTACK] == 130
    assert stats[S.DEFENSE] == 100


def test_cloud_nine_suppresses_without_erasing_weather(neutral_type):
    _, state = neutral_type
    state = state.replace(weather=Weather(W.SUN, 5), team=state.team.replace(
        ability=state.team.ability.at[0, 0, 0].set(A.CLOUD_NINE)))
    assert jax.jit(effective_weather)(state) == W.NONE
    assert state.weather.weather == W.SUN


def test_pixilate_changes_effective_move_without_changing_slot(neutral_type):
    from ninjax.game_logic import effective_move
    from ninjax.enum_types import Type
    _, state = neutral_type
    state = state.replace(team=state.team.replace(ability=state.team.ability.at[0, 0, 0].set(A.PIXILATE)))
    original = state.active[0].moves[0]
    move = jax.jit(effective_move)(state, 0, original)
    assert move.type[0] == Type.FAIRY
    assert move.base_power[0] == 96
    assert state.active[0].moves[0].type[0] == Type.NORMAL


def test_moody_avoids_capped_stages(neutral_type):
    from ninjax.game_logic import step_moody
    key, state = neutral_type
    state = state.replace(team=state.team.replace(ability=state.team.ability.at[0, 0, 0].set(A.MOODY)),
                          boosts=state.boosts.replace(normal_boosts=state.boosts.normal_boosts.at[0, 1:].set(6)))
    _, out = jax.jit(step_moody)(key, state)
    # No stat can rise; exactly one can fall.
    assert jnp.sum(out.boosts.normal_boosts[0, 1:] == 5) == 1
    assert jnp.sum(out.boosts.normal_boosts[0, 1:] == 6) == 4


def test_single_hit_and_move_finalization_are_separate(neutral_type):
    from ninjax.game_logic import do_damaging_move, resolve_damaging_hit, finish_damaging_move
    from ninjax.events import EventKind
    key, state = neutral_type
    state = state.replace(team=state.team.replace(item=state.team.item.at[0, 0, 0].set(I.LIFE_ORB)))
    move = state.active[0].moves[0]
    hit = jax.jit(resolve_damaging_hit)
    hit_key, hit_state, damage, defender_before = hit(key, state, 0, move)
    assert damage.shape == (1,)
    assert hit_state.active.current_hp[0, 0] == state.active.current_hp[0, 0]
    assert hit_state.active.current_hp[1, 0] < defender_before.current_hp[0]
    finished = jax.jit(finish_damaging_move)(hit_state, 0, move, damage, defender_before)
    assert finished.active.current_hp[0, 0] == state.active.current_hp[0, 0] - state.active.max_hp[0] // 10
    assert jnp.any((finished.events.kind == EventKind.HP) & (finished.events.target == 0))
    full_key, full = jax.jit(lambda k, s, m: do_damaging_move(k, s, 0, m, 0, 0))(key, state, move)
    assert jnp.array_equal(jax.random.key_data(hit_key), jax.random.key_data(full_key))
    assert all(bool(jnp.array_equal(a, b)) for a, b in zip(jax.tree.leaves(finished), jax.tree.leaves(full), strict=True))
