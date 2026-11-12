"""Ability behavior, public knowledge, and curated ability-mode checks."""
import jax
import jax.numpy as jnp
import pytest
from unit_tests.test_battle import neutral_type
from ninjax.battle import Battle, BattleParams
from ninjax.enum_types import AbilityEnum as A, ItemEnum as I, WeatherEnum as W, TerrainEnum as T, MoveType, Type, Status
from ninjax.game_logic import swap_is_alive, swap_out
from ninjax.ability_effects import frisk
from ninjax.held_effects import consume
from ninjax.move_effects import after_every_hit
from ninjax.side import set_status, take_damage_value
from ninjax.curated import load_sets, make_ability_pool, CURATED_ABILITIES


def frisk_state(state):
    """Give player zero Frisk and the opposing lead a visible-on-reveal item."""
    return state.replace(team=state.team.replace(
        ability=state.team.ability.at[0, 0, 0].set(A.FRISK),
        item=state.team.item.at[1, 0, 0].set(I.LIFE_ORB)))


def test_frisk_reveals_only_items_and_retains_knowledge(neutral_type):
    """Frisk reveals the opposing item without exposing private sheet fields."""
    key, battle = neutral_type
    battle = frisk_state(battle)
    env = Battle(battle.team)
    state, ts = env.reset(key)
    obs = ts.observation
    env.observation_spec.validate(obs)
    assert obs.team.item_known[0, 1, 0]
    assert obs.team.item[0, 1, 0] == I.LIFE_ORB
    assert not jnp.any(obs.team.item_known[0, 1, 1:])
    assert not obs.team.sheet_known[0, 1, 0]
    assert jnp.all(obs.team.stats[0, 1] == 0)
    assert jnp.all(obs.team.nature[0, 1] == 0)
    switched = swap_out(key, state.battle, 1, 1)
    obs = env.observe(state.replace(battle=switched))
    assert obs.team.item_known[0, 1, 0] and obs.team.item[0, 1, 0] == I.LIFE_ORB
    assert not obs.team.item_known[0, 1, 1]
    consumed = consume(state.battle, 1)
    obs = env.observe(state.replace(battle=consumed))
    assert obs.team.item_known[0, 1, 0] and obs.team.item[0, 1, 0] == I.NONE
    sash = battle.replace(team=battle.team.replace(item=battle.team.item.at[1, 0, 0].set(I.FOCUS_SASH)))
    sash = take_damage_value(sash, 1, jnp.array([10000]), True)
    assert sash.known_items[1, 0] == I.NONE


def test_frisk_preview_and_empty_item(neutral_type):
    """Entry abilities wait for team selection; Frisk does not reveal an empty slot."""
    key, battle = neutral_type
    env = Battle(frisk_state(battle).team, BattleParams(team_preview=True))
    state, ts = env.reset(key)
    assert not jnp.any(ts.observation.team.item_known[:, 1])
    action = env.action_spec.generate_value().replace(selection=jnp.ones((2, 6), bool))
    state, ts = env.step(state, action)
    assert ts.observation.team.item_known[0, 1, 0]
    assert ts.observation.team.item[0, 1, 0] == I.LIFE_ORB
    empty = frisk(battle.replace(team=battle.team.replace(ability=battle.team.ability.at[0, 0, 0].set(A.FRISK))), 0)
    assert jnp.all(empty.known_items == -1)
    assert empty.events.count == 0


@pytest.mark.parametrize('ability,weather,terrain', [
    (A.DRIZZLE, W.RAIN, T.NONE), (A.SAND_STREAM, W.SANDSTORM, T.NONE),
    (A.SNOW_WARNING, W.SNOW, T.NONE), (A.GRASSY_SURGE, W.NONE, T.GRASSY),
    (A.PSYCHIC_SURGE, W.NONE, T.PSYCHIC), (A.MISTY_SURGE, W.NONE, T.MISTY)])
def test_entry_field_abilities(neutral_type, ability, weather, terrain):
    """Curated weather and terrain setters activate on entry."""
    _, state = neutral_type
    state = state.replace(team=state.team.replace(ability=state.team.ability.at[0, 0, 0].set(ability)))
    out = swap_is_alive(state, 0)
    assert out.weather.weather == weather and out.terrain.terrain == terrain
    assert out.weather.duration == (5 if weather != W.NONE else 0)
    assert out.terrain.duration == (5 if terrain != T.NONE else 0)


def test_regenerator_and_hit_abilities(neutral_type):
    """Regenerator heals on exit; Weak Armor and Thermal Exchange stay distinct."""
    key, state = neutral_type
    team = state.team.replace(ability=state.team.ability.at[0, 0, 0].set(A.REGENERATOR),
                             current_hp=state.team.current_hp.at[0, 0, 0].set(100))
    state = state.replace(team=team)
    out = swap_out(key, state, 0, 1)
    assert out.team.current_hp[0, 0, 0] == min(100 + int(state.active.max_hp[0]) // 3, int(state.active.max_hp[0]))
    move = state.active[0].moves[0].replace(type=jnp.array([Type.FIRE]))
    weak = state.replace(team=state.team.replace(ability=state.team.ability.at[1, 0, 0].set(A.WEAK_ARMOR)))
    _, physical = after_every_hit(key, weak, 0, move, jnp.array([False]))
    assert jnp.array_equal(physical.boosts.normal_boosts[1], jnp.array([0, 0, -1, 0, 0, 2]))
    _, special = after_every_hit(key, weak, 0, move.replace(move_type=jnp.array([MoveType.SPECIAL])), jnp.array([False]))
    assert jnp.all(special.boosts.normal_boosts[1] == 0)
    thermal = state.replace(team=state.team.replace(ability=state.team.ability.at[1, 0, 0].set(A.THERMAL_EXCHANGE)))
    _, out = after_every_hit(key, thermal, 0, move, jnp.array([False]))
    assert out.boosts.normal_boosts[1, 1] == 1
    _, out = set_status(key, thermal, 1, jnp.array([Status.BURN]))
    assert out.active.status[1, 0] == Status.NONE


def test_ability_pool_reports_unsupported_entries():
    """Opt-in pool enables only the explicitly supported abilities and retains omissions."""
    sets = load_sets()
    pool, reports = make_ability_pool(sets)
    for i, (config, report) in enumerate(zip(sets, reports, strict=True)):
        if config.ability in CURATED_ABILITIES:
            assert pool.ability[i, 0] == A[CURATED_ABILITIES[config.ability]]
            assert report['ability'] is None
        else:
            assert pool.ability[i, 0] == A.NONE
            assert report['ability'] == config.ability
    assert jnp.all(pool.item == I.NONE)


@pytest.mark.jit
def test_frisk_jit_vmap_scan(neutral_type):
    """Public knowledge and event updates compile under all three transformations."""
    _, state = neutral_type
    state = frisk_state(state)
    out = jax.jit(frisk)(state, 0)
    assert out.known_items[1, 0] == I.LIFE_ORB
    batch = jax.tree.map(lambda x: jnp.stack((x, x)), state)
    out = jax.jit(jax.vmap(frisk, in_axes=(0, None)))(batch, 0)
    assert jnp.all(out.known_items[:, 1, 0] == I.LIFE_ORB)
    def reveal(s, _):
        """Repeat an entry reveal in a scanned state transition."""
        s = frisk(s, 0)
        return s, s.known_items
    final, history = jax.jit(lambda s: jax.lax.scan(reveal, s, None, length=2))(state)
    assert jnp.all(history[:, 1, 0] == I.LIFE_ORB)
    assert not final.events.overflow


@pytest.mark.jit
def test_curated_ability_battle():
    """Run enabled curated abilities through the compiled environment interface."""
    from ninjax.prototype import run_curated
    result = run_curated(seed=42, max_turns=12, abilities=True)
    assert 0 < result['turns'] <= 12
    assert result['reward'] in ([0., 0.], [1., -1.], [-1., 1.])
    assert any(report['ability'] is None for report in result['omissions'])
    assert 'abilities enabled' in result['sandbox']


def test_surge_terrain_protection_and_power(neutral_type):
    """Surge-created terrains affect statuses, priority and grounded damage."""
    from ninjax.enum_types import Terrain
    from ninjax.game_logic import move_hits
    from ninjax.damage import compute_base_power
    from ninjax.prototype import make_move
    key, state = neutral_type
    misty = state.replace(terrain=Terrain(T.MISTY, 5))
    _, out = set_status(key, misty, 1, jnp.array([Status.BURN]))
    assert out.active.status[1, 0] == Status.NONE
    airborne = misty.replace(team=misty.team.replace(type_list=misty.team.type_list.at[1, 0].set(jnp.array([Type.FLYING, Type.NONE]))))
    _, out = set_status(key, airborne, 1, jnp.array([Status.BURN]))
    assert out.active.status[1, 0] == Status.BURN
    _, out = set_status(key, airborne.replace(gravity_duration=2), 1, jnp.array([Status.BURN]))
    assert out.active.status[1, 0] == Status.NONE
    electric = state.replace(terrain=Terrain(T.ELECTRIC, 5))
    _, out = set_status(key, electric, 1, jnp.array([Status.SLEEP]))
    assert out.active.status[1, 0] == Status.NONE
    _, out = set_status(key, electric, 1, jnp.array([Status.BURN]))
    assert out.active.status[1, 0] == Status.BURN
    quake = make_move('Earthquake')
    power = compute_base_power(state.active[0], state.active[1], quake, Terrain(T.GRASSY, 5))
    assert power[0] == quake.base_power[0] // 2
    dragon = quake.replace(type=jnp.array([Type.DRAGON]))
    assert compute_base_power(state.active[0], state.active[1], dragon, misty.terrain)[0] == dragon.base_power[0] // 2
    priority = state.team.moves.replace(priority=state.team.moves.priority.at[0, 0, 0, 0].set(1))
    psychic = state.replace(team=state.team.replace(moves=priority), terrain=Terrain(T.PSYCHIC, 5))
    _, out = move_hits(key, psychic, 0, 0)
    assert jnp.array_equal(out.active.current_hp, state.active.current_hp)


def test_cloud_nine_blocks_residual_weather(neutral_type):
    """Weather suppression ends when its ability holder faints."""
    from ninjax.enum_types import Weather
    from ninjax.game_logic import end_turn_damage
    _, state = neutral_type
    state = state.replace(weather=Weather(W.SANDSTORM, 5), team=state.team.replace(
        ability=state.team.ability.at[0, 0, 0].set(A.CLOUD_NINE)))
    out = end_turn_damage(state)
    assert jnp.array_equal(out.active.current_hp, state.active.current_hp)
    dead = state.replace(team=state.team.replace(current_hp=state.team.current_hp.at[0, 0, 0].set(0)))
    out = end_turn_damage(dead)
    assert out.active.current_hp[1, 0] < state.active.current_hp[1, 0]
