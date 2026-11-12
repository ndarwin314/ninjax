"""Mechanics checks for the expanded Scarlet/Violet curated move catalogue."""
import jax
import jax.numpy as jnp
import pytest

from unit_tests.test_battle import neutral_type
from ninjax.prototype import make_move
from ninjax.curated import load_sets, make_sandbox_pool
from ninjax.curated_move_effects import field_status, move_overrides, secondary_effects
from ninjax.enum_types import AbilityEnum as A, ItemEnum as I, WeatherEnum as W, Weather, TerrainEnum as T, Terrain, Type
from ninjax.game_logic import resolve_damaging_hit, do_status_move, effective_move, move_hits, finish_damaging_move
from ninjax.side import update_active


def equip_move(state, name, player=0):
    """Put one catalogue move into the active user's first slot."""
    mon = state.active[player]
    moves = jax.tree.map(lambda bank, move: bank.at[0].set(move), mon.moves, make_move(name))
    return update_active(state, player, mon.replace(moves=moves))


@pytest.mark.parametrize('name', ['Night Shade', 'Super Fang', 'Ruination', 'Endeavor'])
def test_fixed_damage_ignores_screens_and_resistance(neutral_type, name):
    """Fixed damage uses level, half current HP, or the HP difference."""
    key, state = neutral_type
    # Damage is checked directly; the dispatch tests separately cover immunities.
    state = state.replace(reflect=jnp.ones((2, 1), jnp.int32), light_screen=jnp.ones((2, 1), jnp.int32))
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.MULTISCALE])))
    mon = state.active[0]
    state = update_active(state, 0, mon.replace(current_hp=mon.current_hp // 2))
    hp = state.active.current_hp[1, 0]
    expected = state.active.level[0, 0] if name == 'Night Shade' else (hp-state.active.current_hp[0, 0] if name == 'Endeavor' else hp//2)
    _, out, damage, _ = resolve_damaging_hit(key, state, 0, make_move(name))
    assert damage[0] == expected
    assert out.active.current_hp[1, 0] == hp-expected


def test_drain_uses_actual_loss_before_berry_healing(neutral_type):
    """Overkill and Sitrus healing cannot inflate or erase drain credit."""
    key, state = neutral_type
    mon = state.active[0]
    state = update_active(state, 0, mon.replace(current_hp=jnp.array([1])))
    target = state.active[1]
    state = update_active(state, 1, target.replace(current_hp=jnp.array([3])))
    _, out, _, _ = resolve_damaging_hit(key, state, 0, make_move('Bitter Blade'))
    assert out.active.current_hp[0, 0] == 3  # round(3/2) heals 2
    target = target.replace(item=jnp.array([I.SITRUS_BERRY]))
    state = update_active(state, 1, target)
    _, out, _, _ = resolve_damaging_hit(key, state, 0, make_move('Giga Drain').replace(base_power=jnp.array([20])))
    loss = out.volatile_status.last_damage[1, 0]
    assert out.active.current_hp[0, 0] == 1+jnp.floor(loss*.5+.5)


def test_secondary_drop_and_shield_dust(neutral_type):
    """Lumina Crash's guaranteed drop respects secondary-effect immunity."""
    key, state = neutral_type
    move = make_move('Lumina Crash')
    _, out = secondary_effects(key, state, 0, move)
    assert out.boosts.normal_boosts[1, 4] == -2
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.SHIELD_DUST])))
    _, out = secondary_effects(key, state, 0, move)
    assert out.boosts.normal_boosts[1, 4] == 0


def test_self_drops_and_two_stat_setup(neutral_type):
    """Close Combat drops both defenses; Calm Mind raises both special stats."""
    key, state = neutral_type
    move = make_move('Close Combat')
    out = finish_damaging_move(state, 0, move, jnp.array([1]), state.active[1])
    assert out.boosts.normal_boosts[0, 2] == -1
    assert out.boosts.normal_boosts[0, 4] == -1
    _, out = do_status_move(key, state, 0, make_move('Calm Mind'), 0, 0)
    assert out.boosts.normal_boosts[0, 3] == 1
    assert out.boosts.normal_boosts[0, 4] == 1


def test_screens_weather_and_court_change(neutral_type):
    """Aurora Veil needs snow; Light Clay extends it; Court Change swaps it."""
    _, state = neutral_type
    move = make_move('Aurora Veil')
    assert field_status(state, 0, move).aurora_veil[0, 0] == 0
    state = state.replace(weather=Weather(W.SNOW, 5))
    state = update_active(state, 0, state.active[0].replace(item=jnp.array([I.LIGHT_CLAY])))
    out = field_status(state, 0, move)
    assert out.aurora_veil[0, 0] == 8
    out = field_status(out, 1, make_move('Court Change'))
    assert out.aurora_veil[1, 0] == 8 and out.aurora_veil[0, 0] == 0


def test_conditional_move_overrides(neutral_type):
    """Weather Ball, Acrobatics, Stored Power, and Grassy Glide use live state."""
    _, state = neutral_type
    assert effective_move(state, 0, make_move('Acrobatics')).base_power[0] == 110
    rain = state.replace(weather=Weather(W.RAIN, 5))
    ball = effective_move(rain, 0, make_move('Weather Ball'))
    assert ball.type[0] == Type.WATER and ball.base_power[0] == 100
    state = state.replace(boosts=state.boosts.replace(normal_boosts=state.boosts.normal_boosts.at[0, 1].set(3)))
    assert effective_move(state, 0, make_move('Stored Power')).base_power[0] == 80
    state = state.replace(terrain=Terrain(T.GRASSY, 5))
    assert effective_move(state, 0, make_move('Grassy Glide')).priority[0] == 1


def test_brick_break_and_hazard_callbacks(neutral_type):
    """Brick Break clears screens and Rapid Spin clears the user's hazards."""
    key, state = neutral_type
    state = state.replace(reflect=state.reflect.at[1, 0].set(5), light_screen=state.light_screen.at[1, 0].set(5))
    _, out, _, _ = resolve_damaging_hit(key, state, 0, make_move('Brick Break'))
    assert out.reflect[1, 0] == out.light_screen[1, 0] == 0
    state = state.replace(spikes=state.spikes.at[0, 0].set(3))
    _, out, _, _ = resolve_damaging_hit(key, state, 0, make_move('Rapid Spin'))
    assert out.spikes[0, 0] == 0 and out.boosts.normal_boosts[0, 5] == 1


def test_fixed_damage_still_respects_ghost_immunity(neutral_type):
    """Super Fang's HP rule cannot bypass a Ghost target's immunity."""
    key, state = neutral_type
    state = equip_move(state, 'Super Fang')
    state = update_active(state, 1, state.active[1].replace(type_list=jnp.array([Type.GHOST, Type.NONE])))
    _, out = move_hits(key, state, 0, 0)
    assert out.active.current_hp[1, 0] == state.active.current_hp[1, 0]


def test_curated_coverage_expanded():
    """The curated projection enables the new batch and retains complex omissions."""
    _, reports = make_sandbox_pool(load_sets())
    remaining = {move for report in reports for move in report['moves']}
    assert len(remaining) == 16
    assert 'Beat Up' in remaining
    assert not {'Encore', 'Disable', 'Taunt'} & remaining
    assert not {'Blizzard', 'Hurricane', 'Substitute'} & remaining
    assert not {'Electro Shot', 'Hyper Beam'} & remaining
    assert not {'Calm Mind', 'Draco Meteor', 'Spore', 'Night Shade'} & remaining


@pytest.mark.jit
def test_new_callbacks_jit_vmap_scan(neutral_type):
    """Shared callbacks preserve their PyTree structure under transformations."""
    _, state = neutral_type
    move = make_move('Light Screen')
    from ninjax.curated_move_effects import stat_changes
    changes = make_move('Dragon Pulse').replace(self_stat=jnp.arange(1, 6), self_stages=jnp.ones(5, jnp.int32))
    def transition(s):
        """Apply one field effect and one conditional power override."""
        s = field_status(s, 0, move)
        s = stat_changes(s, 0, changes.self_stat, changes.self_stages)
        return s, move_overrides(s, 0, make_move('Stored Power')).base_power
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    out, powers = jax.jit(jax.vmap(transition))(batch)
    assert jnp.all(out.light_screen[:, 0, 0] == 5)
    assert jnp.all(out.boosts.normal_boosts[:, 0, 1:] == 1)
    final, _ = jax.jit(lambda s: jax.lax.scan(lambda s, _: transition(s), s, None, length=2))(state)
    assert final.light_screen[0, 0] == 5
    assert jnp.all(final.boosts.normal_boosts[0, 1:] == 2)


@pytest.mark.parametrize('weight,power', [(99, 20), (100, 40), (249, 40), (250, 60), (500, 80), (1000, 100), (2000, 120)])
def test_low_kick_weight_thresholds(neutral_type, weight, power):
    """Weight boundaries use hectograms and the Scarlet/Violet power table."""
    _, state = neutral_type
    state = update_active(state, 1, state.active[1].replace(weight_hg=jnp.array([weight])))
    assert effective_move(state, 0, make_move('Low Kick')).base_power[0] == power


def test_defog_and_perish_song_soundproof(neutral_type):
    """Defog clears both sides' hazards; Soundproof prevents a Perish counter."""
    key, state = neutral_type
    state = state.replace(spikes=jnp.ones((2, 1), jnp.int32), terrain=Terrain(T.PSYCHIC, 5))
    _, out = do_status_move(key, state, 0, make_move('Defog'), 0, 0)
    assert jnp.all(out.spikes == 0) and out.terrain.terrain == T.NONE
    assert out.boosts.acc_boosts[1, 1] == -1
    state = update_active(state, 1, state.active[1].replace(ability=jnp.array([A.SOUNDPROOF])))
    _, out = do_status_move(key, state, 0, make_move('Perish Song'), 0, 0)
    assert out.perish_turns[0, 0] == 4 and out.perish_turns[1, 0] == -1


def test_flinch_interrupts_and_expires(neutral_type):
    """A successful flinch prevents the next action and clears at turn end."""
    from ninjax.battle import step_move
    from ninjax.game_logic import step_side_conditions
    key, state = neutral_type
    # Force the existing secondary roll without relying on one seed's outcome.
    move = make_move('Iron Head').replace(secondary_chance=jnp.array([1.]))
    key, state = secondary_effects(key, state, 0, move)
    assert state.volatile_status.flinched[1, 0]
    pp = state.active.moves.current_pp[1, 0, 0]
    key, state = step_move(key, state, 1, 0, False)
    assert state.active.moves.current_pp[1, 0, 0] == pp
    _, state = step_side_conditions(key, state)
    assert not state.volatile_status.flinched[1, 0]


def test_body_press_and_sacred_sword_stage_selection(neutral_type):
    """Body Press uses Defense; Sacred Sword ignores target defensive stages."""
    key, state = neutral_type
    boosted = state.replace(boosts=state.boosts.replace(normal_boosts=state.boosts.normal_boosts.at[0, 2].set(2)))
    move = make_move('Body Press')
    _, _, base, _ = resolve_damaging_hit(key, state, 0, move)
    _, _, raised, _ = resolve_damaging_hit(key, boosted, 0, move)
    assert raised[0] > base[0]
    boosted = state.replace(boosts=state.boosts.replace(normal_boosts=state.boosts.normal_boosts.at[1, 2].set(6)))
    move = make_move('Sacred Sword')
    _, _, base, _ = resolve_damaging_hit(key, state, 0, move)
    _, _, raised, _ = resolve_damaging_hit(key, boosted, 0, move)
    assert raised[0] == base[0]


def test_trick_room_and_tailwind_order(neutral_type):
    """Trick Room reverses Speed only within a priority bracket."""
    from ninjax.battle import action_order
    key, state = neutral_type
    state = state.replace(tailwind=state.tailwind.at[0, 0].set(4))
    assert action_order(key, state, jnp.array([0, 0]))[0] == 0
    state = state.replace(trick_room_duration=5)
    assert action_order(key, state, jnp.array([0, 0]))[0] == 1
    assert action_order(key, state, jnp.array([1, 0]))[0] == 0
    # 257 Speed at +1 becomes floor(257*1.5)*2 = 770 under Tailwind.
    odd = state.replace(boosts=state.boosts.replace(normal_boosts=state.boosts.normal_boosts.at[0, 5].set(1)))
    raw = odd.active.stats[0, 5]
    assert odd.boosted_stats[0, 5] == jnp.floor(raw*1.5)*2


def test_stat_change_vectors_stack_with_scalar_moves():
    """One-stat and multi-stat specifications normalize to stackable shapes."""
    from ninjax.tree import stack
    single = make_move('Overheat')
    double = make_move('Close Combat')
    all_stats = make_move('Dragon Pulse').replace(
        self_stat=jnp.arange(1, 6), self_stages=jnp.ones(5, jnp.int32), self_chance=jnp.array([.1]))
    moves = stack([single, double, all_stats])
    assert moves.self_stat.shape == (3, 7)
    assert moves.current_pp.shape == (3, 1)
    assert jnp.array_equal(moves.self_stages[0], jnp.array([-2, 0, 0, 0, 0, 0, 0]))
    assert jnp.array_equal(moves.self_stages[1], jnp.array([-1, -1, 0, 0, 0, 0, 0]))
    assert jnp.array_equal(moves.self_stages[2], jnp.array([1, 1, 1, 1, 1, 0, 0]))


@pytest.mark.parametrize('chance', [0., 1.])
def test_one_chance_gates_the_whole_self_boost_vector(neutral_type, chance):
    """A self-effect chance controls all five ordinary stats together."""
    from ninjax.game_logic import do_damaging_move
    key, state = neutral_type
    move = make_move('Dragon Pulse').replace(self_stat=jnp.arange(1, 6),
        self_stages=jnp.ones(5, jnp.int32), self_chance=jnp.array([chance]))
    _, out = do_damaging_move(key, state, 0, move, 0, 0)
    assert jnp.all(out.boosts.normal_boosts[0, 1:] == int(chance))
