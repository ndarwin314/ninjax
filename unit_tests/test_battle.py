import pytest
import jax.numpy as jnp
from jax import random
import jax
import dataclass_array as dca

from ninjax.battle import Battle, BattleState, BattleParams, step_move
from ninjax.pokemon import Pokemon
from ninjax.enum_types import Type, MoveType, Status, AbilityEnum
from ninjax.game_logic import move_used, do_move_damage, do_contact
from ninjax.move import Move
from ninjax.stats import StatTable, Nature, StatBoosts
from ninjax.side import BattleState

jax.config.update("jax_traceback_filtering", "off")

@pytest.fixture
def neutral_type():
    type_list = jnp.array([0,0])
    moves = Move(
        move_type=jnp.zeros((4, 1)),
        max_pp=8 * jnp.ones((4, 1)),
        current_pp=8 * jnp.ones((4, 1)),
        type=jnp.ones((4, 1)),
        base_power=80 * jnp.ones((4, 1)),
        accuracy=jnp.ones((4, 1)),
        priority=jnp.zeros((4, 1)),
        offensive_stat=jnp.ones((4, 1)),
        defensive_stat=4 * jnp.ones((4, 1)),
        crit_stage=jnp.zeros((4, 1))
    )

    mon = Pokemon(type_list=type_list, moves=moves)
    mons = dca.stack([mon for _ in range(6)])
    mons = dca.stack([mons, mons])
    state = BattleState(team=mons)
    key = random.key(5)
    return key, state

@pytest.fixture
def battle_state():
    jax.config.update("jax_traceback_filtering", "off")

    type_list = jnp.array([Type.FIRE, Type.ROCK])
    moves = Move(
        move_type=jnp.zeros((4, 1)),
        max_pp=8 * jnp.ones((4, 1)),
        current_pp=8 * jnp.ones((4, 1)),
        type=jnp.zeros((4, 1)),
        base_power=80 * jnp.ones((4, 1)),
        accuracy=jnp.ones((4, 1)),
        priority=jnp.zeros((4, 1)),
        offensive_stat=jnp.ones((4, 1)),
        defensive_stat=4 * jnp.ones((4, 1)),
        crit_stage=jnp.zeros((4, 1))
    )

    mon = Pokemon(type_list=type_list, moves=moves)
    mons = dca.stack([mon for _ in range(6)])
    bad = Pokemon(type_list=jnp.array((Type.GRASS, 0)), moves=moves)
    bads = dca.stack([bad for _ in range(6)])
    mons = dca.stack([bads, mons])
    state = BattleState(team=mons)
    key = random.key(5)
    battle = Battle()
    params = BattleParams()
    return key, state

class TestStats:
    @pytest.mark.parametrize(
        "base_stats, evs, ivs, level, computed_stats",
        [(100, 0, 0, 100, (310, 205)),
         (100, 1, 0, 100, (310, 205)),
         (100, 4, 0, 100, (311, 206)),
         (100, 0, 1, 100, (311, 206)),
         (100, 0, 31, 100, (341, 236)),
         (100, 252, 31, 100, (404, 299)),
         (100, 0, 0, 31, (103, 67)),
         (100, 0, 31, 31, (112, 76)),
         (100, 0, 31, 50, (175, 120)),
         (100, 4, 31, 50, (176, 121)),
         (100, 8, 31, 50, (176, 121)),
         (100, 12, 31, 50, (177, 122))])
    def test_computed_stats(self, base_stats, evs, ivs, level, computed_stats):
        stats = StatTable(level=[level], base_stats=6*[base_stats], evs=6*[evs], ivs=6*[ivs])
        stats = stats.stats
        assert stats[0], stats[1] == computed_stats

    @pytest.mark.parametrize(
        "increased, decreased, computed_stats",
        [(1,1, (299, 299)),
         (1,2, (328, 269))])
    def test_natures(self, increased, decreased, computed_stats):
        nature = Nature(increased=[increased], decreased=[decreased])
        stats = StatTable(level=[100], base_stats=6 * [100], evs=6 * [252], ivs=6 * [31], nature=nature)
        stats = stats.stats
        assert stats[increased], stats[decreased] == computed_stats

    @pytest.mark.parametrize(
        "stage, stat",
        [(-6, 64),
         (-5, 73),
         (-4, 85),
         (-3, 102),
         (-2, 128),
         (-1, 171),
         (0, 257),
         (1, 385),
         (2, 514),
         (3, 642),
         (4, 771),
         (5, 899),
         (6,1028)])
    def test_boosts(self, stage, stat):
        type_list = jnp.array([1, 2])
        moves = Move(
            move_type=jnp.zeros((4, 1)),
            max_pp=8 * jnp.ones((4, 1)),
            current_pp=8 * jnp.ones((4, 1)),
            type=jnp.zeros((4, 1)),
            base_power=80 * jnp.ones((4, 1)),
            accuracy=jnp.ones((4, 1)),
            priority=jnp.zeros((4, 1)),
            offensive_stat=jnp.ones((4, 1)),
            defensive_stat=4 * jnp.ones((4, 1)),
            crit_stage=jnp.zeros((4, 1))
        )

        mon = Pokemon(type_list=type_list, moves=moves)
        mons = dca.stack([mon for _ in range(6)])
        boosts = StatBoosts(normal_boosts=stage*jnp.ones((2,6), dtype=int))
        state = BattleState(team=mons, boosts=boosts)
        boosted_stats = state.boosted_stats[0]
        assert boosted_stats[1]==stat

class TestDamage:
    @pytest.mark.parametrize(
        "type_, health",
        [(Type.GHOST, 293),
         (Type.FIGHTING, 224),
         (Type.GROUND, 86),
         (Type.FLYING, 328),
         (Type.FIRE, 345)]
    )
    def test_damage_no_modifiers(self, battle_state, type_, health):
        key, state = battle_state
        move = Move(
            move_type= jnp.zeros((1,)),
            max_pp=8 * jnp.ones((1,)),
            current_pp=8 * jnp.ones((1,)),
            type=type_ * jnp.ones((1,)),
            base_power=80 * jnp.ones((1,)),
            accuracy=jnp.ones((1,)),
            priority=jnp.zeros((1,)),
            offensive_stat=jnp.ones((1,)),
            defensive_stat=4 * jnp.ones((1,)),
            crit_stage=jnp.zeros((1,))
        )
        key, state = do_move_damage(key, state, 0, move, 0)
        assert state.active[1].current_hp[0]==health

    def test_damage_stab(self, battle_state):
        key, state = battle_state
        move = Move(
            move_type=jnp.zeros((1,)),
            max_pp=8 * jnp.ones((1,)),
            current_pp=8 * jnp.ones((1,)),
            type=Type.GRASS * jnp.ones((1,)),
            base_power=80 * jnp.ones((1,)),
            accuracy=jnp.ones((1,)),
            priority=jnp.zeros((1,)),
            offensive_stat=jnp.ones((1,)),
            defensive_stat=4 * jnp.ones((1,)),
            crit_stage=jnp.zeros((1,))
        )
        key, state = do_move_damage(key, state, 0, move, 0)
        assert state.active[1].current_hp[0] == 259

    @pytest.mark.parametrize(
        "is_physical, is_guts, damage",
        [(0, 0, 293),
         (1, 0, 328),
         (0, 1, 293),
         (1, 1, 260)]
    )
    def test_burn(self, neutral_type, is_physical, is_guts, damage):
        key, state = neutral_type
        teams = state.team.replace(
            status=Status.BURN*jnp.ones((1,)),
            ability=is_guts*AbilityEnum.GUTS*jnp.ones((1,)))
        state = state.replace(team=teams)
        move = Move(
            move_type=(1-is_physical) * jnp.ones((1,)),
            max_pp=8 * jnp.ones((1,)),
            current_pp=8 * jnp.ones((1,)),
            type=jnp.ones((1,)),
            base_power=80 * jnp.ones((1,)),
            accuracy=jnp.ones((1,)),
            priority=jnp.zeros((1,)),
            offensive_stat=(1+3*(1-is_physical))*jnp.ones((1,)),
            defensive_stat=(2+3*(1-is_physical)) * jnp.ones((1,)),
            crit_stage=jnp.zeros((1,))
        )
        key, state = do_move_damage(key, state, 0, move, 0)
        assert state.active[1].current_hp[0] == damage
