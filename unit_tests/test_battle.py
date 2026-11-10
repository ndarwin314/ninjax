import pytest
import jax.numpy as jnp
from jax import random
import jax
from ninjax.tree import stack

from ninjax.battle import Battle, BattleState, BattleParams, BattleAction, InformationMode, Phase, step_move, step_field, step_switch
from ninjax.pokemon import Pokemon
from ninjax.enum_types import Type, MoveType, Status, AbilityEnum, WeatherEnum, TerrainEnum, StatEnum
from ninjax.game_logic import move_used, do_damaging_move
from ninjax.move_effects import do_contact
from ninjax.move import Move
from ninjax.stats import StatTable, Nature, StatBoosts
from ninjax.side import BattleState, status_helper

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
    mons = stack([mon for _ in range(6)])
    mons = stack([mons, mons])
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
    mons = stack([mon for _ in range(6)])
    bad = Pokemon(type_list=jnp.array((Type.GRASS, 0)), moves=moves)
    bads = stack([bad for _ in range(6)])
    mons = stack([bads, mons])
    state = BattleState(team=mons)
    key = random.key(5)
    battle = Battle()
    params = BattleParams()
    return key, state

def test_contact(neutral_type):
    key, state = neutral_type
    do_contact(key, state, 0)


def test_entry_hazards_before_switch_abilities(neutral_type):
    key, original = neutral_type
    team = original.team.replace(current_hp=original.team.current_hp.at[0, 1, 0].set(1),
                                  ability=original.team.ability.at[0, 1, 0].set(AbilityEnum.DROUGHT))
    state = original.replace(team=team, stealth_rocks=original.stealth_rocks.at[0, 0].set(True))
    switch = jax.jit(lambda s: step_switch(key, s, 0, 1, False)[1])
    fainted = switch(state)
    assert fainted.active.current_hp[0, 0] == 0
    assert fainted.weather.weather == WeatherEnum.NONE
    guarded_team = team.replace(ability=team.ability.at[0, 1, 0].set(AbilityEnum.MAGIC_GUARD))
    guarded = switch(state.replace(team=guarded_team))
    assert guarded.active.current_hp[0, 0] == 1


def test_team_selection_and_visibility(neutral_type):
    key, original = neutral_type
    team = original.team.replace(species=jnp.arange(1, 13).reshape(2, 6, 1))
    env = Battle(team, BattleParams(team_preview=True))
    state, first = jax.jit(env.reset)(key)
    assert state.phase == Phase.TEAM_SELECTION
    assert jnp.all(first.observation.team["species"][:, 1] == 0)
    action = env.action_spec.generate_value()
    selection = jnp.zeros((2, 6), bool).at[0, 2].set(True).at[1, 3].set(True)
    step = jax.jit(env.step)
    invalid, ts = step(state, action)
    assert ts.extras['invalid_action']
    assert invalid.phase == Phase.TEAM_SELECTION
    ready, ts = step(state, action.replace(selection=selection))
    assert jnp.array_equal(ready.battle.active_index[:, 0], jnp.array([2, 3]))
    assert jnp.array_equal(ready.selected, selection)
    assert not jnp.any(env.legal_action_mask(ready)[:, 8:14])
    env.observation_spec.validate(ts.observation)
    assert ts.observation.team["species"][0, 1, 3] == 10
    assert ts.observation.team["species"][0, 1, 0] == 0
    preview = Battle(team, BattleParams(team_preview=True, information_mode=InformationMode.SPECIES))
    _, ts = preview.reset(key)
    assert jnp.array_equal(ts.observation.team["species"][0, 1], jnp.arange(7, 13))
    assert jnp.all(ts.observation.team["stats"][0, 1] == 0)


def test_forced_struggle_and_knockout_outcomes(neutral_type):
    key, original = neutral_type
    moves = original.team.moves
    moves = moves.replace(current_pp=jnp.zeros_like(moves.current_pp))
    env = Battle(original.team.replace(moves=moves))
    state, _ = env.reset(key)
    assert jnp.all(env.legal_action_mask(state)[:, 4])
    after, _ = jax.jit(env.step)(state, jnp.array([4, 4]))
    assert after.turn == 1
    assert jnp.all(after.battle.active.current_hp < state.battle.active.current_hp)
    assert jnp.all(after.battle.team.moves.current_pp == 0)
    # Self-KO and recoil both ending the battle have different winners.
    base = original.replace(roster=jnp.zeros((2, 6), bool).at[:, 0].set(True))
    base = base.replace(team=base.team.replace(current_hp=base.team.current_hp.at[:, 0, 0].set(1)))
    move = base.active[0].moves[0]
    run = jax.jit(lambda m: do_damaging_move(key, base, 0, m, 0, 0)[1])
    assert run(move.replace(self_ko=jnp.array([True]))).winner == 1
    assert run(move.replace(recoil_percent=jnp.array([1.]))).winner == 0
    perish = base.replace(perish_turns=jnp.ones((2, 1), jnp.int32),
                          team=base.team.replace(stat_table=base.team.stat_table.replace(
                              base_stats=base.team.stat_table.base_stats.at[0, :, StatEnum.SPEED].set(1))))
    _, out = jax.jit(step_field)(key, perish)
    assert out.winner == 0
    _, out = jax.jit(step_field)(key, perish.replace(trick_room_duration=jnp.int32(3)))
    assert out.winner == 1


def test_generic_move_effects(neutral_type):
    key, original = neutral_type
    moves = original.team.moves
    configured = moves.replace(move_type=jnp.full_like(moves.move_type, MoveType.STATUS),
        status_effect=jnp.full_like(moves.status_effect, Status.BURN),
        boost_stat=jnp.full_like(moves.boost_stat, StatEnum.ATTACK),
        boost_stages=jnp.full_like(moves.boost_stages, -1),
        weather_effect=jnp.full_like(moves.weather_effect, WeatherEnum.RAIN),
        terrain_effect=jnp.full_like(moves.terrain_effect, TerrainEnum.GRASSY),
        hazard_effect=jnp.full_like(moves.hazard_effect, 3))
    state = original.replace(team=original.team.replace(moves=configured))
    run = jax.jit(lambda k, s: move_used(k, s, 0, 0))
    _, result = run(key, state)
    assert result.active.status[1, 0] == Status.BURN
    assert result.boosts.normal_boosts[1, StatEnum.ATTACK] == -1
    assert result.weather.weather == WeatherEnum.RAIN
    assert result.terrain.terrain == TerrainEnum.GRASSY
    assert result.spikes[1, 0] == 1
    assert result.active.moves.current_pp[0, 0, 0] == 7
    self_moves = configured.replace(target_self=jnp.ones_like(configured.target_self),
        status_effect=jnp.zeros_like(configured.status_effect),
        boost_stages=jnp.full_like(configured.boost_stages, 2))
    _, boosted = run(key, original.replace(team=original.team.replace(moves=self_moves)))
    assert boosted.boosts.normal_boosts[0, StatEnum.ATTACK] == 2
    unsupported = configured.replace(implemented=jnp.zeros_like(configured.implemented))
    _, noop = run(key, original.replace(team=original.team.replace(moves=unsupported)))
    assert jnp.array_equal(noop.team.current_hp, original.team.current_hp)
    assert jnp.array_equal(noop.team.status, original.team.status)
    assert jnp.all(noop.spikes == 0)

def test_stat_helper(neutral_type):
    key, state = neutral_type
    status_helper(key, state.active[0], jnp.ones(1))


def test_compiled_episode_reset_and_timeout(neutral_type):
    key, battle_state = neutral_type
    env = Battle(teams=battle_state.team, params=BattleParams(max_steps_in_episode=1))
    state, first = jax.jit(env.reset)(key)
    next_state, last = jax.jit(env.step)(state, jnp.array([0, 0], dtype=jnp.int32))
    assert first.first()
    assert last.last()
    assert jnp.all(last.discount == 1)
    assert state.turn == 0
    assert next_state.turn == 1


def test_compiled_damage_transforms(neutral_type):
    key, state = neutral_type

    def hit(key, state):
        return do_damaging_move(key, state, 0, state.active[0].moves[0], 0, 0)

    assert not jax.config.jax_disable_jit
    expected_key, expected = hit(key, state)
    compiled_key, compiled = jax.jit(hit)(key, state)
    assert jnp.array_equal(compiled.team.current_hp, expected.team.current_hp)

    keys = random.split(key, 2)
    _, batched = jax.jit(jax.vmap(hit))(keys, stack([state, state]))
    assert batched.team.current_hp.shape == (2, 2, 6, 1)

    def turn(carry, unused):
        next_carry = hit(*carry)
        return next_carry, next_carry[1].team.current_hp

    (_, final), hp = jax.jit(lambda carry: jax.lax.scan(turn, carry, None, length=2))((key, state))
    assert hp.shape == (2, 2, 6, 1)
    assert jnp.all(final.team.current_hp <= state.team.current_hp)


def test_decision_phases(neutral_type):
    key, battle = neutral_type
    pp = battle.team.moves
    # Player zero's first move is a pivot and has priority over player one.
    moves = pp.replace(switch_out=pp.switch_out.at[0, :, 0, 0].set(True),
                       priority=pp.priority.at[0, :, 0, 0].set(1))
    env = Battle(teams=battle.team.replace(moves=moves))
    state, _ = env.reset(key)
    step = jax.jit(env.step)
    waiting, timestep = step(state, jnp.array([0, 0]))
    assert waiting.phase == Phase.REPLACEMENT
    assert jnp.array_equal(waiting.acting_player_mask, jnp.array([True, False]))
    assert waiting.turn == 0
    assert waiting.cursor == 1
    assert waiting.battle.team.moves.current_pp[0, 0, 0, 0] == 7
    assert waiting.battle.team.moves.current_pp[1, 0, 0, 0] == 8
    from ninjax.events import EventKind
    assert timestep.observation.transitions.kind[0, 0] == EventKind.MOVE
    assert jnp.sum(timestep.observation.transitions.kind[0] == EventKind.MOVE) == 1
    resumed, resumed_ts = step(waiting, jnp.array([9, -99]))
    assert resumed_ts.observation.transitions.kind[0, 0] == EventKind.SWITCH
    assert jnp.sum(resumed_ts.observation.transitions.kind[0] == EventKind.MOVE) == 1
    assert resumed.phase == Phase.TURN
    assert resumed.battle.active_index[0, 0] == 1
    assert resumed.turn == 1
    assert resumed.battle.team.moves.current_pp[1, 0, 0, 0] == 7
    assert resumed.battle.team.current_hp[0, 1, 0] < state.battle.team.current_hp[0, 1, 0]
    # Both players replace fainted actives without starting another turn.
    team = resumed.battle.team.replace(current_hp=resumed.battle.team.current_hp.at[0, 1, 0].set(0).at[1, 0, 0].set(0))
    fainted = resumed.replace(battle=resumed.battle.replace(team=team),
                              phase=jnp.int32(Phase.REPLACEMENT), acting_player_mask=jnp.ones(2, bool))
    replaced, _ = step(fainted, jnp.array([10, 11]))
    assert replaced.turn == 1
    assert jnp.array_equal(replaced.battle.active_index[:, 0], jnp.array([2, 3]))
    invalid, info = step(replaced, jnp.array([8, 999]))
    assert info.extras['invalid_action']
    assert not jnp.any(info.observation.transitions.valid)
    assert invalid.turn == replaced.turn
    # A faster knockout skips the fainted opponent's queued move.
    plain_moves = state.battle.team.moves.replace(switch_out=jnp.zeros_like(moves.switch_out))
    weak_team = state.battle.team.replace(moves=plain_moves,
                                         current_hp=state.battle.team.current_hp.at[1, 0, 0].set(1))
    knockout, _ = step(state.replace(battle=state.battle.replace(team=weak_team)), jnp.array([0, 0]))
    assert knockout.phase == Phase.REPLACEMENT
    assert jnp.array_equal(knockout.acting_player_mask, jnp.array([False, True]))
    assert knockout.battle.team.moves.current_pp[1, 0, 0, 0] == 8
    assert knockout.turn == 1

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
        mons = stack([mon for _ in range(6)])
        mons = stack([mons, mons])
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
        key, state = do_damaging_move(key, state, 0, move, 0, 0)
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
        key, state = do_damaging_move(key, state, 0, move, 0, 0)
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
        key, state = do_damaging_move(key, state, 0, move, 0, 0)
        assert state.active[1].current_hp[0] == damage


def test_ability_speed_and_marvel_scale_are_per_pokemon(neutral_type):
    from ninjax.enum_types import Weather
    key, state = neutral_type
    abilities = state.team.ability.at[0, 0, 0].set(AbilityEnum.CHLOROPHYLL)
    abilities = abilities.at[1, 0, 0].set(AbilityEnum.MARVEL_SCALE)
    statuses = state.team.status.at[1, 0, 0].set(Status.BURN)
    state = state.replace(team=state.team.replace(ability=abilities, status=statuses),
                          weather=Weather(WeatherEnum.SUN, jnp.int32(5)))
    raw = state.active.stats
    stats = jax.jit(lambda s: s.boosted_stats)(state)
    assert stats[0, StatEnum.SPEED] == 2 * raw[0, StatEnum.SPEED]
    assert stats[1, StatEnum.SPEED] == raw[1, StatEnum.SPEED]
    assert stats[1, StatEnum.DEFENSE] == jnp.floor(raw[1, StatEnum.DEFENSE] * 1.5)
    assert stats[1, StatEnum.SPECIAL_DEFENSE] == raw[1, StatEnum.SPECIAL_DEFENSE]
    batched = jax.jit(jax.vmap(lambda s: s.boosted_stats))(stack([state, state]))
    assert jnp.array_equal(batched, jnp.stack([stats, stats]))


def test_damage_ability_effectiveness_and_gravity(neutral_type):
    from ninjax.damage import compute_damage_multipliers
    from ninjax.enum_types import Weather
    key, state = neutral_type
    attacker, defender = state.active[0], state.active[1]
    move = attacker.moves[0]
    def damage(attacker, defender, move, gravity):
        return compute_damage_multipliers(key, attacker, defender, Weather(0, 0), move,
                                          jnp.array([100]), jnp.array([False]), 1., gravity)[0]
    damage = jax.jit(damage)
    water = move.replace(type=jnp.array([Type.WATER]))
    fire = defender.replace(type_list=jnp.array([Type.FIRE, Type.NONE]))
    grass = defender.replace(type_list=jnp.array([Type.GRASS, Type.NONE]))
    assert damage(attacker, fire, water, False) == 200
    assert damage(attacker, fire.replace(ability=jnp.array([AbilityEnum.FILTER])), water, False) == 150
    assert damage(attacker.replace(ability=jnp.array([AbilityEnum.NEUROFORCE])), fire, water, False) == 250
    assert damage(attacker, grass, water, False) == 50
    assert damage(attacker.replace(ability=jnp.array([AbilityEnum.TINTED_LENS])), grass, water, False) == 100
    ground = move.replace(type=jnp.array([Type.GROUND]))
    levitate = defender.replace(type_list=jnp.array([Type.NORMAL, Type.NONE]), ability=jnp.array([AbilityEnum.LEVITATE]))
    assert damage(attacker, levitate, ground, False) == 0
    assert damage(attacker, levitate, ground, True) == 100
    flying = levitate.replace(type_list=jnp.array([Type.FLYING, Type.NONE]), ability=jnp.array([0]))
    assert damage(attacker, flying, ground, False) == 0
    assert damage(attacker, flying, ground, True) == 100


def test_levitate_blocks_ground_moves_unless_gravity_is_active(neutral_type):
    key, state = neutral_type
    team = state.team.replace(
        moves=state.team.moves.replace(type=jnp.full_like(state.team.moves.type, Type.GROUND)),
        ability=state.team.ability.at[1, 0, 0].set(AbilityEnum.LEVITATE))
    state = state.replace(team=team)
    apply = jax.jit(lambda s: move_used(key, s, 0, 0)[1])
    assert apply(state).active.current_hp[1, 0] == state.active.current_hp[1, 0]
    grounded = apply(state.replace(gravity_duration=jnp.int32(3)))
    assert grounded.active.current_hp[1, 0] < state.active.current_hp[1, 0]


@pytest.mark.parametrize('ability,move_type', [(AbilityEnum.STORM_DRAIN, Type.WATER),
                                               (AbilityEnum.LIGHTNING_ROD, Type.ELECTRIC)])
def test_redirect_abilities_belong_to_defender(neutral_type, ability, move_type):
    key, state = neutral_type
    moves = state.team.moves.replace(type=jnp.full_like(state.team.moves.type, move_type))
    team = state.team.replace(moves=moves, ability=state.team.ability.at[1, 0, 0].set(ability))
    state = state.replace(team=team)
    apply = jax.jit(lambda s: move_used(key, s, 0, 0)[1])
    result = apply(state)
    assert result.active.current_hp[1, 0] == state.active.current_hp[1, 0]
    assert result.boosts.normal_boosts[1, StatEnum.SPECIAL_ATTACK] == 1
    # The attacker's own ability must not redirect its move.
    team = team.replace(ability=team.ability.at[1, 0, 0].set(0).at[0, 0, 0].set(ability))
    ordinary = apply(state.replace(team=team))
    assert ordinary.active.current_hp[1, 0] < state.active.current_hp[1, 0]
    assert ordinary.boosts.normal_boosts[1, StatEnum.SPECIAL_ATTACK] == 0


def test_flash_fire_activation_and_switch_reset(neutral_type):
    from ninjax.side import clear_volatile_status
    key, state = neutral_type
    moves = state.team.moves.replace(type=jnp.full_like(state.team.moves.type, Type.FIRE),
                                     accuracy=jnp.zeros_like(state.team.moves.accuracy))
    team = state.team.replace(moves=moves, ability=state.team.ability.at[1, 0, 0].set(AbilityEnum.FLASH_FIRE))
    state = state.replace(team=team)
    hit = jax.jit(lambda s: move_used(key, s, 0, 0)[1])
    activated = hit(state)
    assert activated.active.current_hp[1, 0] == state.active.current_hp[1, 0]
    assert activated.volatile_status.flash_fire[1, 0]
    assert not activated.volatile_status.flash_fire[0, 0]
    repeated = hit(activated)
    assert repeated.volatile_status.flash_fire[1, 0]
    assert repeated.active.current_hp[1, 0] == state.active.current_hp[1, 0]
    retaliate = jax.jit(lambda s: do_damaging_move(key, s, 1, s.active[1].moves[0], 0, 0)[1])
    unboosted = activated.replace(volatile_status=activated.volatile_status.replace(
        flash_fire=activated.volatile_status.flash_fire.at[1, 0].set(False)))
    assert retaliate(activated).active.current_hp[0, 0] < retaliate(unboosted).active.current_hp[0, 0]
    # Other fields reset for the leaving side and remain intact for its opponent.
    marked = repeated.volatile_status.replace(confusion_turns=jnp.array([[3], [2]]),
        transformed=jnp.array([[True], [True]]), flash_fire=jnp.ones((2, 1), bool))
    cleared = jax.jit(clear_volatile_status)(repeated.replace(volatile_status=marked), 1)
    assert cleared.volatile_status.confusion_turns[1, 0] == 0
    assert not cleared.volatile_status.transformed[1, 0]
    assert not cleared.volatile_status.flash_fire[1, 0]
    assert cleared.volatile_status.confusion_turns[0, 0] == 3
    assert cleared.volatile_status.flash_fire[0, 0]
    switched = jax.jit(lambda s: step_switch(key, s, 1, 1, False)[1])(repeated.replace(volatile_status=marked))
    assert switched.active_index[1, 0] == 1
    assert not switched.volatile_status.flash_fire[1, 0]
    assert switched.volatile_status.flash_fire[0, 0]
    batched = jax.jit(jax.vmap(clear_volatile_status, in_axes=(0, None)))(stack([state, repeated]), 1)
    assert not jnp.any(batched.volatile_status.flash_fire[:, 1, 0])


def test_flash_fire_modifies_only_fire_damage_when_active(neutral_type):
    from ninjax.damage import compute_base_damage
    from ninjax.enum_types import StatEnum
    key, state = neutral_type
    active = state.active[0]
    move = active.moves[0].replace(type=jnp.array([Type.FIRE]), offensive_stat=jnp.array([StatEnum.ATTACK]),
                                   defensive_stat=jnp.array([StatEnum.DEFENSE]))
    stats = jnp.full(6, 100)
    damage = jax.jit(lambda m, ability, active: compute_base_damage(
        ability, jnp.array([0]), jnp.array([1.]), jnp.array([100]), stats, stats,
        m, m.base_power, jnp.array([0]), jnp.array([0]), active)[0])
    flash_fire = jnp.array([AbilityEnum.FLASH_FIRE])
    assert damage(move, flash_fire, False) == 69
    assert damage(move, flash_fire, True) == 102
    special = move.replace(move_type=jnp.array([MoveType.SPECIAL]),
                           offensive_stat=jnp.array([StatEnum.SPECIAL_ATTACK]),
                           defensive_stat=jnp.array([StatEnum.SPECIAL_DEFENSE]))
    assert damage(special, flash_fire, True) == 102
    assert damage(move.replace(type=jnp.array([Type.WATER])), flash_fire, True) == 69
    assert damage(move, jnp.array([0]), True) == 69


def test_sturdy_only_survives_lethal_full_health_attack(neutral_type):
    from ninjax.side import take_damage_value
    key, state = neutral_type
    team = state.team.replace(ability=state.team.ability.at[1, 0, 0].set(AbilityEnum.STURDY))
    state = state.replace(team=team)
    hp = state.active.current_hp[1, 0]
    hit = jax.jit(lambda s, amount, attack: take_damage_value(s, 1, amount, attack))
    assert hit(state, hp, True).active.current_hp[1, 0] == 1
    assert hit(state, 10, True).active.current_hp[1, 0] == hp - 10
    assert hit(state, hp, False).active.current_hp[1, 0] == 0
    damaged = hit(state, 10, True)
    assert hit(damaged, hp, True).active.current_hp[1, 0] == 0


def test_multiscale_reduces_attacks_only_at_full_hp(neutral_type):
    from ninjax.side import take_damage_value
    key, state = neutral_type
    team = state.team.replace(ability=state.team.ability.at[1, 0, 0].set(AbilityEnum.MULTISCALE))
    state = state.replace(team=team)
    hp = state.active.current_hp[1, 0]
    hit = jax.jit(lambda s, amount, attack: take_damage_value(s, 1, amount, attack))
    reduced = hit(state, 100, True)
    assert reduced.active.current_hp[1, 0] == hp - 50
    assert hit(reduced, 100, True).active.current_hp[1, 0] == hp - 150
    assert hit(state, 100, False).active.current_hp[1, 0] == hp - 100


def test_flash_fire_activation_under_vmap_and_scan(neutral_type):
    key, state = neutral_type
    moves = state.team.moves.replace(type=jnp.full_like(state.team.moves.type, Type.FIRE))
    team = state.team.replace(moves=moves, ability=state.team.ability.at[1, 0, 0].set(AbilityEnum.FLASH_FIRE))
    protected = state.replace(team=team)
    ordinary = state.replace(team=team.replace(ability=team.ability.at[1, 0, 0].set(0)))
    def hit(s):
        return move_used(key, s, 0, 0)[1]
    results = jax.jit(jax.vmap(hit))(stack([protected, ordinary]))
    assert jnp.array_equal(results.volatile_status.flash_fire[:, 1, 0], jnp.array([True, False]))
    assert results.team.current_hp[0, 1, 0, 0] == protected.team.current_hp[1, 0, 0]
    assert results.team.current_hp[1, 1, 0, 0] < ordinary.team.current_hp[1, 0, 0]
    def turn(carry, unused):
        updated = hit(carry)
        return updated, updated.volatile_status.flash_fire[1, 0]
    final, flags = jax.jit(lambda s: jax.lax.scan(turn, s, None, length=2))(protected)
    assert jnp.all(flags)
    assert final.team.moves.current_pp[0, 0, 0, 0] == 6


def test_observation_events_and_private_hp(neutral_type):
    from ninjax.events import EventKind, Cause, EventBuffer
    from ninjax.side import take_damage_value, set_status, add_boosts
    key, battle = neutral_type
    env = Battle(battle.team)
    state, first = env.reset(key)
    assert not jnp.any(first.observation.transitions.valid)
    assert jnp.all(first.observation.team['stats'][:, 1] == 0)
    assert jnp.all(first.observation.team['current_hp'][:, 1] == 0)
    assert jnp.array_equal(first.observation.team['stats'][0, 0], battle.team.stat_table.stats[0])
    def effects(b):
        b = b.replace(events=EventBuffer.empty(), event_source=jnp.int32(0), event_cause=jnp.int32(Cause.MOVE))
        b = take_damage_value(b, 1, 13, True)
        b = take_damage_value(b, 1, -5, False)
        _, b = set_status(key, b, 1, jnp.array([Status.BURN]))
        return add_boosts(b, 0, StatEnum.ATTACK, 2)
    changed = jax.jit(effects)(battle)
    obs = env.observe(state.replace(battle=changed))
    env.observation_spec.validate(obs)
    e = obs.transitions
    assert jnp.array_equal(e.kind[0, :4], jnp.array([EventKind.HP, EventKind.HP, EventKind.STATUS, EventKind.BOOST]))
    assert jnp.array_equal(e.hp_change[1, :2], jnp.array([-13, 5]))
    assert jnp.all(e.hp_change[0] == 0)
    assert jnp.all(e.before[0, :2] == 0)
    assert jnp.array_equal(e.target[:, 0], jnp.array([1, 0]))
    assert not jnp.any(e.overflow)
    batched = jax.jit(jax.vmap(effects))(stack([battle, battle]))
    assert jnp.all(batched.events.count == 4)
    def turn(b, _):
        b = effects(b)
        return b, b.events.count
    final, counts = jax.jit(lambda b: jax.lax.scan(turn, b, None, length=2))(battle)
    assert counts.shape == (2,)


def test_transition_buffer_bounds_and_hp_clipping(neutral_type):
    from ninjax.events import CAPACITY, EventKind, emit
    from ninjax.side import take_damage_value
    _, battle = neutral_type
    dead = jax.jit(lambda b: take_damage_value(b, 1, 100000, True))(battle)
    assert dead.events.hp_change[0] == -battle.active.current_hp[1, 0]
    assert dead.events.kind[1] == EventKind.FAINT
    full = jax.jit(lambda b: jax.lax.fori_loop(0, CAPACITY + 2,
        lambda _, b: emit(b, EventKind.MOVE), b))(battle)
    assert full.events.count == CAPACITY
    assert full.events.overflow
    assert jnp.all(full.events.valid)


def test_full_observation_rollout_transforms(neutral_type):
    key, battle = neutral_type
    env = Battle(battle.team, BattleParams(max_steps_in_episode=1))
    def episode(key):
        state, first = env.reset(key)
        def turn(state, _):
            state, ts = env.step(state, jnp.array([0, 0]))
            return state, ts.observation
        return jax.lax.scan(turn, state, None, length=2)
    final, observations = jax.jit(jax.vmap(episode))(random.split(key, 2))
    assert jnp.all(final.phase == Phase.TERMINAL)
    assert observations.team['species'].shape == (2, 2, 2, 2, 6)
    assert jnp.all(observations.transitions.count[:, 0] > 0)
    assert jnp.all(observations.transitions.count[:, 1] == 0)
    assert not jnp.any(observations.transitions.overflow)


def test_residual_effect_order_and_events(neutral_type):
    from ninjax.game_logic import end_turn_damage
    from ninjax.events import EventKind
    from ninjax.enum_types import Weather, Terrain
    _, state = neutral_type
    state = state.replace(
        weather=Weather(WeatherEnum.SANDSTORM, 5),
        terrain=Terrain(TerrainEnum.GRASSY, 5),
        toxic_counter=jnp.array([[2], [0]]),
        team=state.team.replace(status=state.team.status.at[0, 0, 0].set(Status.TOXIC)
                               .at[1, 0, 0].set(Status.BURN)))
    out = jax.jit(end_turn_damage)(state)
    hp = state.active.max_hp
    sand = jnp.maximum(1, hp // 16)
    poison = jnp.maximum(1, hp[0] // 8)
    expected = jnp.array([-sand[0], -sand[1], sand[0], sand[1], -poison, -sand[1]])
    assert jnp.array_equal(out.events.hp_change[:6], expected)
    assert jnp.all(out.events.kind[:6] == EventKind.HP)
    assert jnp.array_equal(out.events.target[:6], jnp.array([0, 1, 0, 1, 0, 1]))
    assert out.events.count == 6
