"""Opt-in batched and scanned compatibility checks."""
from unit_tests.test_battle import (
    AbilityEnum,
    Battle,
    BattleParams,
    Phase,
    StatEnum,
    Status,
    Type,
    WeatherEnum,
    do_damaging_move,
    jax,
    jnp,
    move_used,
    neutral_type,
    pytest,
    random,
    stack,
    step_switch,
)
from ninjax.enum_types import Weather
from ninjax.side.updates import clear_volatile_status
from ninjax.events import EventKind, Cause, EventBuffer
from ninjax.side.hp_effects import take_damage_value
from ninjax.side.status_effects import set_status
from ninjax.side.updates import add_boosts


@pytest.mark.jit
def test_compiled_damage_transforms(neutral_type):
    """Validate damage transitions under actual jit, vmap, and scan."""
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


def test_ability_speed_and_marvel_scale_are_per_pokemon(neutral_type):
    """Check compatibility with batched JAX transforms."""

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


def test_flash_fire_activation_and_switch_reset(neutral_type):
    """Check compatibility with batched JAX transforms."""

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


@pytest.mark.jit
def test_flash_fire_activation_under_vmap_and_scan(neutral_type):
    """Compile switch-cleared ability state across batched and scanned hits."""
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
    """Check compatibility with batched JAX transforms."""


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


@pytest.mark.jit
def test_full_observation_rollout_transforms(neutral_type):
    """Compile full observations and terminal transitions in batched rollouts."""
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
