"""Controlled checks for multi-hit rules and per-hit state transitions."""
import jax
import jax.numpy as jnp
import pytest
from unit_tests.test_battle import neutral_type
from ninjax.enum_types import AbilityEnum as A, ItemEnum as I, StatEnum as S
from ninjax.events import EventKind
from ninjax.game_logic import resolve_multi_hit, sample_hit_count
from ninjax.prototype import make_move


def hit_events(state, target=1):
    """Select damage events for a target from the fixed event buffer."""
    events = state.events
    return events.valid & (events.kind == EventKind.HP) & (events.target == target) & (events.hp_change < 0)


def test_counts_distribution_and_modifiers():
    """Check Gen 9 hit distributions and hit-count modifiers."""
    keys = jax.random.split(jax.random.key(9), 20000)
    draw = jax.jit(jax.vmap(sample_hit_count, in_axes=(0, None, None, None)))
    _, single = draw(keys, 1, A.NONE, I.NONE)
    assert jnp.all(single == 1)
    _, values = draw(keys, 2, A.NONE, I.NONE)
    frequencies = jnp.bincount(values, length=6) / values.size
    assert jnp.all(jnp.abs(frequencies[2:6] - jnp.array([.35, .35, .15, .15])) < .015)
    _, dice = draw(keys, 2, A.NONE, I.LOADED_DICE)
    assert jnp.all((dice == 4) | (dice == 5))
    assert .48 < jnp.mean(dice == 4) < .52
    _, linked = draw(keys, 2, A.SKILL_LINK, I.NONE)
    assert jnp.all(linked == 5)
    _, population = draw(keys, 4, A.NONE, I.LOADED_DICE)
    assert jnp.min(population) == 4 and jnp.max(population) == 10
    _, fixed = draw(keys, 3, A.SKILL_LINK, I.LOADED_DICE)
    assert jnp.all(fixed == 3)


def test_catalogue_distinguishes_multi_hit_rules():
    """Move metadata describes each implemented curated hit pattern."""
    assert int(make_move('Bullet Seed').hit_rule[0]) == 2
    assert int(make_move('Surging Strikes').hit_rule[0]) == 3
    assert bool(make_move('Surging Strikes').always_crit[0])
    assert bool(make_move('Triple Axel').per_hit_accuracy[0])
    assert bool(make_move('Triple Axel').escalating_power[0])
    assert int(make_move('Population Bomb').hit_rule[0]) == 4
    assert not bool(make_move('Beat Up').implemented[0])


def test_technician_checks_each_triple_axel_power(neutral_type):
    """Technician boosts each of Triple Axel's 20/40/60-power hits."""
    from ninjax.damage import compute_base_power
    _, state = neutral_type
    attacker = state.active[0].replace(ability=jnp.array([A.TECHNICIAN]))
    defender = state.active[1]
    move = make_move('Triple Axel')
    powers = [compute_base_power(attacker, defender, move.replace(base_power=jnp.array([base])), state.terrain)[0]
              for base in (20, 40, 60)]
    assert jnp.array_equal(jnp.array(powers), jnp.array([30, 60, 90]))


def test_stamina_responds_to_special_hits(neutral_type):
    """Stamina reacts to damaging special hits as well as physical hits."""
    from ninjax.enum_types import MoveType
    from ninjax.move_effects import after_every_hit
    key, state = neutral_type
    team = state.team.replace(ability=state.team.ability.at[1, 0, 0].set(A.STAMINA))
    move = make_move('Bullet Seed').replace(move_type=jnp.array([MoveType.SPECIAL]))
    _, out = jax.jit(after_every_hit)(key, state.replace(team=team), 0, move, jnp.array([False]))
    assert out.boosts.normal_boosts[1, S.DEFENSE] == 1


def test_multi_hit_effects_accuracy_and_fainting(neutral_type):
    """Verify per-hit effects, late misses, damage patterns, and early KOs."""
    key, state = neutral_type
    # Keep the defender alive through five ordinary hits and let Stamina react.
    table = state.team.stat_table.replace(base_stats=state.team.stat_table.base_stats.at[1, 0, 0].set(1000))
    team = state.team.replace(stat_table=table,
        ability=state.team.ability.at[0, 0, 0].set(A.SKILL_LINK).at[1, 0, 0].set(A.STAMINA),
        item=state.team.item.at[0, 0, 0].set(I.LIFE_ORB))
    team = team.replace(current_hp=team.current_hp.at[1, 0, 0].set(table.stats[1, 0, 0]))
    state = state.replace(team=team)
    run = jax.jit(resolve_multi_hit)
    _, out = run(key, state, 0, make_move('Bullet Seed'))
    assert jnp.sum(hit_events(out)) == 5
    assert out.boosts.normal_boosts[1, S.DEFENSE] == 5
    # Life Orb costs one tenth once per move, not once per hit.
    assert out.active.current_hp[0, 0] == state.active.current_hp[0, 0] - state.active.max_hp[0] // 10

    # The berry activates between hits, before the following damage event.
    berry_team = state.team.replace(
        item=state.team.item.at[1, 0, 0].set(I.SITRUS_BERRY),
        current_hp=state.team.current_hp.at[1, 0, 0].set(state.active.max_hp[1] // 2 + 1))
    _, berry = run(key, state.replace(team=berry_team), 0, make_move('Bullet Seed'))
    item_pos = jnp.nonzero(berry.events.valid & (berry.events.kind == EventKind.ITEM) &
                           (berry.events.detail == I.SITRUS_BERRY), size=1)[0][0]
    heal_pos = jnp.nonzero(berry.events.valid & (berry.events.kind == EventKind.HP) &
                           (berry.events.target == 1) & (berry.events.hp_change > 0), size=1)[0][0]
    damage_pos = jnp.nonzero(hit_events(berry), size=5)[0]
    assert berry.active.item[1, 0] == I.NONE
    assert damage_pos[0] < item_pos < heal_pos < damage_pos[1]

    # Contact damage can faint the attacker and prevent later hits.
    helmet_team = state.team.replace(
        ability=state.team.ability.at[1, 0, 0].set(A.NONE),
        item=state.team.item.at[0, 0, 0].set(I.NONE).at[1, 0, 0].set(I.ROCKY_HELMET),
        current_hp=state.team.current_hp.at[0, 0, 0].set(state.active.max_hp[0] // 6 + 1))
    _, helmet = run(key, state.replace(team=helmet_team), 0, make_move('Tail Slap'))
    assert jnp.sum(hit_events(helmet)) == 2
    assert helmet.active.current_hp[0, 0] == 0

    # Select a deterministic key whose first contact roll triggers Effect
    # Spore's sleep outcome. A sleeping attacker cannot continue these hits.
    def sleeps_on_first_contact(k):
        """Mirror only the documented split boundaries to choose a seed."""
        k, _, _ = jax.random.split(k, 3)  # hit count
        k, _, _ = jax.random.split(k, 3)  # critical hit and damage roll
        _, contact_key, _ = jax.random.split(k, 3)
        roll = jax.random.uniform(contact_key)
        return (roll > .19) & (roll < .30)
    keys = jax.random.split(key, 128)
    candidates = jax.vmap(sleeps_on_first_contact)(keys)
    assert jnp.any(candidates)
    sleep_key = keys[jnp.argmax(candidates)]
    spore_team = state.team.replace(
        ability=state.team.ability.at[1, 0, 0].set(A.EFFECT_SPORE),
        item=state.team.item.at[0, 0, 0].set(I.NONE))
    _, asleep = run(sleep_key, state.replace(team=spore_team), 0, make_move('Tail Slap'))
    from ninjax.enum_types import Status
    assert asleep.active.status[0, 0] == Status.SLEEP
    assert jnp.sum(hit_events(asleep)) == 1

    # Direct resolver assumes the first hit passed its move-level check. With
    # zero accuracy, a per-hit move therefore lands once, then records a miss.
    no_link = state.replace(team=state.team.replace(ability=state.team.ability.at[0, 0, 0].set(A.NONE)))
    axel = make_move('Triple Axel').replace(accuracy=jnp.array([0.]))
    _, missed = run(key, no_link, 0, axel)
    assert jnp.sum(hit_events(missed)) == 1
    assert jnp.sum(missed.events.kind == EventKind.MISS) == 1
    # Skill Link removes later accuracy checks for Triple Axel, not its fixed count.
    _, linked = run(key, state, 0, axel)
    assert jnp.sum(hit_events(linked)) == 3
    assert not jnp.any(linked.events.kind == EventKind.MISS)

    # Battle Armor removes critical-roll variance so the three power levels
    # must produce strictly increasing damage.
    armor = state.replace(team=state.team.replace(ability=state.team.ability.at[1, 0, 0].set(A.BATTLE_ARMOR)))
    _, escalating = run(key, armor, 0, axel)
    positions = jnp.nonzero(hit_events(escalating), size=3)[0]
    losses = -escalating.events.hp_change[positions]
    assert jnp.all(jnp.diff(losses) > 0)
    _, blocked_crit = run(key, armor, 0, make_move('Surging Strikes'))
    assert jnp.sum(hit_events(blocked_crit)) == 3
    assert not jnp.any(blocked_crit.events.kind == EventKind.CRITICAL)

    dice = no_link.replace(team=no_link.team.replace(item=no_link.team.item.at[0, 0, 0].set(I.LOADED_DICE)))
    bomb = make_move('Population Bomb').replace(accuracy=jnp.array([0.]))
    _, bomb_miss = run(key, no_link, 0, bomb)
    assert jnp.sum(hit_events(bomb_miss)) == 1
    assert jnp.sum(bomb_miss.events.kind == EventKind.MISS) == 1
    _, bombed = run(key, dice, 0, bomb)
    assert 4 <= jnp.sum(hit_events(bombed)) <= 10
    assert not jnp.any(bombed.events.kind == EventKind.MISS)

    _, scaled = run(key, no_link, 0, make_move('Scale Shot'))
    assert scaled.boosts.normal_boosts[0, S.DEFENSE] == -1
    assert scaled.boosts.normal_boosts[0, S.SPEED] == 1

    # Focus Sash prevents the first KO and is then consumed before hit two.
    sash = no_link.replace(team=no_link.team.replace(
        item=no_link.team.item.at[1, 0, 0].set(I.FOCUS_SASH)))
    knockout = make_move('Surging Strikes').replace(base_power=jnp.array([10000]))
    _, fainted = run(key, sash, 0, knockout)
    assert jnp.sum(hit_events(fainted)) == 2
    assert fainted.active.current_hp[1, 0] == 0
    assert fainted.active.item[1, 0] == I.NONE
    assert jnp.sum(fainted.events.kind == EventKind.CRITICAL) == 2


def test_first_miss_consumes_one_pp_without_hits(neutral_type):
    """PP is consumed once whether the move misses or hits repeatedly."""
    from ninjax.game_logic import move_used
    key, state = neutral_type
    selected = make_move('Triple Axel').replace(accuracy=jnp.array([0.]))
    # Move is a PyTree, so replace the chosen team slot leaf by leaf.
    moves = jax.tree.map(lambda all_moves, one: all_moves.at[0, 0, 0].set(one), state.team.moves, selected)
    state = state.replace(team=state.team.replace(moves=moves))
    use = jax.jit(move_used)
    _, out = use(key, state, 0, 0)
    assert out.active.moves.current_pp[0, 0, 0] == selected.current_pp[0] - 1
    assert jnp.array_equal(out.active.current_hp, state.active.current_hp)
    assert jnp.sum(hit_events(out)) == 0
    assert jnp.sum(out.events.kind == EventKind.MISS) == 1
    bullet = make_move('Bullet Seed')
    moves = jax.tree.map(lambda all_moves, one: all_moves.at[0, 0, 0].set(one), state.team.moves, bullet)
    _, hit = use(key, state.replace(team=state.team.replace(moves=moves)), 0, 0)
    assert hit.active.moves.current_pp[0, 0, 0] == bullet.current_pp[0] - 1
    assert 2 <= jnp.sum(hit_events(hit)) <= 5


@pytest.mark.jit
def test_multihit_jit_vmap_scan(neutral_type):
    """The hit loop composes with batch and sequential JAX transformations."""
    key, state = neutral_type
    # Small hits leave both sides alive across the scan.
    move = make_move('Bullet Seed').replace(base_power=jnp.array([5]))
    keys = jax.random.split(key, 2)
    states = jax.tree.map(lambda x: jnp.stack((x, x)), state)
    _, batch = jax.jit(jax.vmap(lambda k, s: resolve_multi_hit(k, s, 0, move)))(keys, states)
    assert jnp.all(batch.team.current_hp[:, 1, 0, 0] < states.team.current_hp[:, 1, 0, 0])
    def turn(carry, _):
        """Apply a move within a scanned sequence of decisions."""
        key, state = carry
        key, state = resolve_multi_hit(key, state, 0, move)
        return (key, state), state.active.current_hp[1, 0]
    (_, final), hp = jax.jit(lambda k, s: jax.lax.scan(turn, (k, s), None, length=2))(key, state)
    assert hp[1] < hp[0] < state.active.current_hp[1, 0]
    assert not final.events.overflow
