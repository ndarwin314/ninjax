"""Shared callbacks for the enabled Scarlet/Violet curated move subset.

All predicates are JAX arrays. Helpers return immutable battle state and use
Showdown move numbers so no string dispatch runs inside compiled transitions.
"""
import jax
import jax.numpy as jnp

from ninjax.enum_types import AbilityEnum as A, ItemEnum as I, WeatherEnum as W, TerrainEnum as T, Terrain, Type, Status, FormEnum
from ninjax.side import add_boosts, reduce_boosts, clear_boosts, set_status, update_active, take_damage_value
from ninjax.held_effects import effective_weather


def direct_status_move(key, state, player, move):
    """Resolve Rest and Pain Split without running ordinary damage callbacks."""
    def rest(args):
        """Replace an existing major status with two full turns of Rest sleep."""
        key, s = args
        mon = s.active[player]
        eligible = (mon.current_hp[0] < mon.max_hp) & (mon.status[0] != Status.SLEEP) & (mon.current_hp[0] > 0)
        # Reuse terrain and ability sleep checks, but allow Rest to replace a
        # burn/poison/paralysis. Failed sleep attempts restore the original state.
        candidate = update_active(s, player, mon.replace(status=jnp.zeros_like(mon.status)))
        key, candidate = set_status(key, candidate, player, jnp.array([Status.SLEEP]))
        success = eligible & (candidate.active.status[player, 0] == Status.SLEEP)
        def heal(c):
            """Use the existing sleep counter's decrement-before-action convention."""
            from ninjax.events import EventKind, emit
            c = update_active(c, player, c.active[player].replace(sleep_counter=jnp.array([3])))
            c = c.replace(toxic_counter=c.toxic_counter.at[player].set(0))
            c = emit(c, EventKind.STATUS, player, before=mon.status[0], after=Status.SLEEP)
            return take_damage_value(c, player, -mon.max_hp, False)
        return key, jax.lax.cond(success, heal, lambda _: s, candidate)
    def split(args):
        """Set both HP values to their original mean before reactive healing."""
        from ninjax.events import EventKind, emit
        from ninjax.held_effects import after_hp_change
        key, s = args
        average = jnp.maximum(1, jnp.sum(s.active.current_hp) // 2)
        def assign(p, s):
            """Assign HP directly: Pain Split is neither damage nor healing."""
            mon = s.active[p]
            hp = jnp.minimum(average, mon.max_hp).reshape((1,))
            s = emit(s, EventKind.HP, p, before=mon.current_hp, after=hp,
                hp_change=hp-mon.current_hp,
                hp_fraction_change=(jnp.ceil(hp/mon.max_hp*100)-jnp.ceil(mon.current_hp/mon.max_hp*100))/100,
                enabled=hp[0] != mon.current_hp[0])
            return update_active(s, p, mon.replace(current_hp=hp))
        s = jax.lax.fori_loop(0, 2, assign, s)
        return key, jax.lax.fori_loop(0, 2, lambda p, s: after_hp_change(s, p), s)
    return jax.lax.cond(move.identifier[0] == 156, rest, split, (key, state))


def stages(state, player, stat, amount, opposing=False):
    """Apply one stat change, invoking opposing-drop abilities only on drops."""
    def ordinary(s):
        """Apply an Attack through Speed change using the existing hooks."""
        return jax.lax.cond((amount < 0) & opposing,
            lambda s: reduce_boosts(s, player, stat, -amount),
            lambda s: add_boosts(s, player, stat, amount), s)
    def accuracy(s):
        """Apply accuracy/evasion stages in their separate two-column array."""
        from ninjax.side import set_boosts
        from ninjax.stats import StatBoosts
        delta = amount * jnp.where(s.ability[player, 0] == A.CONTRARY, -1, 1)
        delta = jnp.where(opposing & (amount < 0) & (s.ability[player, 0] == A.CLEAR_BODY), 0, delta)
        return set_boosts(s, player, StatBoosts(normal_boosts=s.boosts.normal_boosts[player],
            acc_boosts=s.boosts.acc_boosts[player].at[stat-6].add(delta)))
    return jax.lax.cond((amount != 0) & (stat >= 1) & (stat <= 7),
        lambda s: jax.lax.cond(stat <= 5, ordinary, accuracy, s), lambda s: s, state)


def stat_changes(state, player, indices, amounts, opposing=False):
    """Apply a padded stat-change vector sequentially in a fixed JAX scan."""
    def apply(s, change):
        """Consume one stat index and its matching stage increment."""
        return stages(s, player, change[0], change[1], opposing), None
    return jax.lax.scan(apply, state, (indices, amounts))[0]


def move_overrides(state, player, move):
    """Resolve represented move power, typing and grounded terrain priority."""
    mon = state.effective_active[player]
    ident = move.identifier
    weather = effective_weather(state)
    grounded = state.is_grounded(player)
    power = jnp.where((ident == 512) & (mon.item == I.NONE), 110, move.base_power)
    positive = jnp.sum(jnp.maximum(state.boosts.normal_boosts[player, 1:], 0)) + jnp.sum(jnp.maximum(state.boosts.acc_boosts[player], 0))
    power = jnp.where(ident == 500, 20 + 20 * positive, power)
    target = state.effective_active[1-player]
    def weight(mon):
        """Apply represented weight abilities in integer hectograms."""
        return jnp.maximum(1, mon.weight_hg * jnp.where(mon.ability == A.HEAVY_METAL, 2, 1) // jnp.where(mon.ability == A.LIGHT_METAL, 2, 1))
    target_weight = weight(target)
    low_kick = 20 + 20 * sum((target_weight >= threshold).astype(jnp.int32) for threshold in (100, 250, 500, 1000, 2000))
    heavy_slam = jnp.where(weight(mon) >= 5*target_weight, 120,
                 jnp.where(weight(mon) >= 4*target_weight, 100,
                 jnp.where(weight(mon) >= 3*target_weight, 80,
                 jnp.where(weight(mon) >= 2*target_weight, 60, 40))))
    power = jnp.where(ident == 67, low_kick, power)
    power = jnp.where(ident == 484, heavy_slam, power)
    weather_ball = (ident == 311) & (weather != W.NONE)
    move_type = jnp.where(weather_ball & (weather == W.SUN), Type.FIRE, move.type)
    move_type = jnp.where(weather_ball & (weather == W.RAIN), Type.WATER, move_type)
    move_type = jnp.where(weather_ball & (weather == W.SANDSTORM), Type.ROCK, move_type)
    move_type = jnp.where(weather_ball & (weather == W.SNOW), Type.ICE, move_type)
    # Ivy Cudgel follows the user's current form, including when copied.
    ivy_type = jnp.where(mon.form == FormEnum.OGERPON_WELLSPRING, Type.WATER,
               jnp.where(mon.form == FormEnum.OGERPON_HEARTHFLAME, Type.FIRE,
               jnp.where(mon.form == FormEnum.OGERPON_CORNERSTONE, Type.ROCK, move_type)))
    move_type = jnp.where(ident == 904, ivy_type, move_type)
    # Revelation Dance uses the first current type; ordinary Tera overrides it.
    first_type = jnp.where(mon.type_list[0] != Type.NONE, mon.type_list[0],
                           jnp.where(mon.type_list[1] != Type.NONE, mon.type_list[1], Type.NORMAL))
    dance_type = jnp.where(mon.is_terastallized[0] & (mon.tera_type[0] != Type.NONE), mon.tera_type[0], first_type)
    move_type = jnp.where(ident == 686, dance_type, move_type)
    power = jnp.where(weather_ball, 100, power)
    priority = jnp.where((ident == 803) & grounded & (state.terrain.terrain == T.GRASSY), 1, move.priority)
    return move.replace(base_power=power, type=move_type, priority=priority)


def secondary_effects(key, state, player, move):
    """Roll one supported secondary effect, respecting Shield Dust and survival."""
    key, roll_key = jax.random.split(key)
    target = 1-player
    ability = state.ability[target, 0]
    chance = move.secondary_chance[0] * jnp.where(state.ability[player, 0] == A.SERENE_GRACE, 2., 1.)
    triggered = (jax.random.uniform(roll_key) < chance) & (ability != A.SHIELD_DUST) & (state.active.current_hp[target, 0] > 0)
    def apply(args):
        key, state = args
        key, state = jax.lax.cond(move.secondary_status[0] != 0,
            lambda args: set_status(args[0], args[1], target, move.secondary_status), lambda args: args, (key, state))
        from ninjax.side import apply_confusion
        key, state = jax.lax.cond(move.secondary_confusion[0],
            lambda args: apply_confusion(args[0], args[1], target), lambda args: args, (key, state))
        state = stat_changes(state, target, move.secondary_stat, move.secondary_stages, True)
        v = state.volatile_status
        flinch = move.secondary_flinch[0] & (ability != A.INNER_FOCUS)
        return key, state.replace(volatile_status=v.replace(flinched=v.flinched.at[target, 0].set(v.flinched[target, 0] | flinch)))
    return jax.lax.cond(triggered, apply, lambda args: args, (key, state))


def after_damage(state, player, move):
    """Apply guaranteed hit callbacks for terrain, hazards, and stage clearing."""
    ident = move.identifier[0]
    target = 1-player
    state = jax.lax.cond(ident == 499, lambda s: clear_boosts(s, target), lambda s: s, state)
    state = state.replace(
        stealth_rocks=state.stealth_rocks.at[target].set(state.stealth_rocks[target] | (ident == 830)),
        spikes=state.spikes.at[target].set(jnp.minimum(3, state.spikes[target] + (ident == 845))))
    state = jax.lax.cond(ident == 861,
        lambda s: s.replace(terrain=Terrain(T.NONE, 0)), lambda s: s, state)
    def spin(s):
        """Remove represented entry hazards and raise the surviving user's Speed."""
        s = s.replace(**{name: getattr(s, name).at[player].set(jnp.zeros_like(getattr(s, name)[player]))
                         for name in ('stealth_rocks', 'spikes', 'toxic_spikes', 'sticky_webs')})
        return jax.lax.cond(s.active.current_hp[player, 0] > 0,
            lambda s: add_boosts(s, player, 5, 1), lambda s: s, s)
    return jax.lax.cond(ident == 229, spin, lambda s: s, state)


def field_status(state, player, move):
    """Set screens and timers, swap side conditions, or clear stages field-wide."""
    ident = move.identifier[0]
    duration = jnp.where(state.active.item[player, 0] == I.LIGHT_CLAY, 8, 5)
    for name, enabled in [('light_screen', ident == 113),
                          ('aurora_veil', (ident == 694) & (effective_weather(state) == W.SNOW)),
                          ('tailwind', ident == 366)]:
        current = getattr(state, name)
        value = jnp.where(enabled & (current[player, 0] == 0), jnp.where(ident == 366, 4, duration), current[player, 0])
        state = state.replace(**{name: current.at[player, 0].set(value)})
    state = state.replace(trick_room_duration=jnp.where(ident == 433, jnp.where(state.trick_room_duration > 0, 0, 5), state.trick_room_duration))
    state = jax.lax.cond(ident == 114, lambda s: clear_boosts(clear_boosts(s, 0), 1), lambda s: s, state)
    def defog(s):
        """Lower target evasion, clear both sides' hazards and target screens."""
        from ninjax.side import set_boosts
        from ninjax.stats import StatBoosts
        target = 1-player
        amount = jnp.where(s.ability[target, 0] == A.CONTRARY, 1, -1)
        amount = jnp.where(s.ability[target, 0] == A.CLEAR_BODY, 0, amount)
        boosts = StatBoosts(normal_boosts=s.boosts.normal_boosts[target],
            acc_boosts=s.boosts.acc_boosts[target].at[1].add(amount))
        s = set_boosts(s, target, boosts)
        s = s.replace(**{name: jnp.zeros_like(getattr(s, name)) for name in ('stealth_rocks', 'spikes', 'toxic_spikes', 'sticky_webs')})
        s = s.replace(**{name: getattr(s, name).at[target].set(0) for name in ('reflect', 'light_screen', 'aurora_veil')})
        return s.replace(terrain=Terrain(T.NONE, 0))
    state = jax.lax.cond(ident == 432, defog, lambda s: s, state)
    return jax.lax.cond(ident == 756, lambda s: s.replace(**{
        name: getattr(s, name)[::-1] for name in ('stealth_rocks', 'spikes', 'toxic_spikes', 'sticky_webs',
                                                'reflect', 'light_screen', 'aurora_veil', 'tailwind')}), lambda s: s, state)
