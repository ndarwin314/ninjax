"""Form changes triggered by attacks, weather, HP and switching."""

import jax
import jax.numpy as jnp

from ninjax.enums.abilities import AbilityEnum as A
from ninjax.enums.field import WeatherEnum
from ninjax.enums.items import ItemEnum as I
from ninjax.enums.moves import MoveType
from ninjax.enums.pokemon import FormEnum as F, StatEnum, Status, Type
from ninjax.records.events import EventKind, emit
from ninjax.side.hp_effects import take_damage_percent
from ninjax.side.queries import effective_weather
from ninjax.side.stat_effects import conditional_reduce_boosts
from ninjax.side.status_effects import set_status
from ninjax.side.updates import update_active_fields


# HP is identical within each pair. The values use Pokédex stat order.
EISCUE = (75, 80, 110, 65, 90, 50)
EISCUE_NOICE = (75, 80, 70, 65, 50, 130)
MINIOR_CORE = (60, 100, 60, 100, 60, 120)
MINIOR_METEOR = (60, 60, 100, 60, 100, 60)
PALAFIN_HERO = (100, 160, 97, 106, 87, 100)
MELOETTA_ARIA = (100, 77, 77, 128, 128, 90)
MELOETTA_PIROUETTE = (100, 128, 90, 77, 77, 128)


def _keep_state(state, *unused):
    """Accept unapplied form operands."""
    return state


def _set_party_form(state, player, form, base_stats, weight_hg):
    """Persist a form with the holder's own nature, levels and training."""
    slot = state.active_index[player, 0]
    team = state.team
    stats = team.stat_table
    previous = team.form[player, slot, 0]
    state = update_active_fields(state, player, form=jnp.array([form]), weight_hg=jnp.array([weight_hg]))
    state = state.replace(team=state.team.replace(stat_table=stats.replace(
        base_stats=stats.base_stats.at[player, slot].set(base_stats))))
    return emit(state, EventKind.FORM, player, detail=team.species[player, slot, 0], before=previous, after=form)


@jax.jit
def reactive_form_update(state, player, snow_trigger):
    """Update Shields Down and restore a broken Ice Face on its snow trigger."""
    ability = state.ability[player, 0]
    species = state.active_identity('species', player)[0]
    form = state.active_identity('form', player)[0]
    hp = state.active_field('current_hp', player)[0]
    max_hp = state.active_stats(player)[0]
    weather = effective_weather(state.weather.weather, state.ability, state.active_field('current_hp'))
    meteor = 2 * hp > max_hp
    minior = (species == 774) & (ability == A.SHIELDS_DOWN) & ~state.volatile_status.transformed[player, 0]
    restore = (species == 875) & (ability == A.ICE_FACE) & (form == F.EISCUE_NOICE) & (
        snow_trigger & (weather == WeatherEnum.SNOW))
    change = (hp > 0) & ((minior & (meteor != (form == F.MINIOR_METEOR))) | restore)
    new_form = jnp.where(minior & meteor, F.MINIOR_METEOR, F.DEFAULT)
    stats = jnp.where(minior, jnp.where(meteor, jnp.array(MINIOR_METEOR), jnp.array(MINIOR_CORE)), jnp.array(EISCUE))
    weight = jnp.where(minior, jnp.where(meteor, 400, 3), 890)
    return jax.lax.cond(change, _set_party_form, _keep_state, state, player, new_form, stats, weight)


def palafin_departure(state, player):
    """Keep Palafin's Hero appearance and stats after its first live departure."""
    ready = (state.ability[player, 0] == A.ZERO_TO_HERO) & (
        state.active_field('species', player)[0] == 964) & (
        state.active_field('form', player)[0] != F.PALAFIN_HERO) & (state.active_field('current_hp', player)[0] > 0)
    return jax.lax.cond(ready, _set_party_form, _keep_state, state, player,
                        F.PALAFIN_HERO, jnp.array(PALAFIN_HERO), 974)


def _bust_disguise(state, player):
    """Replace Mimikyu's disguise and pay its indirect HP penalty."""
    previous = state.active_field('form', player)[0]
    state = update_active_fields(state, player, form=jnp.array([F.MIMIKYU_BUSTED]))
    state = emit(state, EventKind.FORM, player, detail=778, before=previous, after=F.MIMIKYU_BUSTED)
    return take_damage_percent(state, player, 1 / 8)


def _bust_ice_face(state, player):
    """Switch Eiscue to Noice Face and update the stored raw stats."""
    return _set_party_form(state, player, F.EISCUE_NOICE, jnp.array(EISCUE_NOICE), 890)


def shield_form(state, player, ability, category, direct):
    """Recognize a direct move hit absorbed by Disguise or Ice Face."""
    species = state.active_identity('species', player)[0]
    form = state.active_identity('form', player)[0]
    disguise = direct & (species == 778) & (ability == A.DISGUISE) & (form != F.MIMIKYU_BUSTED)
    ice = direct & (species == 875) & (ability == A.ICE_FACE) & (form != F.EISCUE_NOICE) & (category == MoveType.PHYSICAL)
    return disguise, ice


@jax.jit
def break_form_shield(state, player, disguise, ice):
    """Commit a damage shield break after the attack's zero-HP hit record."""
    branch = jnp.select((disguise, ice), (1, 2), default=0)
    return jax.lax.switch(branch, (_keep_state, _bust_disguise, _bust_ice_face), state, player)


def _change_meloetta(state, player):
    """Change Meloetta's switch-cleared typing and non-HP stats."""
    v = state.volatile_status
    p = v.properties
    dance = state.active_identity('form', player)[0] != F.MELOETTA_PIROUETTE
    form = jnp.where(dance, F.MELOETTA_PIROUETTE, F.DEFAULT)
    types = jnp.array([Type.NORMAL, jnp.where(dance, Type.FIGHTING, Type.PSYCHIC)])
    base = jnp.where(dance, jnp.array(MELOETTA_PIROUETTE), jnp.array(MELOETTA_ARIA))
    slot = state.active_index[player, 0]
    raw = state.team.stat_table[player, slot].replace(base_stats=base).stats
    stats = p.stats.at[player, 1:].set(raw[1:])
    state = update_active_fields(state, player, type_list=types)
    p = p.replace(stats=stats, form=p.form.at[player, 0].set(form),
                  added_type=p.added_type.at[player, 0].set(Type.NONE))
    state = state.replace(volatile_status=v.replace(properties=p))
    return emit(state, EventKind.FORM, player, detail=648, after=form)


def relic_song_form(state, player, success):
    """Toggle the user's forme after Meloetta lands Relic Song."""
    eligible = success & (state.active_identity('species', player)[0] == 648) & (
        ~state.volatile_status.transformed[player, 0]) & (state.active_field('current_hp', player)[0] > 0)
    return jax.lax.cond(eligible, _change_meloetta, _keep_state, state, player)


def _set_gulp_form(state, player, form):
    """Update Cramorant's switch-cleared carried prey without changing stats."""
    v = state.volatile_status
    p = v.properties
    state = state.replace(volatile_status=v.replace(properties=p.replace(
        form=p.form.at[player, 0].set(form))))
    return emit(state, EventKind.FORM, player, detail=845, after=form)


def catch_gulp_missile(state, player, trigger):
    """Catch prey during Dive preparation or before Surf's primary hit."""
    hp = state.active_field('current_hp', player)[0]
    enabled = trigger & (state.ability[player, 0] == A.GULP_MISSILE) & (
        state.active_identity('species', player)[0] == 845) & (
        state.active_identity('form', player)[0] == F.DEFAULT)
    form = jnp.where(2 * hp <= state.active_stats(player)[0], F.CRAMORANT_GORGING, F.CRAMORANT_GULPING)
    return jax.lax.cond(enabled, _set_gulp_form, _keep_state, state, player, form)


def _skip_status(key, state, player, status):
    """Skip the paralysis callback while accepting its operands."""
    return key, state


def _launch_prey(key, state, player, form):
    """Damage the attacker and apply the caught prey's additional effect."""
    attacker = 1-player
    indirect = state.ability[attacker, 0] != A.MAGIC_GUARD
    state = take_damage_percent(state, attacker, jnp.where(indirect, .25, 0.))
    living = state.active_field('current_hp', attacker)[0] > 0
    gulping = form == F.CRAMORANT_GULPING
    state = conditional_reduce_boosts(state, attacker, living & gulping, StatEnum.DEFENSE, 1, True)
    key, state = jax.lax.cond(living & ~gulping, set_status, _skip_status,
                             key, state, attacker, jnp.array([Status.PARALYZE]))
    return key, _set_gulp_form(state, player, F.DEFAULT)


def _keep_prey(key, state, player, form):
    """Preserve carried prey on ineffective damage events."""
    return key, state


@jax.jit
def gulp_missile_hit(key, state, player, hit):
    """Respond to a direct hit against a Cramorant that is carrying prey."""
    form = state.active_identity('form', player)[0]
    enabled = hit & (state.ability[player, 0] == A.GULP_MISSILE) & (
        (form == F.CRAMORANT_GULPING) | (form == F.CRAMORANT_GORGING)) & (
        state.active_field('current_hp', 1-player)[0] > 0) & (
        state.volatile_status.continuation.semi_invulnerability[player, 0] == 0)
    return jax.lax.cond(enabled, _launch_prey, _keep_prey, key, state, player, form)


def _enter_terastal_form(state, player):
    """Apply Tera Shift's form, HP increase and persistent new base ability."""
    before = state.active_stats(player)[0]
    state = _set_party_form(state, player, F.TERAPAGOS_TERASTAL,
                            jnp.array([95, 95, 110, 105, 110, 85]), 160)
    after = state.active_stats(player)[0]
    hp = state.active_field('current_hp', player)
    state = update_active_fields(state, player, ability=jnp.array([A.TERA_SHELL]),
                                 current_hp=jnp.minimum(after, hp + after - before))
    slot = state.active_index[player, 0]
    state = state.replace(known_abilities=state.known_abilities.at[player, slot].set(A.TERA_SHELL))
    return emit(state, EventKind.ABILITY, player, detail=A.TERA_SHIFT, after=A.TERA_SHELL)


@jax.jit
def tera_shift(state, player):
    """Activate a living Terapagos's automatic entry form."""
    enabled = (state.ability[player, 0] == A.TERA_SHIFT) & (
        state.active_field('species', player)[0] == 1024) & (
        state.active_field('current_hp', player)[0] > 0)
    return jax.lax.cond(enabled, _enter_terastal_form, _keep_state, state, player)


def _set_origin_form(state, player, form, base, weight):
    """Install the item-bound Origin forme, including Giratina's ability."""
    state = _set_party_form(state, player, form, base, weight)
    ability = state.active_field('ability', player)
    return update_active_fields(state, player,
        ability=jnp.where(form == F.GIRATINA_ORIGIN, A.LEVITATE, ability))


@jax.jit
def held_origin_form(state, player):
    """Project the three Gen 9 Origin items when their species enters battle."""
    species = state.active_field('species', player)[0]
    item = state.active_field('item', player)[0]
    flags = ((species == 487) & (item == I.GRISEOUS_CORE),
             (species == 483) & (item == I.ADAMANT_CRYSTAL),
             (species == 484) & (item == I.LUSTROUS_GLOBE))
    idx = jnp.select(flags, (0, 1, 2), default=0)
    form = jnp.array([F.GIRATINA_ORIGIN, F.DIALGA_ORIGIN, F.PALKIA_ORIGIN])[idx]
    stats = jnp.array([[150, 120, 100, 120, 100, 90],
                       [100, 100, 120, 150, 120, 90],
                       [90, 100, 100, 150, 120, 120]])[idx]
    weight = jnp.array([6500, 8500, 6600])[idx]
    enabled = (flags[0] | flags[1] | flags[2]) & (state.active_field('form', player)[0] != form)
    return jax.lax.cond(enabled, _set_origin_form, _keep_state, state, player, form, stats, weight)


def _crown_held_form(state, player, sword):
    """Apply the Crowned form and replace Iron Head with its battle move."""
    form = jnp.where(sword, F.ZACIAN_CROWNED, F.ZAMAZENTA_CROWNED)
    base = jnp.where(sword, jnp.array([92, 150, 115, 80, 115, 148]), jnp.array([92, 120, 140, 80, 140, 128]))
    weight = jnp.where(sword, 3550, 7850)
    state = _set_party_form(state, player, form, base, weight)
    types = jnp.array([jnp.where(sword, Type.FAIRY, Type.FIGHTING), Type.STEEL])
    state = update_active_fields(state, player, type_list=types, base_types=types)
    slot = state.active_index[player, 0]
    table = state.team.moves
    move = state.active_moves(player)
    replace = move.identifier == 442
    identifier = jnp.where(sword, 781, 782)
    secondary = table.secondaries
    probabilities = jnp.where(replace, 0., move.secondaries.probabilities)
    table = table.replace(
        identifier=table.identifier.at[player, slot].set(jnp.where(replace, identifier, move.identifier)),
        base_power=table.base_power.at[player, slot].set(jnp.where(replace, 100, move.base_power)),
        current_pp=table.current_pp.at[player, slot].set(jnp.where(replace, 8, move.current_pp)),
        max_pp=table.max_pp.at[player, slot].set(jnp.where(replace, 8, move.max_pp)),
        native_pp=table.native_pp.at[player, slot].set(jnp.where(replace, 5, move.native_pp)),
        secondaries=secondary.replace(probabilities=secondary.probabilities.at[player, slot].set(probabilities)))
    return state.replace(team=state.team.replace(moves=table))


@jax.jit
def held_crowned_form(state, player):
    """Apply Rusted Sword and Shield's species-bound transformations."""
    species = state.active_field('species', player)[0]
    item = state.active_field('item', player)[0]
    form = state.active_field('form', player)[0]
    sword = (species == 888) & (item == I.RUSTED_SWORD) & (form != F.ZACIAN_CROWNED)
    shield = (species == 889) & (item == I.RUSTED_SHIELD) & (form != F.ZAMAZENTA_CROWNED)
    return jax.lax.cond(sword | shield, _crown_held_form, _keep_state, state, player, sword)
