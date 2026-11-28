"""Player-relative observation PyTrees and their Jumanji specification.

Every leaf starts with a two-player observation axis. Team leaves then have
(self/opponent, six slots) axes. Active/side leaves have a self/opponent axis.
Unknown values are zeroed and have explicit known masks. Opponent HP is rounded
up to public percentage points; private stats and exact HP are never exposed.
"""
from dataclasses import fields

import jax
import jax.numpy as jnp
from flax import struct
from jumanji import specs

from ninjax.records.events import CAPACITY, TransitionEvents
from ninjax.restrictions import active_restriction_mask


class _Block:
    def __getitem__(self, name):
        return getattr(self, name)

@struct.dataclass
class TeamObservation(_Block):
    species: jax.Array
    species_known: jax.Array
    form: jax.Array
    hp_fraction: jax.Array
    hp_known: jax.Array
    current_hp: jax.Array
    stats: jax.Array
    status: jax.Array
    level: jax.Array
    types: jax.Array
    ability: jax.Array
    ability_known: jax.Array
    item: jax.Array
    item_known: jax.Array
    nature: jax.Array
    sheet_known: jax.Array
    moves: jax.Array
    moves_known: jax.Array
    move_pp: jax.Array
    move_max_pp: jax.Array
    move_disabled: jax.Array
    selected: jax.Array
    tera_type: jax.Array
    tera_type_known: jax.Array
    is_terastallized: jax.Array

@struct.dataclass
class ActiveObservation(_Block):
    slot: jax.Array
    boosts: jax.Array
    accuracy_evasion: jax.Array
    added_type: jax.Array
    transformed: jax.Array
    ability_suppressed: jax.Array
    flash_fire: jax.Array
    perish_turns: jax.Array
    disabled_move_id: jax.Array
    disable_turns: jax.Array
    encore_move_id: jax.Array
    encore_turns: jax.Array
    taunt_turns: jax.Array
    throat_chop_turns: jax.Array
    heal_block_turns: jax.Array
    aqua_ring: jax.Array
    ingrain: jax.Array
    leech_seed: jax.Array
    curse: jax.Array
    salt_cure: jax.Array
    syrup_turns: jax.Array
    yawn_turns: jax.Array
    wish_turns: jax.Array
    binding_turns: jax.Array
    confused: jax.Array
    substitute: jax.Array
    charging_move_id: jax.Array
    recharging: jax.Array
    destiny_bond: jax.Array
    roost: jax.Array
    active_move_actions: jax.Array
    previous_move_result: jax.Array

@struct.dataclass
class SideObservation(_Block):
    stealth_rocks: jax.Array
    sticky_webs: jax.Array
    spikes: jax.Array
    toxic_spikes: jax.Array
    reflect: jax.Array
    light_screen: jax.Array
    aurora_veil: jax.Array
    tailwind: jax.Array
    safeguard: jax.Array
    mist: jax.Array

@struct.dataclass
class FieldObservation(_Block):
    turn: jax.Array
    weather: jax.Array
    weather_duration: jax.Array
    terrain: jax.Array
    terrain_duration: jax.Array
    trick_room_duration: jax.Array
    gravity_duration: jax.Array
    magic_room_duration: jax.Array
    wonder_room_duration: jax.Array
    fairy_lock_turns: jax.Array

@struct.dataclass
class DecisionObservation(_Block):
    phase: jax.Array
    acting: jax.Array
    legal_actions: jax.Array
    legal_selection: jax.Array
    tera_actions: jax.Array

@struct.dataclass
class Observation:
    team: TeamObservation
    active: ActiveObservation
    sides: SideObservation
    field: FieldObservation
    decision: DecisionObservation
    transitions: TransitionEvents


def observe(env, state):
    """Build player-relative observations with public active restrictions."""
    b, t = state.battle, state.battle.team
    # Each observer sees self first. Keep private engine state out of this tree.
    def relative(x):
        return jnp.stack((x, x[::-1]), axis=1)
    own = jnp.broadcast_to(jnp.array([True, False])[None, :, None], (2, 2, 6))
    visible = relative(state.revealed)
    species_known = own | visible | (env.params.information_mode != 0)
    sheet = own | jnp.full((2, 2, 6), env.params.information_mode == 2)
    public = own | visible
    def masked(x, known):
        x = relative(x)
        while known.ndim < x.ndim:
            known = known[..., None]
        return jnp.where(known, x, jnp.zeros_like(x))
    fraction = relative(t.hp_percent)
    hp = jnp.where(own, fraction, jnp.ceil(fraction * 100) / 100)
    moves_known = relative(state.revealed_moves) | sheet[..., None]
    tera_known = sheet | relative(t.is_terastallized[..., 0])
    item_known = sheet | relative(b.known_items >= 0)
    # Frisk reveals only the item; it must not reveal ability, nature or stats.
    items = jnp.where(sheet, relative(t.item[..., 0]), jnp.maximum(relative(b.known_items), 0))
    ability_known = sheet | relative(b.known_abilities >= 0)
    # The owner's active ability is always visible, even while suppressed.
    current = t.ability[..., 0].at[jnp.arange(2), b.active_index[:, 0]].set(
        jnp.where(b.volatile_status.ability_state.ability_override[:, 0] >= 0,
                  b.volatile_status.ability_state.ability_override[:, 0], b.active_field('ability')[:, 0]))
    abilities = jnp.where(own, relative(current), jnp.where(relative(b.known_abilities >= 0),
        jnp.maximum(relative(b.known_abilities), 0), jnp.where(sheet, relative(t.ability[..., 0]), 0)))

    disabled = t.moves.disabled[..., 0]
    disabled = disabled.at[jnp.arange(2), b.active_index[:, 0]].set(
        disabled[jnp.arange(2), b.active_index[:, 0]] | ~active_restriction_mask(b))
    slots = (jnp.arange(2), b.active_index[:, 0])
    species = t.species.at[slots].set(b.active_identity('species'))
    forms = t.form.at[slots].set(b.active_identity('form'))
    types = t.type_list.at[slots].set(b.effective_types())
    stats = t.stat_table.stats.at[slots].set(b.active_stats())
    copied = b.volatile_status.properties.copied_slots
    hidden_copies = jnp.zeros_like(state.revealed_moves).at[slots].set(copied)
    # Earlier move reveals for a party slot refer to its original moves.
    moves_known &= ~relative(hidden_copies) | own[..., None]
    team = dict(species=masked(species[..., 0], species_known), species_known=species_known,
        form=masked(forms[..., 0], species_known),
        hp_fraction=jnp.where(public, hp, 0), hp_known=public,
        current_hp=masked(t.current_hp[..., 0], own), stats=masked(stats, own),
        status=masked(t.status[..., 0], public), level=masked(t.stat_table.level[..., 0], public), types=masked(types, public),
        ability=abilities, ability_known=ability_known, item=items, item_known=item_known,
        nature=masked(jnp.concatenate((t.stat_table.nature.increased, t.stat_table.nature.decreased), -1), sheet),
        sheet_known=sheet, moves=masked(t.moves.identifier[..., 0], moves_known), moves_known=moves_known,
        move_pp=masked(t.moves.current_pp[..., 0], own), move_max_pp=masked(t.moves.max_pp[..., 0], own), move_disabled=masked(disabled, own),
        selected=masked(state.selected, own), tera_type=masked(t.tera_type[..., 0], tera_known),
        tera_type_known=tera_known,
        is_terastallized=masked(t.is_terastallized[..., 0], public))
    r = b.volatile_status.residuals
    active = dict(slot=relative(b.active_index[:, 0]), boosts=relative(b.boosts.normal_boosts),
                  accuracy_evasion=relative(b.boosts.acc_boosts),
                  added_type=relative(b.volatile_status.properties.added_type[:, 0]),
                  transformed=relative(b.volatile_status.transformed[:, 0]),
                  ability_suppressed=relative(((b.ability == 0) & (b.ability_identity != 0))[:, 0]),
                  flash_fire=relative(b.volatile_status.ability_state.flash_fire[:, 0]), perish_turns=relative(b.perish_turns[:, 0]),
                  disabled_move_id=relative(jnp.where(b.volatile_status.restrictions.disable_turns[:, 0] > 0, b.volatile_status.restrictions.disabled_move_id[:, 0], 0)),
                  disable_turns=relative(b.volatile_status.restrictions.disable_turns[:, 0]),
                  encore_move_id=relative(jnp.where(b.volatile_status.restrictions.encore_turns[:, 0] > 0, b.volatile_status.restrictions.encore_move_id[:, 0], 0)),
                  encore_turns=relative(b.volatile_status.restrictions.encore_turns[:, 0]),
                  taunt_turns=relative(b.volatile_status.restrictions.taunt_turns[:, 0]),
                  throat_chop_turns=relative(b.volatile_status.restrictions.throat_chop_turns[:, 0]),
                  heal_block_turns=relative(b.volatile_status.restrictions.heal_block_turns[:, 0]),
                  aqua_ring=relative(r.aqua_ring[:, 0] != 0), ingrain=relative(r.ingrain[:, 0] != 0),
                  leech_seed=relative(r.leech_seed[:, 0] != 0), curse=relative(r.curse[:, 0] != 0),
                  salt_cure=relative(r.salt_cure[:, 0] != 0), syrup_turns=relative(r.syrup_turns[:, 0]),
                  yawn_turns=relative(r.yawn_turns[:, 0]), wish_turns=relative(b.slot_residuals.wish_turns[:, 0]),
                  binding_turns=relative(b.volatile_status.trapping.binding_turns[:, 0]),
                  confused=relative(b.volatile_status.confusion_turns[:, 0] > 0),
                  substitute=relative(b.volatile_status.substitute_hp[:, 0] > 0),
                  charging_move_id=relative(b.volatile_status.continuation.charging_move_id[:, 0]),
                  recharging=relative(b.volatile_status.continuation.recharging[:, 0]),
                  destiny_bond=relative(b.volatile_status.destiny_bond[:, 0]),
                  roost=relative(b.volatile_status.roost[:, 0]),
                  active_move_actions=relative(b.volatile_status.move_history.active_move_actions[:, 0]),
                  previous_move_result=relative(b.volatile_status.move_history.previous_move_result[:, 0]))
    hazards, conditions = b.hazards, b.side_conditions
    sides = dict(stealth_rocks=relative(hazards.stealth_rocks[:, 0]), sticky_webs=relative(hazards.sticky_webs[:, 0]),
                 spikes=relative(hazards.spikes[:, 0]), toxic_spikes=relative(hazards.toxic_spikes[:, 0]),
                 reflect=relative(conditions.reflect[:, 0]), light_screen=relative(conditions.light_screen[:, 0]),
                 aurora_veil=relative(conditions.aurora_veil[:, 0]), tailwind=relative(conditions.tailwind[:, 0]),
                 safeguard=relative(conditions.safeguard[:, 0]), mist=relative(conditions.mist[:, 0]))
    field = {name: jnp.broadcast_to(jnp.asarray(value, jnp.int32), (2,)) for name, value in dict(
        turn=state.turn, weather=b.weather.weather, weather_duration=b.weather.duration,
        terrain=b.terrain.terrain, terrain_duration=b.terrain.duration,
        trick_room_duration=b.trick_room_duration, gravity_duration=b.gravity_duration,
        magic_room_duration=b.magic_room_duration, wonder_room_duration=b.wonder_room_duration,
        fairy_lock_turns=b.fairy_lock_turns).items()}
    decision = dict(phase=jnp.broadcast_to(state.phase, (2,)), acting=state.acting_player_mask,
                    legal_actions=env.legal_action_mask(state),
                    legal_selection=jnp.full((2, 6), state.phase == 3),
                    tera_actions=env.tera_action_mask(state))
    def event_view(player):
        e = b.events.unpack()
        target_own = e.target == player
        hp_event = (e.kind == 1)
        return e.replace(source=jnp.where(e.source < 0, -1, (e.source != player).astype(jnp.int32)),
            target=jnp.where(e.target < 0, -1, (e.target != player).astype(jnp.int32)),
            before=jnp.where(hp_event & ~target_own, 0, e.before),
            after=jnp.where(hp_event & ~target_own, 0, e.after),
            hp_change=jnp.where(target_own, e.hp_change, 0))
    transitions = jax.vmap(event_view)(jnp.arange(2))
    return Observation(TeamObservation(**team), ActiveObservation(**active), SideObservation(**sides),
                       FieldObservation(**field), DecisionObservation(**decision), transitions)


def observation_spec():
    """Static schema, available even before callers supply teams."""
    def block(cls, shapes, floats=(), booleans=()):
        return specs.Spec(cls, cls.__name__, **{
            name: specs.Array(shape, jnp.dtype('bool' if name in booleans else
                              'float32' if name in floats else 'int32'), name)
            for name, shape in shapes.items()})
    team = {name: (2, 2, 6) for name in (
        'species', 'species_known', 'form', 'hp_fraction', 'hp_known', 'current_hp', 'status',
        'level', 'ability', 'ability_known', 'item', 'item_known', 'sheet_known', 'selected',
        'tera_type', 'tera_type_known', 'is_terastallized')}
    team.update(stats=(2, 2, 6, 6), types=(2, 2, 6, 2), nature=(2, 2, 6, 2))
    team.update({name: (2, 2, 6, 4) for name in ('moves', 'moves_known', 'move_pp', 'move_max_pp', 'move_disabled')})
    active = {entry.name: (2, 2) for entry in fields(ActiveObservation)}
    active.update(boosts=(2, 2, 6), accuracy_evasion=(2, 2, 2))
    sides = {f.name: (2, 2) for f in fields(SideObservation)}
    field = {f.name: (2,) for f in fields(FieldObservation)}

    events = {f.name: (2, CAPACITY) for f in fields(TransitionEvents)}
    events.update(count=(2,), overflow=(2,))
    return specs.Spec(Observation, 'observation',
        team=block(TeamObservation, team, floats=('hp_fraction',),
                   booleans=('species_known', 'hp_known', 'ability_known', 'item_known', 'sheet_known', 'moves_known', 'move_disabled', 'selected',
                             'tera_type_known', 'is_terastallized')),
        active=block(ActiveObservation, active, booleans=('transformed', 'ability_suppressed',
            'flash_fire', 'confused', 'substitute', 'recharging', 'destiny_bond', 'roost',
            'aqua_ring', 'ingrain', 'leech_seed', 'curse', 'salt_cure')),
        sides=block(SideObservation, sides, booleans=('stealth_rocks', 'sticky_webs')),
        field=block(FieldObservation, field),
        decision=block(DecisionObservation, dict(phase=(2,), acting=(2,), legal_actions=(2, 15),
                                                 legal_selection=(2, 6), tera_actions=(2, 15)),
                       booleans=('acting', 'legal_actions', 'legal_selection', 'tera_actions')),
        transitions=block(TransitionEvents, events, floats=('hp_fraction_change',), booleans=('valid', 'overflow')))
