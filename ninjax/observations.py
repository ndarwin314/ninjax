"""Player-relative observation PyTrees and their Jumanji specification.

Every leaf starts with a two-player observation axis. Team leaves then have
(self/opponent, six slots) axes. Active/side leaves have a self/opponent axis.
Unknown values are zeroed and have explicit known masks. Opponent HP is rounded
up to public percentage points; private stats and exact HP are never exposed.
"""
from dataclasses import fields
from flax import struct
import jax
import jax.numpy as jnp
from jumanji import specs
from ninjax.events import TransitionEvents

class _Block:
    def __getitem__(self, name):
        return getattr(self, name)

@struct.dataclass
class TeamObservation(_Block):
    species: jax.Array
    species_known: jax.Array
    hp_fraction: jax.Array
    hp_known: jax.Array
    current_hp: jax.Array
    stats: jax.Array
    status: jax.Array
    level: jax.Array
    types: jax.Array
    ability: jax.Array
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

@struct.dataclass
class ActiveObservation(_Block):
    slot: jax.Array
    boosts: jax.Array
    accuracy_evasion: jax.Array
    flash_fire: jax.Array
    perish_turns: jax.Array

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

@struct.dataclass
class FieldObservation(_Block):
    turn: jax.Array
    weather: jax.Array
    weather_duration: jax.Array
    terrain: jax.Array
    terrain_duration: jax.Array
    trick_room_duration: jax.Array
    gravity_duration: jax.Array

@struct.dataclass
class DecisionObservation(_Block):
    phase: jax.Array
    acting: jax.Array
    legal_actions: jax.Array
    legal_selection: jax.Array

@struct.dataclass
class Observation:
    team: TeamObservation
    active: ActiveObservation
    sides: SideObservation
    field: FieldObservation
    decision: DecisionObservation
    transitions: TransitionEvents


def observe(env, state):
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
    item_known = sheet | relative(b.known_items >= 0)
    # Frisk reveals only the item; it must not reveal ability, nature or stats.
    items = jnp.where(sheet, relative(t.item[..., 0]), jnp.maximum(relative(b.known_items), 0))
    team = dict(species=masked(t.species[..., 0], species_known), species_known=species_known,
        hp_fraction=jnp.where(public, hp, 0), hp_known=public,
        current_hp=masked(t.current_hp[..., 0], own), stats=masked(t.stat_table.stats, own),
        status=masked(t.status[..., 0], public), level=masked(t.stat_table.level[..., 0], public), types=masked(t.type_list, public),
        ability=masked(t.ability[..., 0], sheet), item=items, item_known=item_known,
        nature=masked(jnp.concatenate((t.stat_table.nature.increased, t.stat_table.nature.decreased), -1), sheet),
        sheet_known=sheet, moves=masked(t.moves.identifier[..., 0], moves_known), moves_known=moves_known,
        move_pp=masked(t.moves.current_pp[..., 0], own), move_max_pp=masked(t.moves.max_pp[..., 0], own), move_disabled=masked(t.moves.disabled[..., 0], own),
        selected=masked(state.selected, own))
    active = dict(slot=relative(b.active_index[:, 0]), boosts=relative(b.boosts.normal_boosts),
                  accuracy_evasion=relative(b.boosts.acc_boosts),
                  flash_fire=relative(b.volatile_status.flash_fire[:, 0]), perish_turns=relative(b.perish_turns[:, 0]))
    sides = {name: relative(getattr(b, name)[:, 0]) for name in
        ('stealth_rocks', 'sticky_webs', 'spikes', 'toxic_spikes', 'reflect', 'light_screen', 'aurora_veil', 'tailwind')}
    field = {name: jnp.broadcast_to(jnp.asarray(value, jnp.int32), (2,)) for name, value in dict(
        turn=state.turn, weather=b.weather.weather, weather_duration=b.weather.duration,
        terrain=b.terrain.terrain, terrain_duration=b.terrain.duration,
        trick_room_duration=b.trick_room_duration, gravity_duration=b.gravity_duration).items()}
    decision = dict(phase=jnp.broadcast_to(state.phase, (2,)), acting=state.acting_player_mask,
                    legal_actions=env.legal_action_mask(state),
                    legal_selection=jnp.full((2, 6), state.phase == 3))
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
        'species', 'species_known', 'hp_fraction', 'hp_known', 'current_hp', 'status',
        'level', 'ability', 'item', 'item_known', 'sheet_known', 'selected')}
    team.update(stats=(2, 2, 6, 6), types=(2, 2, 6, 2), nature=(2, 2, 6, 2))
    team.update({name: (2, 2, 6, 4) for name in ('moves', 'moves_known', 'move_pp', 'move_max_pp', 'move_disabled')})
    active = {name: (2, 2) for name in ('slot', 'flash_fire', 'perish_turns')}
    active.update(boosts=(2, 2, 6), accuracy_evasion=(2, 2, 2))
    sides = {f.name: (2, 2) for f in fields(SideObservation)}
    field = {f.name: (2,) for f in fields(FieldObservation)}
    from ninjax.events import CAPACITY
    events = {f.name: (2, CAPACITY) for f in fields(TransitionEvents)}
    events.update(count=(2,), overflow=(2,))
    return specs.Spec(Observation, 'observation',
        team=block(TeamObservation, team, floats=('hp_fraction',),
                   booleans=('species_known', 'hp_known', 'item_known', 'sheet_known', 'moves_known', 'move_disabled', 'selected')),
        active=block(ActiveObservation, active, booleans=('flash_fire',)),
        sides=block(SideObservation, sides, booleans=('stealth_rocks', 'sticky_webs')),
        field=block(FieldObservation, field),
        decision=block(DecisionObservation, dict(phase=(2,), acting=(2,), legal_actions=(2, 15), legal_selection=(2, 6)),
                       booleans=('acting', 'legal_actions', 'legal_selection')),
        transitions=block(TransitionEvents, events, floats=('hp_fraction_change',), booleans=('valid', 'overflow')))
