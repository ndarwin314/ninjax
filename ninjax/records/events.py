"""Fixed-capacity, ordered transition events; no Python containers grow while tracing."""
from functools import partial
from enum import IntEnum

import jax
import jax.numpy as jnp
from flax import struct


CAPACITY = 128

class EventKind(IntEnum):
    """Eventkind."""
    HP = 1
    STATUS = 2
    BOOST = 3
    SWITCH = 4
    WEATHER = 5
    TERRAIN = 6
    CONDITION = 7
    MOVE = 8
    MISS = 9
    IMMUNE = 10
    FAINT = 11
    INTERRUPTED = 12
    CRITICAL = 13
    ITEM = 14
    ABILITY = 15
    ITEM_REVEALED = 16
    MOVE_FAILED = 17

class Cause(IntEnum):
    """Cause."""
    OTHER = 0
    MOVE = 1
    INDIRECT = 2
    SWITCH = 3
    FIELD = 4
    RECOIL = 5
    CONTACT = 6
    WEATHER = 7
    TERRAIN = 8
    STATUS = 9
    PERISH = 10
    CONFUSION = 11
    DESTINY_BOND = 12
    SELF_KO = 13

@struct.dataclass
class TransitionEvents:
    """Fixed-capacity public event arrays with valid-entry and overflow tracking."""
    valid: jax.Array
    kind: jax.Array
    source: jax.Array
    target: jax.Array
    slot: jax.Array
    detail: jax.Array
    cause: jax.Array
    move: jax.Array
    before: jax.Array
    after: jax.Array
    hp_change: jax.Array
    hp_fraction_change: jax.Array
    count: jax.Array
    overflow: jax.Array

    @classmethod
    def empty(cls):
        """Empty."""
        return cls(valid=jnp.zeros(CAPACITY, bool),
                   **{name: jnp.zeros(CAPACITY, jnp.int32) for name in
                      ('kind', 'source', 'target', 'slot', 'detail', 'cause', 'move', 'before', 'after', 'hp_change')},
                   hp_fraction_change=jnp.zeros(CAPACITY, jnp.float32),
                   count=jnp.int32(0), overflow=jnp.bool_(False))


INT_FIELDS = ('valid', 'kind', 'source', 'target', 'slot', 'detail', 'cause', 'move', 'before', 'after', 'hp_change')

@struct.dataclass
class EventBuffer:
    """Compact ordered event rows with a fixed capacity and overflow tracking."""
    data: jax.Array
    hp_fraction_change: jax.Array
    count: jax.Array
    overflow: jax.Array

    @classmethod
    def empty(cls):
        """Empty."""
        return cls(jnp.zeros((CAPACITY, len(INT_FIELDS)), jnp.int32),
                   jnp.zeros(CAPACITY, jnp.float32), jnp.int32(0), jnp.bool_(False))

    def unpack(self):
        """Unpack."""
        return TransitionEvents(**{name: getattr(self, name) for name in INT_FIELDS},
            hp_fraction_change=self.hp_fraction_change, count=self.count, overflow=self.overflow)

    @property
    def valid(self):
        """Valid."""
        return self.data[..., 0].astype(bool)

    @property
    def kind(self):
        """Kind."""
        return self.data[..., 1]

    @property
    def source(self):
        """Source."""
        return self.data[..., 2]

    @property
    def target(self):
        """Target."""
        return self.data[..., 3]

    @property
    def slot(self):
        """Slot."""
        return self.data[..., 4]

    @property
    def detail(self):
        """Detail."""
        return self.data[..., 5]

    @property
    def cause(self):
        """Cause."""
        return self.data[..., 6]

    @property
    def move(self):
        """Move."""
        return self.data[..., 7]

    @property
    def before(self):
        """Before."""
        return self.data[..., 8]

    @property
    def after(self):
        """After."""
        return self.data[..., 9]

    @property
    def hp_change(self):
        """Hp change."""
        return self.data[..., 10]

def emit(state, kind, target=-1, detail=0, before=0, after=0,
         hp_change=0, hp_fraction_change=0., enabled=True):
    """Append only actual events. Overflow is explicit and never overwrites history."""
    events = _append_event(state.events, state.active_index, state.event_source,
                           state.event_cause, state.event_move, kind, target, detail,
                           before, after, hp_change, hp_fraction_change, enabled)
    return state.replace(events=events)


def _append_event(e, active_index, source, cause, move, kind, target, detail, before, after,
                  hp_change, hp_fraction_change, enabled):
    """Append one event using the buffer and the fields recorded in its rows."""
    enabled = jnp.asarray(enabled).reshape(())
    room = e.count < CAPACITY
    target = jnp.asarray(target, jnp.int32).reshape(())
    slot = jnp.where(target >= 0, active_index[jnp.maximum(target, 0), 0], -1)
    values = dict(valid=True, kind=kind, source=source, target=target,
                  slot=slot, detail=detail, cause=cause, move=move,
                  before=before, after=after, hp_change=hp_change,
                  hp_fraction_change=hp_fraction_change)
    index = jnp.where(enabled & room, e.count, CAPACITY)
    row = jnp.stack([jnp.asarray(values[name], jnp.int32).reshape(()) for name in INT_FIELDS])
    e = e.replace(data=e.data.at[index].set(row, mode='drop'),
                  hp_fraction_change=e.hp_fraction_change.at[index].set(jnp.asarray(hp_fraction_change, jnp.float32).reshape(()), mode='drop'),
                  count=e.count + (enabled & room).astype(jnp.int32))
    return e.replace(overflow=e.overflow | (enabled & ~room))


_SIDE_CONDITIONS = ('stealth_rocks', 'sticky_webs', 'spikes', 'toxic_spikes',
                    'reflect', 'light_screen', 'aurora_veil', 'tailwind', 'perish_turns')


def condition_values(state):
    """Read the field and side values used for scheduler condition events."""
    shared = jnp.array([state.weather.weather, state.terrain.terrain], jnp.int32)
    sides = tuple(getattr(state, name)[:, 0] for name in _SIDE_CONDITIONS)
    endings = jnp.array([state.trick_room_duration, state.gravity_duration], jnp.int32)
    return jnp.concatenate((shared, *sides, state.volatile_status.flash_fire[:, 0], endings)).astype(jnp.int32)


def capture_conditions(before, after):
    """Record field and side changes from two battle states."""
    return capture_conditions_from_values(condition_values(before), after)


def capture_conditions_from_values(old, after):
    """Record condition changes in their existing field and side order."""
    # Detail IDs 0–8 are side conditions, 9 is Flash Fire, and 10–11
    # are Trick Room and Gravity. Other volatile events are emitted elsewhere.
    sides = len(_SIDE_CONDITIONS) + 1
    kinds = jnp.array([EventKind.WEATHER, EventKind.TERRAIN] + [EventKind.CONDITION] * (sides * 2 + 2))
    targets = jnp.array([-1, -1] + [0, 1] * sides + [-1, -1])
    details = jnp.array([0, 0] + [detail for detail in range(sides) for _ in range(2)] + [10, 11])
    record = partial(_record_condition, active_index=after.active_index, source=after.event_source,
                     cause=after.event_cause, move=after.event_move)
    events, _ = jax.lax.scan(record, after.events, (kinds, targets, details, old, condition_values(after)))
    return after.replace(events=events)


def _record_condition(events, row, *, active_index, source, cause, move):
    """Append a changed condition without carrying the battle through the scan."""
    kind, target, detail, a, b = row
    return _append_event(events, active_index, source, cause, move, kind, target, detail,
                         a, b, 0, 0., a != b), None
