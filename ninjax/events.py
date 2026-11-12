"""Fixed-capacity, ordered transition events; no Python containers grow while tracing."""
from enum import IntEnum
from flax import struct
import jax
import jax.numpy as jnp

CAPACITY = 128

class EventKind(IntEnum):
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

class Cause(IntEnum):
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

@struct.dataclass
class TransitionEvents:
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
        return cls(valid=jnp.zeros(CAPACITY, bool),
                   **{name: jnp.zeros(CAPACITY, jnp.int32) for name in
                      ('kind', 'source', 'target', 'slot', 'detail', 'cause', 'move', 'before', 'after', 'hp_change')},
                   hp_fraction_change=jnp.zeros(CAPACITY, jnp.float32),
                   count=jnp.int32(0), overflow=jnp.bool_(False))


INT_FIELDS = ('valid', 'kind', 'source', 'target', 'slot', 'detail', 'cause', 'move', 'before', 'after', 'hp_change')

@struct.dataclass
class EventBuffer:
    """Packed internal storage keeps every append to two indexed writes."""
    data: jax.Array
    hp_fraction_change: jax.Array
    count: jax.Array
    overflow: jax.Array

    @classmethod
    def empty(cls):
        return cls(jnp.zeros((CAPACITY, len(INT_FIELDS)), jnp.int32),
                   jnp.zeros(CAPACITY, jnp.float32), jnp.int32(0), jnp.bool_(False))

    def unpack(self):
        return TransitionEvents(**{name: getattr(self, name) for name in INT_FIELDS},
            hp_fraction_change=self.hp_fraction_change, count=self.count, overflow=self.overflow)

    @property
    def valid(self):
        return self.data[..., 0].astype(bool)

    @property
    def kind(self):
        return self.data[..., 1]

    @property
    def source(self):
        return self.data[..., 2]

    @property
    def target(self):
        return self.data[..., 3]

    @property
    def slot(self):
        return self.data[..., 4]

    @property
    def detail(self):
        return self.data[..., 5]

    @property
    def cause(self):
        return self.data[..., 6]

    @property
    def move(self):
        return self.data[..., 7]

    @property
    def before(self):
        return self.data[..., 8]

    @property
    def after(self):
        return self.data[..., 9]

    @property
    def hp_change(self):
        return self.data[..., 10]

def emit(state, kind, target=-1, detail=0, before=0, after=0,
         hp_change=0, hp_fraction_change=0., enabled=True):
    """Append only actual events. Overflow is explicit and never overwrites history."""
    e = state.events
    enabled = jnp.asarray(enabled).reshape(())
    room = e.count < CAPACITY
    target = jnp.asarray(target, jnp.int32).reshape(())
    slot = jnp.where(target >= 0, state.active_index[jnp.maximum(target, 0), 0], -1)
    values = dict(valid=True, kind=kind, source=state.event_source, target=target,
                  slot=slot, detail=detail, cause=state.event_cause, move=state.event_move,
                  before=before, after=after, hp_change=hp_change,
                  hp_fraction_change=hp_fraction_change)
    index = jnp.where(enabled & room, e.count, CAPACITY)
    row = jnp.stack([jnp.asarray(values[name], jnp.int32).reshape(()) for name in INT_FIELDS])
    e = e.replace(data=e.data.at[index].set(row, mode='drop'),
                  hp_fraction_change=e.hp_fraction_change.at[index].set(jnp.asarray(hp_fraction_change, jnp.float32).reshape(()), mode='drop'),
                  count=e.count + (enabled & room).astype(jnp.int32))
    return state.replace(events=e.replace(overflow=e.overflow | (enabled & ~room)))


def capture_conditions(before, after):
    """Record field/side changes not already emitted by HP/status/stage helpers.

    CONDITION detail IDs: hazards 0..3, screens 4..6, tailwind 7,
    Perish Song 8, Flash Fire 9, Trick Room 10, gravity 11.
    """
    kinds, targets, details, old, new = [], [], [], [], []
    def add(kind, target, detail, a, b):
        kinds.append(kind)
        targets.append(target)
        details.append(detail)
        old.append(jnp.asarray(a, jnp.int32))
        new.append(jnp.asarray(b, jnp.int32))
    add(EventKind.WEATHER, -1, 0, before.weather.weather, after.weather.weather)
    add(EventKind.TERRAIN, -1, 0, before.terrain.terrain, after.terrain.terrain)
    names = ('stealth_rocks', 'sticky_webs', 'spikes', 'toxic_spikes',
             'reflect', 'light_screen', 'aurora_veil', 'tailwind', 'perish_turns')
    for detail, name in enumerate(names):
        for player in range(2):
            add(EventKind.CONDITION, player, detail, getattr(before, name)[player, 0], getattr(after, name)[player, 0])
    for player in range(2):
        add(EventKind.CONDITION, player, 9, before.volatile_status.flash_fire[player, 0], after.volatile_status.flash_fire[player, 0])
    for detail, name in ((10, 'trick_room_duration'), (11, 'gravity_duration')):
        add(EventKind.CONDITION, -1, detail, getattr(before, name), getattr(after, name))
    data = tuple(jnp.asarray(x) for x in (kinds, targets, details, old, new))
    def record(state, row):
        kind, target, detail, a, b = row
        return emit(state, kind, target, detail, a, b, enabled=a != b), None
    return jax.lax.scan(record, after, data)[0]
