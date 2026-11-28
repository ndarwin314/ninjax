"""Gen 9 singles Terastallization eligibility and activation."""
import jax.numpy as jnp

from ninjax.enums.pokemon import Type
from ninjax.records.events import Cause, EventKind, emit
from ninjax.side.updates import update_active_fields


def tera_eligible(battle):
    """Return per-player eligibility for ordinary, represented Tera types."""
    tera_type = battle.active_field('tera_type')[:, 0]
    species = battle.active_identity('species')[:, 0]
    # The Ogerpon and Terapagos transformation forms require dedicated
    # after-Terastallization lifecycles.
    ordinary_species = (species != 1017) & (species != 1024)
    return (battle.can_tera[:, 0] & (battle.active_field('current_hp')[:, 0] > 0) &
            ~jnp.any(battle.team.is_terastallized[..., 0], axis=1) &
            (tera_type >= Type.NORMAL) & (tera_type <= Type.FAIRY) & ordinary_species)


def terastallize(battle, player):
    """Transform one eligible active Pokémon and record its public type."""
    tera_type = battle.active_field('tera_type', player)[0]
    old_context = battle.event_context
    battle = battle.replace(event_context=old_context.replace(
        source=jnp.asarray(player, jnp.int32), move_id=jnp.int32(0), cause=jnp.int32(Cause.OTHER)))
    battle = update_active_fields(battle, player, is_terastallized=jnp.array([True]))
    v = battle.volatile_status
    battle = battle.replace(
        can_tera=battle.can_tera.at[player, 0].set(False),
        volatile_status=v.replace(properties=v.properties.replace(
            added_type=v.properties.added_type.at[player, 0].set(Type.NONE))))
    battle = emit(battle, EventKind.TYPE, player, detail=tera_type, after=tera_type)
    return battle.replace(event_context=old_context)
