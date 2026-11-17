"""Move item hooks and compatibility exports for extracted effects."""

from ninjax.moves.finalization import (
    _apply_throat_spray as _apply_throat_spray,
    after_move as after_move,
)


# Compatibility exports for the extracted battle phases.
from ninjax.residuals import leftovers as leftovers, residual_orbs as residual_orbs


from ninjax.side.queries import (
    is_choice_item as is_choice_item,
    berry_allowed as berry_allowed,
    contact_reactions_allowed as contact_reactions_allowed,
    conditional_mult_round as conditional_mult_round,
    effective_weather as effective_weather,
    item_transfer_allowed as item_transfer_allowed,
    can_remove_item as can_remove_item,
    modify_stats as modify_stats,
)
from ninjax.side.item_effects import (
    cure_status as cure_status,
    cheek_pouch as cheek_pouch,
    status_berry as status_berry,
    white_herb as white_herb,
    update_items as update_items,
    kings_rock_flinch as kings_rock_flinch,
    consume as consume,
    _assign_item as _assign_item,
    _activate_received_item as _activate_received_item,
    remove_item as remove_item,
    swap_items as swap_items,
    refresh_paradox as refresh_paradox,
    after_hp_change as after_hp_change,
    after_hit as after_hit,
)
