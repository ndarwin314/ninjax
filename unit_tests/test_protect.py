"""Protect and Unseen Fist singles interactions."""
from ninjax.battle.residuals import step_side_conditions
from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.moves import MoveFlags
from ninjax.moves.routing import move_hits
from ninjax.moves.status import do_status_move
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type


def test_protect_blocks_and_unseen_fist_bypasses(neutral_type):
    """Protect blocks contact unless the attacking ability bypasses it."""
    key, state = neutral_type
    protect = make_move('Protect')
    key, state = do_status_move(key, state, 1, protect, 0, 0)
    assert state.volatile_status.protection.protected[1, 0]
    assert state.volatile_status.protection.protect_chain[1, 0] == 1
    flags = state.team.moves.move_flags.at[0, 0, 0, 0].set(int(MoveFlags.CONTACT))
    state = state.replace(team=state.team.replace(moves=state.team.moves.replace(move_flags=flags)))
    hp = state.active.current_hp[1, 0]
    _, blocked = move_hits(key, state, 0, 0)
    assert blocked.active.current_hp[1, 0] == hp
    state = state.replace(team=state.team.replace(ability=state.team.ability.at[0, 0, 0].set(AbilityEnum.UNSEEN_FIST)))
    _, hit = move_hits(key, state, 0, 0)
    assert hit.active.current_hp[1, 0] < hp
    _, cleared = step_side_conditions(key, hit)
    assert not cleared.volatile_status.protection.protected[1, 0]
    assert cleared.volatile_status.protection.protect_chain[1, 0] == 1
