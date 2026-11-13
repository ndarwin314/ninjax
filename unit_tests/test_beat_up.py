"""Beat Up builds its hits from eligible party members' base Attack."""
import jax.numpy as jnp
from ninjax.enum_types import Status
from ninjax.game_logic import do_damaging_move
from ninjax.prototype import make_move
from unit_tests.test_battle import neutral_type


def test_beat_up_uses_eligible_party(neutral_type):
    """Only the user and selected, healthy, status-free teammates contribute."""
    key, state = neutral_type
    team = state.team.replace(status=state.team.status.at[0, 0, 0].set(Status.BURN).at[0, 2, 0].set(Status.POISON),
        current_hp=state.team.current_hp.at[0, 3, 0].set(0))
    roster = state.roster.at[0, 4:].set(False)
    state = state.replace(team=team, roster=roster)
    _, out = do_damaging_move(key, state, 0, make_move('Beat Up'), 0, 0)
    assert jnp.sum(out.hit_results.data[:, 4] != 0) == 2
    assert out.active.current_hp[1, 0] < state.active.current_hp[1, 0]


def test_beat_up_base_attack_drives_hit_power(neutral_type):
    """Increasing a contributor's base Attack increases its hit damage."""
    key, state = neutral_type
    state = state.replace(roster=state.roster.at[0, 2:].set(False))
    _, ordinary = do_damaging_move(key, state, 0, make_move('Beat Up'), 0, 0)
    table = state.team.stat_table
    # Change only a bench contributor, leaving the user's Attack unchanged.
    table = table.replace(base_stats=table.base_stats.at[0, 1, 1].set(150))
    stronger = state.replace(team=state.team.replace(stat_table=table))
    _, boosted = do_damaging_move(key, stronger, 0, make_move('Beat Up'), 0, 0)
    assert boosted.hit_results[0].damage == ordinary.hit_results[0].damage
    assert boosted.active.current_hp[1, 0] < ordinary.active.current_hp[1, 0]
