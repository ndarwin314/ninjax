"""Singles Terastallization action, visibility and move-property checks."""
import jax
import jax.numpy as jnp
import pytest

from ninjax.battle.environment import Battle
from ninjax.battle.terastallization import tera_eligible, terastallize
from ninjax.battle.types import ActionType, BattleAction, BattleParams, InformationMode, Transformation
from ninjax.enums.moves import MoveType
from ninjax.enums.pokemon import Type
from ninjax.moves.properties import effective_move, resolve_move_category
from ninjax.teams.construction import make_move
from unit_tests.test_battle import neutral_type


def with_tera_type(state, player, slot, tera_type):
    """Supply a party slot's Tera type while preserving its other data."""
    team = state.team.replace(tera_type=state.team.tera_type.at[player, slot, 0].set(tera_type))
    return state.replace(team=team)


def test_tera_eligibility_and_activation(neutral_type):
    """One side spends its battle allowance and retains its Tera type."""
    _, battle = neutral_type
    battle = with_tera_type(battle, 0, 0, Type.FIRE)
    assert tera_eligible(battle)[0]
    assert not tera_eligible(battle)[1]
    battle = terastallize(battle, 0)
    assert battle.team.is_terastallized[0, 0, 0]
    assert not battle.can_tera[0, 0]
    assert not tera_eligible(battle)[0]


def test_tera_action_mask_and_visibility(neutral_type):
    """The action mask follows eligibility, and hidden types stay private."""
    key, battle = neutral_type
    battle = with_tera_type(battle, 0, 0, Type.FIRE)
    env = Battle(battle.team)
    state, timestep = env.reset(key)
    assert timestep.observation.decision.tera_actions[0, 0]
    assert not timestep.observation.decision.tera_actions[1, 0]
    assert timestep.observation.team.tera_type[0, 0, 0] == Type.FIRE
    assert timestep.observation.team.tera_type[1, 1, 0] == Type.NONE
    revealed = state.replace(battle=terastallize(state.battle, 0))
    obs = env.observe(revealed)
    assert obs.team.tera_type[1, 1, 0] == Type.FIRE
    assert obs.team.is_terastallized[1, 1, 0]
    sheet = Battle(battle.team, BattleParams(information_mode=InformationMode.OPEN_SHEET)).observe(state)
    assert sheet.team.tera_type[1, 1, 0] == Type.FIRE


@pytest.mark.jit
def test_tera_attack_validation_and_continuation(neutral_type):
    """Typed Tera survives the team-preview branch and is spent on activation."""
    key, battle = neutral_type
    battle = with_tera_type(battle, 0, 0, Type.FIRE)
    env = Battle(battle.team, BattleParams(team_preview=True))
    state, _ = env.reset(key)
    step = jax.jit(env.step)
    preview = BattleAction(jnp.zeros(2, jnp.int32), jnp.full(2, ActionType.NO_OP),
                           selection=jnp.ones((2, 6), bool))
    state, _ = step(state, preview)
    actions = BattleAction(jnp.zeros(2, jnp.int32), jnp.full(2, ActionType.ATTACK),
                           transformation=jnp.array([Transformation.TERA, Transformation.NONE]))
    pp = state.battle.active_move_field('current_pp', 0)[0, 0]
    state, timestep = step(state, actions)
    assert not timestep.extras['invalid_action']
    assert state.battle.team.is_terastallized[0, 0, 0]
    assert not state.battle.can_tera[0, 0]
    assert state.battle.active_move_field('current_pp', 0)[0, 0] == pp - 1


def test_tera_blast_changes_type_and_category(neutral_type):
    """Tera Blast resolves its user type and the greater staged attacking stat."""
    key, state = neutral_type
    state = with_tera_type(state, 0, 0, Type.WATER)
    move = make_move('Tera Blast')
    assert effective_move(state, 0, move).type[0] == Type.NORMAL
    state = terastallize(state, 0)
    move = effective_move(state, 0, move)
    assert move.type[0] == Type.WATER
    _, resolved = resolve_move_category(key, move, 50, 200, 100, 100, 100)
    assert resolved.move_type[0] == MoveType.PHYSICAL
