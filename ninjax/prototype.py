"""Runnable, explicit neutral-ability/item sandbox based on captured Showdown teams.

python -m ninjax.prototype --reference unit_tests/fixtures/showdown/battle-1.json
"""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp

from ninjax.battle.environment import Battle
from ninjax.battle.types import BattleParams
from ninjax.teams.construction import make_sandbox_teams
from ninjax.teams.curated import (
    DEFAULT_SETS,
    load_sets,
    make_ability_pool,
    make_item_pool,
    make_sandbox_pool,
    sample_teams,
)


@jax.jit
def random_actions(key, mask):
    """Sample the 90/10 move/switch policy, honoring forced continue and no-op."""
    roll_key, move_key, switch_key = jax.random.split(key, 3)
    moves = mask[:, :5]
    switches = mask[:, 8:14]
    move = jax.random.categorical(move_key, jnp.where(moves, 0., -jnp.inf), axis=-1)
    switch = 8 + jax.random.categorical(switch_key, jnp.where(switches, 0., -jnp.inf), axis=-1)
    choose_switch = jnp.any(switches, axis=1) & ((jax.random.uniform(roll_key, (2,)) < .1) | ~jnp.any(moves, axis=1))
    choice = jnp.where(choose_switch, switch, move).astype(jnp.int32)
    # Charge releases and recharge turns expose only legacy action 5.
    choice = jnp.where(mask[:, 5], 5, choice)
    return jnp.where(mask[:, 14], 14, choice)


def run(reference_path, seed=0, max_decisions=1000):
    reference = json.loads(Path(reference_path).read_text())
    teams, omissions = make_sandbox_teams(reference)
    result = _run_teams(teams, omissions, seed, max_decisions)
    return {'reference_revision': reference['revision'], **result}


@lru_cache(maxsize=4)
def _sandbox_step(max_turns):
    # Teams live in the dynamic state, so different samples share compilation.
    env = Battle(params=BattleParams(max_steps_in_episode=max_turns))
    return jax.jit(env.step)


def _run_teams(teams, omissions, seed, max_decisions, max_turns=100):
    env = Battle(teams, BattleParams(max_steps_in_episode=max_turns))
    key, battle_key = jax.random.split(jax.random.key(seed))
    state, timestep = env.reset(battle_key)
    step = _sandbox_step(max_turns)
    decisions = []
    for _ in range(max_decisions):
        if bool(timestep.last()):
            break
        key, policy_key = jax.random.split(key)
        actions = random_actions(policy_key, timestep.extras['legal_action_mask'])
        state, timestep = step(state, actions)
        if bool(timestep.extras['invalid_action']):
            raise RuntimeError('Policy selected an invalid action')
        decisions.append({'turn': int(state.turn), 'phase': int(state.phase), 'actions': actions.tolist(),
                          'active': state.battle.active_index[:, 0].tolist(),
                          'hp': state.battle.active.current_hp[:, 0].tolist()})
    if not bool(timestep.last()):
        raise RuntimeError('Decision cap reached without termination')
    return {'seed': seed, 'sandbox': 'neutral abilities/items; unsupported moves unavailable',
            'omissions': omissions, 'turns': int(state.turn), 'reward': timestep.reward.tolist(), 'decisions': decisions}



def run_curated(path=None, seed=0, max_decisions=1000, max_turns=100, abilities=False, items=False):
    """Generate and run two teams from curated exports, reporting all omissions."""
    path = Path(path) if path is not None else DEFAULT_SETS
    sets = load_sets(path)
    pool, reports = make_item_pool(sets, abilities=abilities) if items else (make_ability_pool if abilities else make_sandbox_pool)(sets)
    teams, indices = sample_teams(jax.random.fold_in(jax.random.key(seed), 0), pool)
    chosen = indices.tolist()
    omissions = [reports[i] for side in chosen for i in side]
    result = _run_teams(teams, omissions, seed, max_decisions, max_turns)
    if abilities:
        result['sandbox'] = 'verified curated abilities enabled; neutral items; unsupported moves/abilities omitted'
    if items:
        result['sandbox'] = ('represented curated items enabled; ' +
            ('represented abilities enabled' if abilities else 'neutral abilities') +
            '; unsupported effects omitted')
    return {**result, 'curated_source': str(path), 'set_indices': chosen,
            'teams': [[sets[i].species for i in side] for side in chosen]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--reference')
    source.add_argument('--curated', nargs='?', const=str(Path(__file__).parent / 'data' / 'curated-sets.txt'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default=None)
    parser.add_argument('--abilities', action='store_true', help='Enable verified abilities with --curated')
    parser.add_argument('--items', action='store_true', help='Enable represented items with --curated')
    args = parser.parse_args()
    if args.abilities and not args.curated:
        parser.error('--abilities requires --curated')
    if args.items and not args.curated:
        parser.error('--items requires --curated')
    result = run_curated(args.curated, args.seed, abilities=args.abilities, items=args.items) if args.curated else run(args.reference, args.seed)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('decisions', 'omissions')}))
    print(f"{len(result['decisions'])} decisions; omissions are included in --output")


if __name__ == '__main__':
    main()
