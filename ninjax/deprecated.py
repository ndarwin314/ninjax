
def standard_turn_step(
    key: chex.PRNGKey,
    state: BattleState,
    actions: (int, int)
) -> Tuple[chex.Array, BattleState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
    act1, act2 = actions
    first, second = action_order(state, actions)
    key, state = step_action(key, state, act1, first)
    # also check for like is flinched here, and check for sleep for both or something
    key, state = jax.lax.cond(
        second.active.is_alive,
        step_action,
        lambda k, s, a, _: (key, state), key, state, act2, second)

    key, state = step_field(key, state)
    # set legal action masks here
    mask = jnp.zeros((2, 15))
    alive = (state.sides[0].active.is_alive, state.sides[1].active.is_alive)
    bad = jnp.zeros((2, 6))
    for i in range(2):
        bad = bad.at[i].set(state.sides[i].legal_switch_mask())
    # make sure this broadcast works correctly
    bad = bad * alive
    mask = mask.at[:,8:14].set(bad)
    # make sure this axis is the right way
    mask = mask.at[:,15].set(1 - jnp.any(mask[:,8:14], axis=0))
    state = state.replace(legal_action_mask=mask)

    return jnp.array([0]), state, jnp.array([0]), jnp.array([0]), {}


no_op_func = lambda k, s, a, b, c: (k, s)


def switch_move_step(
    key: chex.PRNGKey,
    state: BattleState,
    actions: (int, int)
) -> (chex.PRNGKey, BattleState):
    bad = True
    mask = jnp.ones((2, 15))
    for i in range(2):
        # TODO: add assertions to verify action is legal
        is_move_action, index, is_tera, is_no_op = decode_action(actions[i])
        key, state = jax.lax.cond(is_no_op, no_op_func, step_switch, key, state, i, index, is_tera)
        is_alive = state.sides[i].active.is_alive
        mask = mask.at[i, 8:14].set(state.sides[i].legal_switch_mask())
        bad = jnp.logical_or(bad, is_alive)
    mask = mask.at[:, 0:8].mul(bad)
    state = state.replace(legal_action_mask=mask)
    # TODO: ugggghhhhhh, run it back if not bad, return the correct stuff
