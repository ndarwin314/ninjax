"""Compare user-written intrinsic power helpers with actual Showdown callbacks."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from ninjax.calculations.damage import (
    electro_ball_base_power,
    gyro_ball_base_power,
    proportional_hp_base_power,
    status_base_power,
    target_hp_base_power,
)
from ninjax.enums.abilities import AbilityEnum
from ninjax.enums.pokemon import Status


REFERENCE = json.loads((Path(__file__).parent / 'fixtures/showdown/variable-power.json').read_text())
STATUS_IDS = {'': Status.NONE, 'brn': Status.BURN, 'psn': Status.POISON,
              'tox': Status.TOXIC, 'par': Status.PARALYZE, 'slp': Status.SLEEP,
              'frz': Status.FREEZE}


@pytest.fixture(scope='module')
def power_helpers():
    """Compile each helper once with runtime numeric inputs."""
    return {'status': jax.jit(status_base_power),
            'user_hp': jax.jit(proportional_hp_base_power),
            'target_hp': jax.jit(target_hp_base_power),
            'electro': jax.jit(electro_ball_base_power),
            'gyro': jax.jit(gyro_ball_base_power)}


@pytest.mark.parametrize('probe', REFERENCE['probes'],
                         ids=[f"{p['helper']}-{i}" for i, p in enumerate(REFERENCE['probes'])])
def test_variable_power_matches_reference(power_helpers, probe):
    """Check status, HP rounding and speed boundaries against the reference."""
    args = list(probe['args'])
    if probe['helper'] == 'status':
        args[1] = STATUS_IDS[args[1]]
        args[2] = AbilityEnum.COMATOSE if args[2] else AbilityEnum.NONE
    result = power_helpers[probe['helper']](*(jnp.asarray(arg, jnp.int32) for arg in args))
    assert result == probe['expected']
    assert result.dtype != jnp.bool_
