"""Fresh-process regression for importing ability constants during a JAX trace."""
import subprocess
import sys
import textwrap


def test_ability_rules_import_inside_trace_does_not_leak():
    """Rule constants remain importable during a trace without loading effects."""
    script = textwrap.dedent('''
        import sys
        import jax
        import jax.numpy as jnp
        from ninjax.enums.abilities import AbilityEnum

        assert 'ninjax.ability_rules' not in sys.modules
        def first_trace(value):
            # Constants now have an owner independent of reactive mechanics.
            from ninjax.ability_rules import UNCHANGEABLE_ABILITIES
            return value + len(UNCHANGEABLE_ABILITIES)

        with jax.checking_leaks():
            jax.make_jaxpr(first_trace)(jnp.int32(0))

        assert 'ninjax.ability_effects' not in sys.modules
        assert 'ninjax.side' not in sys.modules
        from ninjax.ability_rules import UNCHANGEABLE_ABILITIES
        assert isinstance(UNCHANGEABLE_ABILITIES, tuple)
        def is_unchangeable(ability):
            return jnp.any(jnp.asarray(UNCHANGEABLE_ABILITIES, dtype=jnp.int32) == ability)
        check = jax.jit(is_unchangeable)
        assert bool(check(jnp.int32(AbilityEnum.COMATOSE)))
        assert not bool(check(jnp.int32(AbilityEnum.LEVITATE)))
    ''')
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
