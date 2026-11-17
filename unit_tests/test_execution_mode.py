"""Verify the execution mode used by ordinary mechanics tests."""
import jax


def test_jit_is_enabled_for_ordinary_tests():
    """An unmarked test traces a function wrapped in jax.jit."""
    def increment(value):
        """Require tracing before the compiled function runs."""
        assert isinstance(value, jax.core.Tracer)
        return value + 1

    assert not jax.config.jax_disable_jit
    assert jax.jit(increment)(1) == 2
