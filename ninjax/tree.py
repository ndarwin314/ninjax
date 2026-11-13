"""Array operations for immutable Flax PyTrees with explicit leading axes."""
import jax
import jax.numpy as jnp
from flax import struct


@jax.jit
def _index_tree(tree, index):
    """Share dynamic indexing across repeated views of the same tree shape.

    Indexing inside a cached jit keeps every leaf's gather/slice operations
    from being retraced at each Pokémon, move or stat-table read.
    """
    return jax.tree.map(lambda value: value[index], tree)


class ArrayTree(struct.PyTreeNode):
    def __getitem__(self, index):
        """Index all leaves, sharing dynamic gathers and retaining slice syntax."""
        parts = index if isinstance(index, tuple) else (index,)
        # Slice/Ellipsis objects are static Python indexing syntax, not JAX
        # arguments. Preserve these less frequent views without a jit boundary.
        if any(isinstance(part, slice) or part is Ellipsis for part in parts):
            return jax.tree.map(lambda value: value[index], self)
        return _index_tree(self, index)


def stack(trees, axis=0):
    return jax.tree.map(lambda *values: jnp.stack(values, axis=axis), *trees)


def broadcast_tree(tree, batch_shape):
    # Each leaf retains its feature axis; the caller supplies the entity axes.
    return jax.tree.map(lambda value: jnp.broadcast_to(value, batch_shape + value.shape[-1:]), tree)
