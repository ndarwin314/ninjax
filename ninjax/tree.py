"""Array operations for immutable Flax PyTrees with explicit leading axes."""
import jax
import jax.numpy as jnp
from flax import struct


class ArrayTree(struct.PyTreeNode):
    def __getitem__(self, index):
        return jax.tree.map(lambda value: value[index], self)


def stack(trees, axis=0):
    return jax.tree.map(lambda *values: jnp.stack(values, axis=axis), *trees)


def broadcast_tree(tree, batch_shape):
    # Each leaf retains its feature axis; the caller supplies the entity axes.
    return jax.tree.map(lambda value: jnp.broadcast_to(value, batch_shape + value.shape[-1:]), tree)
