import haiku as hk  # type: ignore
from haiku import nets
from jax import nn  # type: ignore
from jax import numpy as jnp
from jax import vmap
from jaxtyping import Array

"""From Paper: An Attention Free Transformer
Authors: Zhai, ..., Susskind

Key equation are Equation 2 and 3.
"""


def attention_free_per_position(q_t: Array, k: Array, v: Array, w_t: Array) -> Array:
    """
    q_t:    [d]
    k:      [t,d]
    v:      [t,d]
    w_t:    [d]
    """
    weights = jnp.exp(k + w_t[:, None])  # [t,d]
    sum_weights = jnp.sum(weights, axis=0)  # [d]

    dotted = weights * v
    weighted_sum_values = jnp.sum(dotted, axis=0)  # [d]

    weighted_average_values = weighted_sum_values / sum_weights

    y = nn.sigmoid(q_t) * weighted_average_values
    return y


attention_free_per_sample = vmap(
    attention_free_per_position, in_axes=(0, None, None, 0)
)  # q_t->q, w_t->w
attention_free_function = vmap(
    attention_free_per_sample, in_axes=(0, 0, 0, None)
)  # q,k,v are now batches, w is shared


def attention_free_block(
    queries: Array,
    keys: Array,
    values: Array,
    learnable_position_biases: Array,
    is_training: bool,
    embed_dim: int,
) -> Array:
    """
    queries:                    [n s e]
    keys:                       [n s e]
    values:                     [n s e]
    learnable_position_biases   [s s]
    """
    q = nets.MLP([embed_dim])(queries)
    k = nets.MLP([embed_dim])(keys)
    v = nets.MLP([embed_dim])(values)

    z = attention_free_function(q, k, v, learnable_position_biases)

    z = hk.LayerNorm(-1, True, True)(z + queries)

    return z
