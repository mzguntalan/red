"""From Paper: Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention (2020)
Authors: Katharopoulos, ..., Fleuret

Key equations are Equation 5 and 6.
"""

import haiku as hk  # type: ignore
from haiku import nets
from jax import nn  # type: ignore
from jax import numpy as jnp
from jaxtyping import Array


_lecun_init = hk.initializers.VarianceScaling(2.0, "fan_in", "truncated_normal")


def dot(x: Array, y: Array) -> Array:
    r: Array = jnp.squeeze(
        jnp.expand_dims(x, axis=-2) @ jnp.expand_dims(y, axis=-1), axis=(-1, -2)
    )

    return r


def fmap(x: jnp.ndarray) -> jnp.ndarray:
    return nn.elu(x) + 1


def linear_attention(q: Array, k: Array, v: Array) -> Array:
    q = fmap(q)
    k = fmap(k)
    kT = jnp.transpose(k, [0, 1, 3, 2])
    z = q @ (kT @ v)
    d = q @ (kT @ jnp.ones_like(v)) + 1e-17

    v = z / d

    return v


def linear_transformer_block(
    queries: Array,
    keys: Array,
    values: Array,
    is_training: bool,
    embed_dim: int,
    num_heads: int,
    ff_dim: int,
) -> Array:
    q = nets.MLP([embed_dim])(queries)
    k = nets.MLP([embed_dim])(keys)
    v = nets.MLP([embed_dim])(values)

    q = jnp.reshape(q, [q.shape[0], q.shape[1], num_heads, q.shape[2] // num_heads])
    k = jnp.reshape(k, [k.shape[0], k.shape[1], num_heads, k.shape[2] // num_heads])
    v = jnp.reshape(v, [v.shape[0], v.shape[1], num_heads, v.shape[2] // num_heads])

    q = jnp.transpose(q, [0, 2, 1, 3])
    k = jnp.transpose(k, [0, 2, 1, 3])
    v = jnp.transpose(v, [0, 2, 1, 3])

    z = linear_attention(q, k, v)

    z = jnp.transpose(z, [0, 2, 1, 3])
    z = jnp.reshape(z, [z.shape[0], z.shape[1], -1])
    z = hk.LayerNorm(-1, True, True)(z + queries)

    z = hk.LayerNorm(-1, True, True)(
        z + nets.MLP([ff_dim, embed_dim], activate_final=False)(z)
    )

    return z
