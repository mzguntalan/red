from jax import jit
from jax import numpy as jnp
from jax import vmap
from jaxtyping import Array


@jit
def pearson_one_sample(x: Array, y: Array) -> Array:
    return jnp.nan_to_num(jnp.corrcoef(x, y)[0, 1])


batch_pearson = jit(vmap(pearson_one_sample, in_axes=0))


@jit
def weighted_pearson_one_sample(x: Array, y: Array, weights: Array) -> Array:
    cov_matrix = jnp.cov(x, y, fweights=weights)
    cov_xy = cov_matrix[0, 1]
    var_x = cov_matrix[0, 0]
    var_y = cov_matrix[1, 1]
    std_x = jnp.sqrt(var_x)
    std_y = jnp.sqrt(var_y)

    pcorr = (cov_xy) / (std_x * std_y)

    return pcorr


batch_weighted_pearson = jit(vmap(weighted_pearson_one_sample, in_axes=0))
