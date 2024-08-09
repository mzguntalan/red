import os
import pickle
import sys
from pathlib import Path
from typing import Tuple
from typing import Union

import haiku as hk
import jax
import numpy as np
import optax


def save(ckpt_dir: str, state: Union[hk.Params, hk.State, optax.OptState]) -> None:
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(ckpt_dir, "arrays.npy"), "wb") as f:
        for x in jax.tree_util.tree_leaves(state):
            np.save(f, x, allow_pickle=False)

    tree_struct = jax.tree_map(lambda t: 0, state)
    with open(os.path.join(ckpt_dir, "tree.pkl"), "wb") as f:
        pickle.dump(tree_struct, f)


def save_triple(
    checkpoint_dir: str, params: hk.Params, state: hk.State, opt_state: optax.OptState
) -> None:
    save(checkpoint_dir + "/params", params)
    save(checkpoint_dir + "/state", state)
    save(checkpoint_dir + "/optimizer", opt_state)


def restore(ckpt_dir: str) -> Union[hk.Params, hk.State, optax.OptState]:
    with open(os.path.join(ckpt_dir, "tree.pkl"), "rb") as f:
        tree_struct = pickle.load(f)

    leaves, treedef = jax.tree_flatten(tree_struct)
    with open(os.path.join(ckpt_dir, "arrays.npy"), "rb") as f:
        flat_state = [np.load(f) for _ in leaves]

    return jax.tree_unflatten(treedef, flat_state)


def restore_triple(checkpoint_dir: str) -> Tuple[hk.Params, hk.State, optax.OptState]:
    return (
        restore(checkpoint_dir + "/params"),
        restore(checkpoint_dir + "/state"),
        restore(checkpoint_dir + "/optimizer"),
    )
