# DISCLAIMER: Idea and code structure from blackjax
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Union

import jax.numpy as jnp
import numpy as np
from simulation_scripts import config

# from typing import Callable

Array = Union[np.ndarray, jnp.ndarray]  # either numpy or jax numpy array
PyTree = Union[
    Array, Iterable[Array], Mapping[Any, Array]
]  # either array, iterable of arrays or mapping of arrays

from dataclasses import dataclass


@dataclass(frozen=False)
class State:
    positions: PyTree
    logp: Union[float, PyTree]
    n_accepted: int
    delta: int

    def __init__(self, positions, logp, n_accepted=0, delta=0 , D_up = 0 , D_down = 0 , D_inv_up= 0 , D_inv_down= 0 , n_accepted_up = 0 , n_accepted_down = 0):
        self.positions = positions
        self.r_dist = self.positions[None, ...] - self.positions[:, None, :]
        self.logp = logp
        self.n_accepted = n_accepted
        self.n_accepted_up = n_accepted_up
        self.n_accepted_down = n_accepted_down
        self.delta = delta
        self.D_up = D_up
        self.D_down = D_down
        self.D_inv_up = D_inv_up
        self.D_inv_down = D_inv_down

        backend = config.backend

        match backend:
            case "numpy":
                self.backend = np
                self.la = np.linalg
            case "jax":
                self.backend = jnp
                self.la = jnp.linalg
                # You might also be able to jit some functions here
            case _:  # noqa
                raise ValueError("Invalid backend:", backend)

    def create_batch_of_states(self, batch_size):
        """
        # TODO: check if batch states are immutable because of the jnp
        """

        # Replicate each property of the state
        batch_positions = self.backend.array([self.positions] * batch_size)
        batch_logp = self.backend.array([self.logp] * batch_size)
        batch_n_accepted = self.backend.array([self.n_accepted] * batch_size)
        batch_n_accepted_up = self.backend.array([self.n_accepted_up] * batch_size)
        batch_n_accepted_down = self.backend.array([self.n_accepted_down] * batch_size)
        batch_delta = self.backend.array([self.delta] * batch_size)
        batch_D_up = self.backend.array([self.D_up] * batch_size)
        batch_D_down = self.backend.array([self.D_down] * batch_size)
        batch_D_inv_up = self.backend.array([self.D_inv_up] * batch_size)
        batch_D_inv_down = self.backend.array([self.D_inv_down] * batch_size)

        # Create a new State object with these batched properties
        batch_state = State(batch_positions, batch_logp, batch_n_accepted, batch_n_accepted_up, batch_n_accepted_down, batch_delta , 
                            batch_D_up , batch_D_down , batch_D_inv_up , batch_D_inv_down)
        return batch_state
