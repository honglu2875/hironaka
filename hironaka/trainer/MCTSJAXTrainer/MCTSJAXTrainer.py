from typing import Union, Tuple

from jax import numpy as jnp


# ------------------ [UNDER CONSTRUCTION] ------------------ #


class MCTSJAXTrainer:
    pass


class JAXObs:
    """
    A class unifying the observation of host and agent.
    """
    def __init__(self, role: str, data: Union[jnp.ndarray, dict], dimension=None):
        if role == 'agent':
            self._preprocess_for_agent(data, dimension=dimension)
        elif role == 'host':
            self._preprocess_for_host(data)
        else:
            raise ValueError(f"role can only be either host or agent. Got {role}.")
        self.role = role

    def _preprocess_for_agent(self, data: Union[jnp.ndarray, dict], dimension=None):
        if isinstance(data, jnp.ndarray):
            if dimension is None:
                raise ValueError(f"The data is already a jax numpy array. A dimension argument must be given.")
            assert len(data.shape) == 2
            self.dimension = dimension
            # The array must come from flattened and concatenated arrays with second axis of dimension:
            #   `dimension * max_num_points + dimension`
            assert (data.shape[1] - self.dimension) % self.dimension == 0
            self.max_num_points = (data.shape[1] - self.dimension) // self.dimension
            self.batch_size = data.shape[0]
            self.data = data
        elif isinstance(data, dict):
            # If input is a dict, it must have `points` and `coords` keys.
            self.batch_size, self.max_num_points, self.dimension = data['points'].shape
            self.data = jnp.concatenate([data['points'].reshape(self.batch_size, -1), data['coords']], axis=1)
        else:
            raise TypeError(f"data must be either a jax numpy array or dict. Got {type(data)}.")

    def _preprocess_for_host(self, data: jnp.ndarray):
        if isinstance(data, jnp.ndarray):
            self.batch_size, self.max_num_points, self.dimension = data.shape
            self.data = data.reshape(self.batch_size, -1)
        else:
            raise TypeError(f"data must be a jax numpy array. Got {type(data)}.")

    def get_features(self) -> jnp.ndarray:
        """
        Returns a flattened jnp array ready for neural network training.
        """
        return self.data

    def get_points(self):
        """
        Return the original unflattened points.
        """
        if self.role == 'host':
            return self.data.reshape(self.batch_size, self.max_num_points, self.dimension)
        else:
            return self.data[:, :self.max_num_points * self.dimension]\
                .reshape(self.batch_size, self.max_num_points, self.dimension)

    def get_coords(self):
        """
        Return the coordinates. If the role is host, return None.
        """
        if self.role == 'host':
            return None
        else:
            return self.data[:, self.max_num_points * self.dimension:]


# An experience consists of
#   observation, action, reward, done, next_observation
Experience = Tuple[JAXObs, jnp.ndarray, jnp.ndarray, jnp.ndarray, JAXObs]
