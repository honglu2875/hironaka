import abc
import logging
import random
from typing import List, Optional, Union

import numpy as np

from hironaka.core import ListPoints
from hironaka.points import Points
from hironaka.policy.policy import Policy


class Agent(abc.ABC):
    """
    An agent can either modify the points in-place, or just return the action (the chosen coordinate)
    Must implement:
        _get_actions
    """

    # Please set the following. They are *class constants*!
    USE_WEIGHTS: bool
    USE_REPOSITION: bool  # apply a self.points.reposition() between shift() and get_newton_polytope()
    must_implement = ["USE_WEIGHTS", "USE_REPOSITION"]

    def __init__(self, ignore_batch_dimension: bool = False, **kwargs):
        self.logger = logging.getLogger(__class__.__name__)

        # If the agent only has one batch and wants to ignore batch dimension in the parameters, set it to True.
        self.ignore_batch_dimension = ignore_batch_dimension

        for s in self.must_implement:
            if not hasattr(self, s) or getattr(self, s) is None:
                raise NotImplementedError(f"Please specify {s} for the subclass.")

    def move(self, points: Union[ListPoints, Points], coords: List, weights: Optional[List] = None, inplace: bool = True):
        """
        Make moves on (the set of) points.
        Parameters:
            points: either `ListPoints` (a batch of different collection of points) or `Points` (a simple wrapper with
                one single collection of points).
            coords: host's choice(-s) of coordinates. Not including the batch axis if points is a `Points` object.
            weights: a list of weights (only for those weight-related games).
            inplace: whether the modification happens directly on mutable objects `points`, `coords` and `weights`.
        """
        if self.USE_WEIGHTS and weights is None:
            raise Exception("Please specify weights in the parameters.")
        if not self.USE_WEIGHTS and weights is not None:
            self.logger.warning("The weights parameter will be ignored.")
        if isinstance(points, Points):
            # `Points` is a wrapper of one single collection of points (i.e., one single game).
            # Here we turn it into a batch with batch size 1.
            points = points.points

        if not self.ignore_batch_dimension:
            assert points.batch_size == len(coords)

        batch_coords = [coords] if self.ignore_batch_dimension else coords
        batch_weights = [weights] if self.ignore_batch_dimension else weights

        batch_actions = self._get_actions(points, batch_coords, batch_weights)
        new_weights = self._get_weights(points, batch_coords, batch_weights, batch_actions)

        actions = batch_actions[0] if self.ignore_batch_dimension else batch_actions
        new_weights = new_weights[0] if self.ignore_batch_dimension else new_weights

        if not inplace:
            return actions if not self.USE_WEIGHTS else (actions, new_weights)

        points.shift(batch_coords, batch_actions)
        if self.USE_REPOSITION:
            points.reposition()
        points.get_newton_polytope()
        if self.USE_WEIGHTS:
            weights[:] = new_weights  # pytype: disable=unsupported-operands
        return actions

    @abc.abstractmethod
    def _get_actions(self, points, batch_coords, batch_weights):
        pass

    def _get_weights(self, points, batch_coords, batch_weights, batch_actions):
        return batch_weights


class RandomAgent(Agent):
    USE_WEIGHTS = False
    USE_REPOSITION = False

    def _get_actions(self, points, batch_coords, batch_weights):
        return [random.choice(coord) if len(coord) > 1 else None for coord in batch_coords]


class ChooseFirstAgent(Agent):
    USE_WEIGHTS = False
    USE_REPOSITION = False

    def _get_actions(self, points, batch_coords, batch_weights):
        return [min(coord) if len(coord) > 1 else None for coord in batch_coords]


class PolicyAgent(Agent):
    USE_WEIGHTS = False
    USE_REPOSITION = False

    def __init__(self, policy: Policy, **kwargs):
        self._policy = policy
        super().__init__(**kwargs)

    def _get_actions(self, points, batch_coords, batch_weights):
        features = points.get_features()
        return self._policy.predict((features, batch_coords)).tolist()


class AgentMorin(Agent):
    USE_WEIGHTS = True
    USE_REPOSITION = True

    def _get_actions(self, points, batch_coords, batch_weights):
        assert points.batch_size == 1, "Currently only support batch size 1."  # TODO: generalize!

        weights = batch_weights[0]
        coords = batch_coords[0]
        if weights[coords[0]] == weights[coords[1]]:
            action = np.random.choice(coords, size=1)[0]
        else:
            action = coords[np.argmin([weights[coords[0]], weights[coords[1]]])]
        return [action]

    def _get_weights(self, points, batch_coords, batch_weights, batch_actions):
        coords = batch_coords[0]
        weights = batch_weights[0]
        action = batch_actions[0]

        changing_coordinate = [coord for coord in coords if coord != action]
        next_weights = [weights[i] if i not in changing_coordinate else 0 for i in range(len(weights))]
        return [next_weights]
