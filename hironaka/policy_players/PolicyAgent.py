from typing import List

from hironaka.abs import Points
from hironaka.agent import Agent
from hironaka.policy.Policy import Policy


class PolicyAgent(Agent):
    def __init__(self, policy: Policy):
        self._policy = policy

    def move(self, points: Points, coords: List[List[int]], inplace=True):
        assert len(
            coords) == points.batch_size  # TODO: wrap the move method for the abstract "Agent" with sanity checks?

        features = points.get_features()

        actions = self._policy.predict((features, coords))

        if inplace:
            points.shift(coords, actions)
            points.get_newton_polytope()

        return actions
