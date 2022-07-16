import numpy as np

from hironaka.core import Points
from hironaka.host import Host
from hironaka.policy.Policy import Policy


class PolicyHost(Host):
    def __init__(self, policy: Policy):
        self._policy = policy

    def select_coord(self, points: Points, debug=False):
        features = points.get_features()

        coords = self._policy.predict(features)
        result = []
        for b in range(coords.shape[0]):
            result.append(np.where(coords[b] == 1)[0])
        return result
