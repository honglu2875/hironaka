from typing import Optional

import numpy as np

from hironaka.core import Points
from hironaka.host import Host
from hironaka.policy.Policy import Policy


class PolicyHost(Host):
    def __init__(self,
                 policy: Policy,
                 use_discrete_actions_for_host: Optional[bool] = False,
                 **kwargs):
        self._policy = policy
        self.use_discrete_actions_for_host = kwargs.get('use_discrete_actions_for_host', use_discrete_actions_for_host)

    def select_coord(self, points: Points, debug=False):
        features = points.get_features()

        coords = self._policy.predict(features)  # return multi-binary array
        result = []
        for b in range(coords.shape[0]):
            result.append(np.where(coords[b] == 1)[0])
        return result
