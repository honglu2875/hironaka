from typing import List, Tuple

import numpy as np

from hironaka.abs import Points
from hironaka.agent import Agent
from hironaka.agentThom import TAgent
from hironaka.host import Host


class GameThom:  # QUESTION: Usage???
    def __init__(self, points: Points, host: Host, agent: TAgent):
        self.state = points
        self.host = host
        self.agent = agent
        self.coord_history = []
        self.move_history = []
        self.weights = [[1] * points.dimension for _ in range(self.state.batch_size)]
        self.stopped = False

    def step(self):
        if self.stopped:
            return
        coords = self.host.select_coord(self.state)
        new_state, action, new_weights = self.agent.move(self.state, self.weights, coords)

        self.state = new_state
        self.weights = new_weights
        self.coord_history.append(coords)
        self.move_history.append(action)

        if len(self.state) == 1:
            self.stopped = True


class GameMorin:

    """The agent is Thom (picks the action coordinate with smallest weight), but the game terminates with a
    label 'NO CONTRIBUTION' if the distinguished Porteous point is not a vertex of the Newton polytope
    after the shift"""

    def __init__(self, points: Points, host: Host, agent: TAgent):
        self.state = points
        self.host = host
        self.agent = agent
        self.coord_history = []
        self.move_history = []
        self.weights = [[1] * points.dimension for _ in range(self.state.batch_size)]
        self.stopped = False

    def step(self):
        if self.stopped:
            return
        coords = self.host.select_coord(self.state)
        new_data = self.agent.move(self.state, self.weights, coords)
        if not new_data:
            print('No contribution')
            self.stopped = True
        else:
            newState, action, new_weights = new_data
            self.state = newState
            self.weights = new_weights
            self.coord_history.append(coords)
            self.move_history.append(action)
            if self.state.ended:
                self.stopped = True
