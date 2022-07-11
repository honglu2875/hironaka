from typing import List, Tuple

import numpy as np

from hironaka.agentThom import TAgent
from hironaka.host import Host


class GameThom:

    state: List[Tuple[int]]
    weights: List[int]
    coordHistory: List[int]
    moveHistory: List[int]
    host: Host
    agent: TAgent

    def __init__(self, points, host, agent):
        self.state = points
        self.host = host
        self.agent = agent
        self.coordHistory = []
        self.moveHistory = []
        self.weights = np.ones(len(points[0]))
        self.stopped = False

    def step(self):
        if self.stopped:
            return
        coords = self.host.selectCoord(self.state)
        newState, action, newweights = self.agent.move(self.state, self.weights, coords)

        self.state = newState
        self.weights = newweights
        self.coordHistory.append(coords)
        self.moveHistory.append(action)

        if len(self.state) == 1:
            self.stopped = True


class GameMorin:

    """The agent is Thom (picks the action coordinate with smallest weight), but the game terminates with a
    label 'NO CONTRIBUTION' if the distinguished Porteous point is not a vertex of the Newton polytope
    after the shift"""

    state: List[Tuple[int]]
    weights: List[int]
    coordHistory: List[int]
    moveHistory: List[int]
    host: Host
    agent: TAgent

    def __init__(self, points, host, agent):
        self.state = points
        self.host = host
        self.agent = agent
        self.coordHistory = []
        self.moveHistory = []
        self.weights = np.ones(len(points[0]))
        self.stopped = False

    def step(self):
        if self.stopped:
            return
        coords = self.host.selectCoord(self.state)
        newData = self.agent.move(self.state, self.weights, coords)
        #print('Hello', coords, self.agent.move(self.state, self.weights, coords))
        if newData == False:
            print('No contribution')
            self.stopped = True
        else:
            newState, action, newweights = newData
            self.state = newState
            self.weights = newweights
            self.coordHistory.append(coords)
            self.moveHistory.append(action)
            if len(self.state) == 1:
                self.stopped = True