from .host import Host
from .agent import Agent
from typing import List, Tuple


class GameThom:
    state: List[Tuple[int]]
    coordHistory: List[int]
    moveHistory: List[int]
    host: Host
    agent: Agent

    def __init__(self, points, host, agent):
        self.state = points
        self.host = host
        self.agent = agent
        self.coordHistory = []
        self.moveHistory = []
        self.stopped = False

    def step(self):
        if self.stopped:
            return
        coords = self.host.selectCoord(self.state)
        newState, action = self.agent.move(self.state, coords)

        self.state = newState
        self.coordHistory.append(coords)
        self.moveHistory.append(action)

        if len(self.state) == 1:
            self.stopped = True
