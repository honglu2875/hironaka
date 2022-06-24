from typing import List, Tuple

from .agent import Agent
from .host import Host


class GameThom:
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
        coords = self.host.select_coord(self.state)
        new_state, action = self.agent.move(self.state, coords)

        self.state = new_state
        self.coordHistory.append(coords)
        self.moveHistory.append(action)

        if len(self.state) == 1:
            self.stopped = True
