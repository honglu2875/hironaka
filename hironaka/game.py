from .host import Host
from .agent import Agent, TAgent
from typing import List, Tuple
import numpy as np


class Game:
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
