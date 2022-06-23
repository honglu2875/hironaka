from .host import Host
from .game import Game
from .agent import Agent
from typing import List, Tuple
from .types import Points
import numpy as np


class GameHironaka(Game):
    state: List[Tuple[int]]
    coordHistory: List[int]
    moveHistory: List[int]
    host: Host
    agent: Agent

    def __init__(self, initState: Points, host, agent):
        self.state = initState
        self.dim = initState.dim
        self.host = host
        self.agent = agent
        self.coordHistory = []
        self.moveHistory = []
        self.stopped = False

    def step(self) -> bool:
        """
            Make one move forward.

            In particular,
            1. the host selects coordinates
                (use: Host.selectCoord(state));
            2. the agent makes one move according to the selected coordinates
                (use: Agent.move(state, coords)).

            Return: True if successful, False if unsuccessful
        """
        if self.stopped:
            return False
        coords = self.host.selectCoord(self.state)
        action = self.agent.move(self.state, coords)

        self.coordHistory.append(coords)
        self.moveHistory.append(action)

        if self.state.ended:
            self.stopped = True
        return True

    def getFeatures(self, length):
        """
            Say the points are ((x_1)_1, ...,(x_1)_n), ...,((x_k)_1, ...,(x_k)_n)
            We generate the Newton polynomials of each coordinates and output the new array as feature.
            The output becomes ((\sum_i (x_i)_1^1), ..., (\sum_i (x_i)_n^1)), ..., ((\sum_i (x_i)_1^length), ..., (\sum_i (x_i)_n^length))
        """
        return np.array(
            [
                [
                    sum([
                        x[i] ^ j for x in self.state
                    ]) for i in range(self.dim)
                ] for j in range(1, length+1)
            ]
        )
