from .host import Host
from .agent import Agent
import abc
from typing import List, Tuple
import numpy as np


class Game(metaclass=abc.ABCMeta):
    """
        This framework simulates a fully autonomous game without interference from outside.
    """
    @abc.abstractmethod
    def __init__(self, initState, host, agent):
        """
            initState: initial state
            host: the host player
            agent: the agent player
        """
        pass

    @abc.abstractmethod
    def step(self) -> bool:
        """
            Make the game one step forward.

            Return: True if successful, False if unsuccessful

            Remark:
                The two agents should proceed according to their own policy based on the current observation. Therefore, no input is given.
                Even in the case of MCTS, the corresponding agent should have specific codes to handle MCTS search, and then come up with an action.
        """
        pass

    def printHistory(self):
        print("Coordinate history (host choices):")
        print(self.coordHistory)
        print("Move history (agent choices):")
        print(self.moveHistory)
