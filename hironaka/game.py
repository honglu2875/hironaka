import abc
import logging
from typing import Optional


class Game(abc.ABC):
    """
        This framework simulates a fully autonomous game without interference from outside.
    """
    logger = None

    @abc.abstractmethod
    def __init__(self,
                 state,
                 host,
                 agent,
                 **kwargs):
        """
            state: initial state
            host: the host player
            agent: the agent player
        """
        if self.logger is None:
            self.logger = logging.getLogger(__class__.__name__)

        self.state = state
        self.dimension = state.dimension if state is not None else None
        self.host = host
        self.agent = agent

        self.coord_history = []
        self.move_history = []

    @abc.abstractmethod
    def step(self, verbose: int = 0) -> bool:
        """
            Make the game one step forward.

            Return: True if successful, False if unsuccessful

            Remark:
                The two agents should proceed according to their own policy based on the current observation.
                Therefore, no input is given.
                Even in the case of MCTS, the corresponding agent should have specific codes to handle MCTS search,
                and then come up with an action.
        """
        pass

    def print_history(self):
        self.logger.info("Coordinate history (host choices):")
        self.logger.info(self.coord_history)
        self.logger.info("Move history (agent choices):")
        self.logger.info(self.move_history)
