import logging
from typing import Optional, Union

from .core import Points
from .agent import Agent
from .game import Game
from .host import Host


class GameHironaka(Game):
    def __init__(self,
                 state: Union[Points, None],
                 host: Host,
                 agent: Agent,
                 scale_observation: Optional[bool] = True,
                 **kwargs):
        if self.logger is None:
            self.logger = logging.getLogger(__class__.__name__)

        super().__init__(state, host, agent, **kwargs)
        self.scale_observation = scale_observation
        self.stopped = False

    def step(self, verbose: int = 0) -> bool:
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

        if verbose:
            self.logger.info(self.state)

        coords = self.host.select_coord(self.state)
        action = self.agent.move(self.state, coords)
        if self.scale_observation:
            self.state.rescale()

        if verbose:
            self.logger.info(f"Host move: {coords}")
            self.logger.info(f"Agent move: {action}")
            self.logger.info(f"Game Ended: {self.state.ended}")

        self.coord_history.append(coords)
        self.move_history.append(action)

        if self.state.ended:
            self.stopped = True
            return False
        return True


