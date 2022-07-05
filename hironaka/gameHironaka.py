import logging

from .abs import Points
from .game import Game


class GameHironaka(Game):
    def __init__(self, state: Points, host, agent):
        if self.logger is None:
            self.logger = logging.getLogger(__class__.__name__)

        super().__init__(state, host, agent)
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


