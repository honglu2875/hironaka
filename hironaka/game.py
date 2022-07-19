import abc
import logging
from typing import Optional, Union

from hironaka.core import Points
from hironaka.agent import Agent
from hironaka.host import Host


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

    def _show(self, coords, action, weights, ended):
        self.logger.info(f"Host move: {coords}")
        self.logger.info(f"Agent move: {action}")
        if weights is not None:
            self.logger.info(f"Weights: {weights}")
        self.logger.info(f"Game Ended: {ended}")

    def print_history(self):
        self.logger.info("Coordinate history (host choices):")
        self.logger.info(self.coord_history)
        self.logger.info("Move history (agent choices):")
        self.logger.info(self.move_history)


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
            self._show(coords, action, None, self.state.ended)

        self.coord_history.append(coords)
        self.move_history.append(action)

        if self.state.ended:
            self.stopped = True
            return False
        return True


class GameMorin(Game):
    """The agent is Thom (picks the action coordinate with the smallest weight), but the game terminates with a
    label 'NO CONTRIBUTION' if the distinguished Porteous point is not a vertex of the Newton polytope
    after the shift"""

    def __init__(self,
                 state: Union[Points, None],
                 host: Host,
                 agent: Agent,
                 scale_observation: Optional[bool] = True,
                 **kwargs):
        if self.logger is None:
            self.logger = logging.getLogger(__class__.__name__)

        super().__init__(state, host, agent, **kwargs)
        self.weights = [[1] * self.state.dimension for _ in range(self.state.batch_size)]
        self.scale_observation = scale_observation
        self.stopped = False

    def step(self, verbose: int = 0) -> bool:
        if self.stopped:
            return False

        if verbose:
            self.logger.info(self.state)

        coords = self.host.select_coord(self.state)
        action = self.agent.move(self.state, coords, self.weights, inplace=True)

        if verbose:
            self._show(coords, action, None, self.state.ended)

        self.coord_history.append(coords)
        self.move_history.append(action)

        if self.state.ended or self.state.distinguished_points[0] is None:
            self.stopped = True
            if self.state.distinguished_points[0] is None:
                self.logger.info("No contribution.")
            return False
        return True
