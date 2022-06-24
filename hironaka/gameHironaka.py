from .game import Game
from .types import Points


class GameHironaka(Game):
    def __init__(self, state: Points, host, agent):
        self.state = state
        self.dim = state.dim
        self.host = host
        self.agent = agent
        self.coord_history = []
        self.move_history = []
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
        coords = self.host.select_coord(self.state)
        action = self.agent.move(self.state, coords)

        self.coord_history.append(coords)
        self.move_history.append(action)

        if self.state.ended:
            self.stopped = True
        return True

    def get_features(self):
        return self.state.get_sym_features()
