import abc


class Game(abc.ABC):
    """
        This framework simulates a fully autonomous game without interference from outside.
    """

    @abc.abstractmethod
    def __init__(self, state, host, agent):
        """
            state: initial state
            host: the host player
            agent: the agent player
        """
        self.state = state
        self.dim = state.dim
        self.host = host
        self.agent = agent

        self.coord_history = []
        self.move_history = []

    @abc.abstractmethod
    def step(self) -> bool:
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
        print("Coordinate history (host choices):")
        print(self.coord_history)
        print("Move history (agent choices):")
        print(self.move_history)
