import abc


class Trainer(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def train(self, steps: int):
        pass
