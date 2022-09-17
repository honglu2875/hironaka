import abc
from typing import Union


class Scheduler(abc.ABC):
    """
    Scheduler for learning rate, exploration rate, etc.
    How it works:
        - It records a value at initialization (`value`).
        - It is fed with an indefinite amount of parameters in `kwargs`. Each subclass chooses what to use.
        - get_value() takes a parameter called steps, and returns a new value.
    It is basically a function. But the minor extra features are:
        - One can optionally record some parameters and states
        - It enforces `mandatory_keys` when fed with parameters at initialization.
    """

    key_name = None
    mandatory_keys = []

    def __init__(self, value: Union[float, int], **kwargs):
        self.value = value
        self.kwargs = kwargs

        for key in self.mandatory_keys:
            assert (key in self.kwargs), \
                f"'{key}' must be in {self.key_name if self.key_name is not None else 'the config'}. Got {self.kwargs}."

    def __call__(self, steps: int, **kwargs):
        return self.get_value(steps)

    @abc.abstractmethod
    def get_value(self, steps: int) -> Union[float, int]:
        pass


class ConstantScheduler(Scheduler):
    def get_value(self, steps: int) -> Union[float, int]:
        return self.value


class ExponentialLRScheduler(Scheduler):
    key_name = "lr_schedule"
    mandatory_keys = ["initial_lr", "rate"]

    def get_value(self, steps: int) -> Union[float, int]:
        initial_lr = self.kwargs["initial_lr"]
        rate = self.kwargs["rate"]
        return initial_lr * (rate ** steps) + self.value * (1 - rate ** steps)


class InverseLRScheduler(Scheduler):
    key_name = "lr_schedule"
    mandatory_keys = ["initial_lr", "rate"]

    def get_value(self, steps: int) -> Union[float, int]:
        initial_lr = self.kwargs["initial_lr"]
        rate = self.kwargs["rate"]
        return rate / (steps + rate / initial_lr)


class ExponentialERScheduler(Scheduler):
    key_name = "er_schedule"
    mandatory_keys = ["initial_er", "rate"]

    def get_value(self, steps: int) -> Union[float, int]:
        initial_er = self.kwargs["initial_er"]
        rate = self.kwargs["rate"]
        return initial_er * (rate ** steps) + self.value * (1 - rate ** steps)
