from typing import Optional, Dict, Any
import logging

from hironaka.core import Points
from hironaka.gameHironaka import GameHironaka
from hironaka.util import generate_batch_points


class HironakaValidator(GameHironaka):
    """
        Given an agent and a host, this class inherits GameHironaka and handles the validation process.
        It behaves like
    """

    def __init__(self,
                 host,
                 agent,
                 config_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs):
        if self.logger is None:
            self.logger = logging.getLogger(__class__.__name__)

        config_kwargs = dict() if config_kwargs is None else config_kwargs
        config = {**config_kwargs, **kwargs}

        self.points_config = {
            'n': config.get('max_number_points', 10),
            'batch_num': 1,
            'dimension': config.get('dimension', 3),
            'max_value': config.get('max_value', 50)
        }
        value_threshold = config.get('value_threshold', None)
        step_threshold = config.get('step_threshold', 1000)

        self.value_threshold = value_threshold
        self.step_threshold = step_threshold

        super().__init__(None, host, agent, **config)
        self.reset()

    def playoff(self, num_steps: int, verbose: int = 0):
        if self.stopped:
            self.reset()
        len_history = []
        len_counter = 0
        for _ in range(num_steps):
            exceed_threshold = False
            if self.value_threshold is not None:
                exceed_threshold = self.state.exceed_threshold()
            if self.step(verbose=verbose) and len_counter < self.step_threshold and not exceed_threshold:
                len_counter += 1
            else:
                len_history.append(len_counter)
                len_counter = 0
                self.reset()

        len_history.append(len_counter)
        self.reset()
        return len_history

    def reset(self):
        self.state = Points(generate_batch_points(**self.points_config), value_threshold=self.value_threshold)
        if self.scale_observation:
            self.state.rescale()
        self.stopped = False
