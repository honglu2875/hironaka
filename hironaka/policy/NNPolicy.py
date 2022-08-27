from typing import Any, Optional, Union

import numpy as np
import torch

from hironaka.policy.Policy import Policy
from hironaka.src import batched_coord_list_to_binary, decode_action, mask_encoded_action, HostActionEncoder


class NNPolicy(Policy):
    """
    The basic policy that uses neural network to predict an action.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 masked: Optional[bool] = True,
                 eval_mode: Optional[bool] = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model.to(self.device)
        self.masked = masked
        self.eval_mode = eval_mode

        # Always use compressed output dimension.
        # E.g., in 3d, 0, 1, 2, 3 -> [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
        if self.mode == 'host':
            self.host_encoder = HostActionEncoder(self.dimension)

    def predict(self, features: Any, debug: Optional[bool] = False) -> np.ndarray:
        """
        Predict an action based on a neural network.
        If self.mode=='host', the output should be turned into a batch of multi-binary array,
        If self.mode=='agent', the output should be the argmax (of the whole array or the masked entries only).
        """
        if self.mode == 'host':
            input_tensor = self.input_preprocess_for_host(features)
        elif self.mode == 'agent':
            input_tensor = self.input_preprocess_for_agent(features)
        else:
            assert False  # Impossible code path unless something is broken.

        # Debug message
        if debug:
            self.logger.debug("Input tensor:")
            self.logger.debug(input_tensor)

        if self.eval_mode:
            self.model.eval()
            with torch.inference_mode(mode=True):
                output_tensor = self._evaluate(input_tensor)
        else:
            output_tensor = self._evaluate(input_tensor)

        # Debug message
        if debug:
            self.logger.debug("Output tensor:")
            self.logger.debug(output_tensor)

        if self.mode == 'agent':
            output_tensor = torch.softmax(output_tensor, dim=1)
            if self.masked:
                mask = torch.tensor(batched_coord_list_to_binary(features[1], self.dimension), device=self.device)
                output_tensor = output_tensor * mask
            return torch.argmax(output_tensor, dim=1).detach().numpy()
        elif self.mode == 'host':
            # action probabilities -> decode into multi-binary.
            # E.g., output [[0.5, 0.7, 0.5, 0.3]] --argmax--> [[1]] --decode_tensor--> [[1, 0, 1]]
            output_tensor = torch.softmax(output_tensor, dim=1)
            return self.host_encoder.decode_tensor(torch.argmax(output_tensor, dim=1)).detach().numpy()

    def _evaluate(self, input_tensor):
        """
        This method evaluates the input_tensor using self._model.
        For certain policies (subclasses of nn.Module), getting the probability tensor may not be like
            `self._model(input_tensor)`
        In these cases, please override this method.
        """
        return self.model(input_tensor)
