import logging
from typing import Dict, Any, Optional

import numpy as np
import torch

from hironaka.policy.Policy import Policy
from hironaka.src import batched_coord_list_to_binary, decode_action, mask_encoded_action


class NNPolicy(Policy):
    """
        The basic policy that uses neural network to predict an action.
    """
    logger = None

    def __init__(self,
                 model,
                 use_cuda: Optional[bool] = False,
                 masked: Optional[bool] = True,
                 eval_mode: Optional[bool] = False,
                 use_discrete_actions_for_host: Optional[bool] = False,
                 config_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs):
        if self.logger is None:
            self.logger = logging.getLogger(__class__.__name__)

        config = kwargs if config_kwargs is None else {**kwargs, **config_kwargs}
        super().__init__(**config)

        self._model = model
        if use_cuda or config.get('use_cuda'):
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')

        self.masked = config.get('masked', masked)
        self.use_discrete_actions_for_host = config.get('use_discrete_actions_for_host', use_discrete_actions_for_host)
        self.discrete_host_mask = torch.FloatTensor(mask_encoded_action(self.dimension)).to(self._device) \
            if self.masked and self.use_discrete_actions_for_host else None
        self.eval_mode = config.get('eval_mode', eval_mode)

    def predict(self, features: Any, debug: Optional[bool] = False) -> Any:
        """
            predict an action based on a neural network.
            The input varies depending on whether it is a host or an agent.
            But the output tensors (directly from the network) are the same: float of shape
                (batch_size, self.dimension).
            If it is a host, the output should be turned into batches of binary array,
            If it is an agent, the output should be the argmax (of the whole array or the masked entries only).
        """
        if self.mode == 'host':
            input_tensor = self.input_preprocess_for_host(features)
        elif self.mode == 'agent':
            input_tensor = self.input_preprocess_for_agent(features)
        else:
            assert False  # Impossible code path unless something is broken.

        input_tensor = input_tensor.to(self._device)

        # Debug message
        if debug:
            self.logger.debug("Input tensor:")
            self.logger.debug(input_tensor)

        self._model.to(self._device)

        if self.eval_mode:
            self._model.eval()
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
                mask = torch.FloatTensor(batched_coord_list_to_binary(features[1], self.dimension)).to(self._device)
                output_tensor = output_tensor * mask
            return torch.argmax(output_tensor, dim=1).detach().cpu().numpy()
        elif self.mode == 'host':
            # discrete action -> one-hot encoding.
            if self.use_discrete_actions_for_host:
                output_tensor = torch.softmax(output_tensor, dim=1)
                if self.masked:
                    output_tensor *= self.discrete_host_mask

                encoded_actions = torch.argmax(output_tensor, dim=1).detach().cpu().numpy()
                # TODO: Slow. Could be improved if necessary.
                return np.array(
                    [decode_action(encoded_actions[b], self.dimension)
                     for b in range(encoded_actions.shape[0])])

            # multi-binary action -> apply sigmoid on each coordinate.
            output_tensor = torch.sigmoid(output_tensor)
            out = torch.gt(output_tensor, 0.5)
            if self.masked:
                for b in range(out.shape[0]):
                    if torch.sum(out[b]) < 2:
                        t = torch.topk(output_tensor[b], k=2)[1]
                        out[b][t[0]], out[b][t[1]] = True, True
            return out.detach().cpu().numpy().astype(int)

    def _evaluate(self, input_tensor):
        """
            This method evaluates the input_tensor using self._model.
            For certain policies (subclasses of nn.Module), getting the probability tensor may not be like
                `self._model(input_tensor)`
            In these cases, please override this method.
        """
        return self._model(input_tensor)
