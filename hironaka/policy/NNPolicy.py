from typing import Dict, Any

import torch

from hironaka.policy.Policy import Policy
from hironaka.src import batched_coord_list_to_binary


class NNPolicy(Policy):
    """
        The basic policy that uses neural network to predict an action.
    """

    def __init__(self, model, use_cuda=False, masked=True, eval_mode=False, config_kwargs: Dict[str, Any] = None,
                 **kwargs):
        config_kwargs = dict() if config_kwargs is None else config_kwargs
        super().__init__(**{**config_kwargs, **kwargs})

        self._model = model
        if use_cuda or (config_kwargs.get('use_cuda') is True):
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        self.masked = masked
        self.eval_mode = eval_mode

    def predict(self, features: Any) -> Any:
        """
            predict an action based on a neural network.
            The input varies depending on whether it is a host or an agent.
            But the output tensors (directly from the network) are the same: float of shape (self.dimension,).
            If it is a host, the output should be turned into a binary array.
            If it is an agent, the output should be the argmax (of the whole array or the masked entries only).
        """
        if self.mode == 'host':
            input_tensor = self.input_preprocess_for_host(features)
        elif self.mode == 'agent':
            input_tensor = self.input_preprocess_for_agent(features)
        else:
            assert False  # Impossible code path unless something is broken.

        input_tensor.to(self._device)
        self._model.to(self._device)
        if self.eval_mode:
            self._model.eval()
            with torch.inference_mode(mode=True):
                output_tensor = self._model(input_tensor)
        else:
            output_tensor = self._model(input_tensor)

        if self.mode == 'agent':
            output_tensor = torch.softmax(output_tensor, dim=1)
            if self.masked:
                mask = torch.FloatTensor(batched_coord_list_to_binary(features[1], self.dimension))
                output_tensor = output_tensor * mask
            return torch.argmax(output_tensor, dim=1).detach().numpy()
        elif self.mode == 'host':
            output_tensor = torch.sigmoid(output_tensor)
            out = torch.gt(output_tensor, 0.5)
            if self.masked:
                for b in range(out.shape[0]):
                    if torch.sum(out[b]) < 2:
                        t = torch.topk(output_tensor[b], k=2)[1]
                        out[b][t[0]], out[b][t[1]] = True, True
            return out.detach().numpy().astype(int)
