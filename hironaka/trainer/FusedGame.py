from typing import Optional, Tuple, Union

import numpy as np
import torch

from hironaka.core import PointsBase, TensorPoints
from hironaka.src import mask_encoded_action, mask_encoded_action_torch
from hironaka.trainer.Timer import Timer


class FusedGame:
    """
        A fused game class processing a large batch of games. It avoids all the wrappers (especially gym) other than
        `TensorPoints`. Aim to maximize speed and simplify training/validation.

        """

    def __init__(self,
                 host_net: torch.nn.Module,
                 agent_net: torch.nn.Module,
                 device_key: Optional[str] = 'cpu',
                 log_time: Optional[bool] = True):
        """
            host_net: a nn.Module where
                input: a 3-dim tensor representing a batch of points.
                    Negative numbers are regarded as padding for removed points.
                output: a 2-dim tensor consisting of the logits of the probability of choosing each coordinate.
            agent_net: a nn.Module where
                input: a dict
                    "points": 3-dim tensor of points.
                    "coords": 2-dim tensor of chosen coordinates. (Is not forced to only take values 0/1.)
        """
        self.device = torch.device(device_key)
        self.use_cuda = device_key != 'cpu'
        self.host_net = host_net.to(self.device)
        self.agent_net = agent_net.to(self.device)
        self.log_time = log_time

        # The following are used as cache inside host action decoding. Sometimes improves performances.
        self.binary_table = None
        self.mask = None

        self.time_log = dict()

    def step(self,
             points: TensorPoints,
             sample_for: str,
             masked=True,
             scale_observation=True,
             exploration_rate=0.2):
        """
            Progress the game and return:
                observations, actions (depending on sample_for), rewards, dones, next_observations
        """
        assert sample_for in ["host", "agent"], f"sample_for must be one of 'host' and 'agent'. Got {sample_for}."

        with Timer(f'step-get_features_total', self.time_log, active=self.log_time, use_cuda=self.use_cuda):
            observations = points.get_features()
        done = points.ended_batch_in_tensor

        # Set the exploration rate. The counter-party should not explore.
        e_r = exploration_rate if sample_for == "host" else 0.0
        with Timer(f'step-host_move', self.time_log, active=self.log_time, use_cuda=self.use_cuda):
            host_move, chosen_actions = self.host_move(points, masked=masked, exploration_rate=e_r)
        e_r = exploration_rate if sample_for == "agent" else 0.0
        with Timer(f'step-agent_move', self.time_log, active=self.log_time, use_cuda=self.use_cuda):
            agent_move = self.agent_move(points, host_move, masked=masked,
                                         scale_observation=scale_observation, inplace=True, exploration_rate=e_r)

        next_done = points.ended_batch_in_tensor
        with Timer(f'step-get_features_total', self.time_log, active=self.log_time, use_cuda=self.use_cuda):
            next_observations = points.get_features()

        # Filter out already-finished states and obtain experiences
        if sample_for == "host":
            with Timer(f'step-host_postprocess_exps', self.time_log, active=self.log_time, use_cuda=self.use_cuda):
                output_obs = observations[~done].clone()
                output_actions = chosen_actions[~done].clone()
                next_observations = next_observations[~done].clone()
        else:
            with Timer(f'step-agent_extra_host_move', self.time_log, active=self.log_time, use_cuda=self.use_cuda):
                next_host_move, _ = self.host_move(points, masked=masked, exploration_rate=exploration_rate)
            with Timer(f'step-agent_postprocess_exps', self.time_log, active=self.log_time, use_cuda=self.use_cuda):
                output_obs = {"points": observations[~done].clone(),
                              "coords": host_move[~done].clone()}
                output_actions = agent_move[~done].clone()
                next_observations = {"points": next_observations[~done].clone(),
                                     "coords": next_host_move[~done].clone()}
        next_done = next_done[~done].clone()

        return (output_obs,
                output_actions.reshape(-1, 1),
                self._rewards(sample_for, output_obs, next_observations, next_done).reshape(-1, 1),
                next_done.reshape(-1, 1),
                next_observations)

    def host_move(self, points: TensorPoints, masked=True, exploration_rate=0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with Timer(f'host_move-host_net_inference', self.time_log, active=self.log_time, use_cuda=self.use_cuda):
            with torch.inference_mode():
                output = self.host_net(points.get_features())

        _TYPE = output.dtype

        noise = torch.rand(output.shape, device=self.device, dtype=_TYPE)
        random_mask = torch.rand(output.shape[0], 1, device=self.device).le(exploration_rate)
        output = output * ~random_mask + noise * random_mask
        with Timer(f'host_move-decode_tensor', self.time_log, active=self.log_time, use_cuda=self.use_cuda):
            # The random noises still have to go through decode_tensor, therefore are still never illegal moves.
            host_move_binary, chosen_actions = self.decode_tensor(output, masked=masked)

        return host_move_binary, chosen_actions

    def agent_move(self, points: PointsBase,
                   host_moves: torch.Tensor,
                   masked: Optional[bool] = True,
                   scale_observation: Optional[bool] = True,
                   inplace: Optional[bool] = True,
                   exploration_rate: Optional[float] = 0.0) -> torch.Tensor:
        with Timer(f'agent_move-agent_net_inference', self.time_log, active=self.log_time, use_cuda=self.use_cuda):
            with torch.inference_mode():
                action_prob = self.agent_net(
                    {"points": points.get_features(), "coords": host_moves})

        if masked:
            minimum = torch.finfo(action_prob.dtype).min
            masked_value = (1 - host_moves) * minimum
            action_prob = action_prob * host_moves + masked_value

        actions = torch.argmax(action_prob, dim=1)

        noise = torch.randint(0, action_prob.shape[1], actions.shape, device=actions.device, dtype=actions.dtype)
        random_mask = torch.rand(actions.shape[0], device=actions.device).le(exploration_rate)
        actions = actions * ~random_mask + noise * random_mask

        with Timer(f'agent_move-point_operations', self.time_log, active=self.log_time, use_cuda=self.use_cuda):
            _TYPE = points.points.dtype
            if inplace:
                with Timer(f'agent_move-pt_ops_shift', self.time_log, active=self.log_time, use_cuda=self.use_cuda):
                    points.shift(host_moves.type(_TYPE), actions.type(_TYPE))
                with Timer(f'agent_move-pt_ops_get_newton_polytope', self.time_log, active=self.log_time, use_cuda=self.use_cuda):
                    points.get_newton_polytope()
                with Timer(f'agent_move-pt_ops_rescale', self.time_log, active=self.log_time, use_cuda=self.use_cuda):
                    if scale_observation:
                        points.rescale()
        return actions

    @staticmethod
    def _rewards(sample_for: str,
                 obs: Union[torch.Tensor, dict],
                 next_obs: Union[torch.Tensor, dict],
                 next_done: torch.Tensor) -> torch.Tensor:
        if sample_for == "host":
            return next_done.clone().type(torch.float32)
        elif sample_for == "agent":
            return (~next_done).clone().type(torch.float32)

    def decode_tensor(self, t, masked=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Decode a batch of encoded host actions into a batch of multi-binary (0 or 1) arrays.
            Parameters:
                t: a 2-dim tensor. Each row is a list of logits for each number from 0 to 2**dim-1.
                masked: filter out choices with less than 2 coordinates (excluding 0, 1, 2, 4, 8, ...).
        """
        device = t.device
        minimum = torch.finfo(t.dtype).min

        dim = int(np.log2(t.shape[1]))
        assert 2 ** dim == t.shape[1], f"The length of the 2nd axis must be length a power of 2. Got {t.shape} instead."

        if masked:
            if self.mask is None:
                self.mask = mask_encoded_action_torch(dim, device=device).unsqueeze(0)
            mask = self.mask
        else:
            mask = torch.ones((1, 2 ** dim), device=device, dtype=torch.float32)

        masked_values = ((1 - mask) * minimum).repeat(t.shape[0], 1)

        if self.binary_table is None:
            self.binary_table = torch.zeros((2 ** dim, dim), device=device)
            # They involve a lot of small CPU ops which may cause undesirable effects if repeated too many times,
            #   so we make sure it only runs once per object life-span.
            # But there might be a smarter vectorized way to generate these.
            for i in range(2 ** dim):
                b = bin(i)[2:]
                for j in range(len(b) - 1, -1, -1):
                    if b[j] == '1':
                        self.binary_table[i][len(b) - 1 - j] = 1
        chosen_actions = torch.argmax(t * mask + masked_values, dim=1)
        return self.binary_table[chosen_actions], chosen_actions
