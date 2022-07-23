from typing import Optional, Tuple

import numpy as np
import torch

from hironaka.core import PointsBase, TensorPoints
from hironaka.src import mask_encoded_action


class FusedGame:
    """
        A fused game class processing a large batch of games. It avoids all the wrappers other than `TensorPoints`.
        Aim to maximize speed and simplify training/validation.

        """
    _TYPE = torch.float32

    def __init__(self,
                 host_net: torch.nn.Module,
                 agent_net: torch.nn.Module,
                 device_key: Optional[str] = 'cpu'):
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
        self.host_net = host_net.to(self.device)
        self.agent_net = agent_net.to(self.device)

    def step(self,
             points: TensorPoints,
             sample_for: str,
             masked=True,
             scale_observation=True,
             exploration_rate=0.2):
        """
            Progress the game and return:
                observations, actions (depending on sample_for), rewards, next_observations
        """
        assert sample_for in ["host", "agent"], f"sample_for must be one of 'host' and 'agent'. Got {sample_for}."

        observations = points.get_features().cpu().detach().numpy().copy()
        done = np.array(points.ended_batch)

        e_r = exploration_rate if sample_for == "host" else 0.0
        host_move, chosen_actions = self._host_move(points, masked=masked, exploration_rate=e_r)
        e_r = exploration_rate if sample_for == "agent" else 0.0
        agent_move = self._agent_move(points, host_move, masked=masked,
                                      scale_observation=scale_observation, inplace=True, exploration_rate=e_r)

        next_done = np.array(points.ended_batch)
        next_observations = points.get_features().cpu().detach().numpy().copy()

        if sample_for == "host":
            output_obs = observations[~done].copy()
            output_actions = chosen_actions.cpu().detach().numpy()
        else:
            output_obs = {"points": observations[~done].copy(),
                          "coords": host_move.cpu().detach().numpy()[~done].copy()}
            output_actions = agent_move.cpu().detach().numpy()

        return output_obs, \
               output_actions[~done].copy(), \
               self._reward(sample_for, observations[~done], next_observations[~done], next_done[~done]).copy(), \
               next_observations[~done].copy()

    def _host_move(self, points: TensorPoints, masked=True, exploration_rate=0.0) -> Tuple[torch.Tensor, torch.Tensor]:

        output = self.host_net(points.get_features().to(self.device))
        _TYPE = output.dtype

        noise = torch.rand(output.shape).type(_TYPE).to(self.device)
        random_mask = torch.rand(output.shape[0], 1).le(exploration_rate).to(self.device)
        output = output * ~random_mask + noise * random_mask
        # The random noises still have to go through decode_tensor, therefore are still never illegal moves.
        host_move_binary, chosen_actions = self.decode_tensor(output, masked=masked)

        return host_move_binary, chosen_actions

    def _agent_move(self, points: PointsBase,
                    host_moves: torch.Tensor,
                    masked: Optional[bool] = True,
                    scale_observation: Optional[bool] = True,
                    inplace: Optional[bool] = True,
                    exploration_rate: Optional[float] = 0.0) -> torch.Tensor:
        action_prob = self.agent_net(
            {"points": points.get_features().to(self.device), "coords": host_moves.to(self.device)})
        if masked:
            minimum = torch.finfo(action_prob.dtype).min
            masked_value = (1 - host_moves.to(self.device)) * minimum
            action_prob = action_prob * host_moves.to(self.device) + masked_value

        actions = torch.argmax(action_prob, dim=1)

        noise = torch.randint(0, action_prob.shape[1], actions.shape).type(actions.dtype).to(actions.device)
        random_mask = torch.rand(actions.shape[0]).le(exploration_rate).to(actions.device)
        actions = actions * ~random_mask + noise * random_mask

        _TYPE = points.points.dtype
        if inplace:
            points.shift(host_moves.type(_TYPE), actions.type(_TYPE))
            points.get_newton_polytope()
            if scale_observation:
                points.rescale()
        return actions

    @staticmethod
    def _reward(sample_for: str,
                obs: torch.Tensor,
                next_obs: torch.Tensor,
                next_done: np.ndarray) -> np.ndarray:
        if sample_for == "host":
            return next_done.astype(float)
        elif sample_for == "agent":
            return (~next_done).astype(float)

    @staticmethod
    def decode_tensor(t, masked=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Decode a batch of encoded host actions into a batch of multi-binary (0 or 1) arrays.
            Parameters:
                t: a 2-dim tensor. Each row is a list of logits for each number from 0 to 2**dim-1.
                masked: filter out choices with less than 2 coordinates (excluding 0, 1, 2, 4, 8, ...).
        """
        device = t.device
        dim = int(np.log2(t.shape[1]))
        assert 2 ** dim == t.shape[1], f"The length of the 2nd axis must be length a power of 2. Got {t.shape} instead."

        binary = torch.zeros(2 ** dim, dim)
        minimum = torch.finfo(t.dtype).min

        if masked:
            mask = torch.FloatTensor(mask_encoded_action(dim)).unsqueeze(0).to(device)
        else:
            mask = torch.ones(1, 2 ** dim).type(torch.float).to(device)

        masked_values = ((1 - mask) * minimum).repeat(t.shape[0], 1)

        for i in range(2 ** dim):
            b = bin(i)[2:]
            for j in range(len(b) - 1, -1, -1):
                if b[j] == '1':
                    binary[i][len(b) - 1 - j] = 1
        chosen_actions = torch.argmax(t * mask + masked_values, dim=1)
        return torch.index_select(binary.to(device), 0, chosen_actions.to(device)), chosen_actions
