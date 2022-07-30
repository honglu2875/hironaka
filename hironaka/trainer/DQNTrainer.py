import torch.optim
from torch import nn

from hironaka.src import polyak_update
from hironaka.trainer.ReplayBuffer import ReplayBuffer
from hironaka.trainer.Trainer import Trainer


class DQNTrainer(Trainer):
    """
        Operate the standard DDQN.

        Extra configs:
            max_grad_norm
    """

    role_specific_hyperparameters = ['batch_size', 'gamma', 'tau', 'initial_rollout_size', 'max_rollout_step',
                                     'steps_before_rollout', 'rollout_size']

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.max_grad_norm = self.config['max_grad_norm']

        # Set up target Q-networks for both host and agent
        for role in ['host', 'agent']:
            head_cls, net_arch, input_dim, output_dim = self.get_net_args(role)
            head = head_cls(self.dimension, self.max_num_points)
            setattr(self, f'{role}_net_target', self._make_network(head, net_arch, input_dim, output_dim))

            # Setting tau=1 is the same as copying weights
            q_net, q_net_target = self.get_net(role), self.get_net_target(role)
            self._update_target_net(q_net, q_net_target, 1)

    def _fit_network(self, q_net: nn.Module, q_net_target: nn.Module,
                     optimizer: torch.optim.Optimizer, replay_buffer: ReplayBuffer,
                     batch_size: int, gamma: int,
                     log_prefix: str = '', current_step: int = 0, **kwargs) -> int:
        """
            Update the q_net and q_net_target. Adopted from stable-baseline3.
            Return:
                loss: int (only for logging purpose)
        """
        # Sample replay buffer
        replay_data = replay_buffer.sample(batch_size)
        observations, actions, rewards, dones, next_observations = replay_data

        with torch.no_grad():
            # Compute the next Q-values using the target network
            next_q_values = q_net_target(next_observations)
            # Follow greedy policy: use the one with the highest value
            next_q_values, _ = next_q_values.max(dim=1)
            # Avoid potential broadcast issue
            next_q_values = next_q_values.reshape(-1, 1)
            # 1-step TD target
            target_q_values = rewards + (~dones) * gamma * next_q_values

        # Get current Q-values estimates
        current_q_values = q_net(observations)

        # Retrieve the q-values for the actions from the replay buffer
        current_q_values = torch.gather(current_q_values, dim=1, index=actions.long())

        # Compute Huber loss (less sensitive to outliers)
        loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values)

        # Optimize the policy
        optimizer.zero_grad()
        loss.backward()
        # Clip gradient norm
        nn.utils.clip_grad_norm_(q_net.parameters(), self.max_grad_norm)
        optimizer.step()

        # Logging
        if self.use_tensorboard and self.layerwise_logging:
            for j, layer in enumerate(q_net.parameters()):
                self.tb_writer.add_scalar(f'{log_prefix}/model/layer-{j}/avg_wt', layer.mean().item(),
                                          self.total_num_steps + current_step)
                self.tb_writer.add_scalar(f'{log_prefix}/model/layer-{j}/std', layer.std().item(),
                                          self.total_num_steps + current_step)
                self.tb_writer.add_scalar(f'{log_prefix}/gradient/layer-{j}/grad_avg',
                                          layer.grad.mean().item(), self.total_num_steps + current_step)
                self.tb_writer.add_scalar(f'{log_prefix}/gradient/layer-{j}/grad_std',
                                          layer.grad.std().item(), self.total_num_steps + current_step)

        return loss.item()

    def _train(self, steps: int):
        losses = []
        param = {'host': self.get_all_role_specific_param('host'),
                 'agent': self.get_all_role_specific_param('agent')}

        for i in range(steps):
            for role in ['host', 'agent']:
                losses.append(self._fit_network(
                    self.get_net(role), self.get_net_target(role), self.get_optim(role), self.get_replay_buffer(role),
                    **param[role],
                    log_prefix=f'{role}-{self.device_num}',
                    current_step=i
                ))

                if self.total_num_steps + i % param[role]['steps_before_rollout'] == 0:
                    self._generate_rollout(role, param[role]['rollout_size'])

        self.total_num_steps += steps

    @staticmethod
    def _update_target_net(q_net: nn.Module, q_net_target: nn.Module, tau: int):
        polyak_update(q_net.parameters(), q_net_target.parameters(), tau)

    # ------- Role specific getters -------
    def get_net_target(self, role):
        return getattr(self, f'{role}_net_target')

    def get_gamma(self, role):
        return getattr(self, f'{role}_gamma')

    def get_tau(self, role):
        return getattr(self, f'{role}_tau')
