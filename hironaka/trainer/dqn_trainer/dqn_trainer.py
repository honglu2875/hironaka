from copy import deepcopy
from typing import Iterable

import torch
from torch import nn

from hironaka.src import polyak_update
from hironaka.trainer.replay_buffer import ReplayBuffer
from hironaka.trainer.timer import Timer
from hironaka.trainer.trainer import Trainer


class DQNTrainer(Trainer):
    """
    Operate the standard DDQN.

    Extra global configs:
        max_grad_norm
    """

    role_specific_hyperparameters = [
        "batch_size",
        "gamma",
        "tau",
        "initial_rollout_size",
        "max_rollout_step",
        "steps_before_rollout",
        "steps_before_update_target",
        "rollout_size",
    ]

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.max_grad_norm = self.config["max_grad_norm"]

        # Set up target Q-networks for both host and agent
        for role in self.trained_roles:
            net = self.get_net(role)
            setattr(self, f"{role}_net_target", deepcopy(net).to(self.device))

            # Setting tau=1 is the same as copying weights
            q_net, q_net_target = self.get_net(role), self.get_net_target(role)
            self._update_target_net(q_net, q_net_target, 1)

    def _fit_network(
        self,
        q_net: nn.Module,
        q_net_target: nn.Module,
        optimizer: torch.optim.Optimizer,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        gamma: int,
        log_prefix: str = "",
        current_step: int = 0,
        **kwargs,
    ) -> int:
        """
        Update the q_net and q_net_target. Adopted from stable-baselines3.
        Return:
            loss: int (only for logging purpose)
        """
        # Sample replay buffer
        with Timer(log_prefix + "-sample_replay_buffer_total", self.time_log, active=self.log_time, use_cuda=self.use_cuda):
            replay_data = replay_buffer.sample(batch_size, device=self.device)
            observations, actions, rewards, dones, next_observations = replay_data

        with Timer(log_prefix + "-get_target_q_value_total", self.time_log, active=self.log_time, use_cuda=self.use_cuda):
            with torch.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = q_net_target(next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = rewards + (~dones) * gamma * next_q_values

        with Timer(log_prefix + "-q_net_fwd_total", self.time_log, active=self.log_time, use_cuda=self.use_cuda):
            # Get current Q-values estimates
            current_q_values = q_net(observations)

        with Timer(log_prefix + "-gather_total", self.time_log, active=self.log_time, use_cuda=self.use_cuda):
            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = torch.gather(current_q_values, dim=1, index=actions.long())

        with Timer(log_prefix + "-loss_total", self.time_log, active=self.log_time, use_cuda=self.use_cuda):
            # Compute Huber loss (less sensitive to outliers)
            loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values)

        with Timer(log_prefix + "-bwd_total", self.time_log, active=self.log_time, use_cuda=self.use_cuda):
            # Optimize the policy
            optimizer.zero_grad()
            loss.backward()

        with Timer(log_prefix + "-grad_step_total", self.time_log, active=self.log_time, use_cuda=self.use_cuda):
            # Clip gradient norm
            nn.utils.clip_grad_norm_(q_net.parameters(), self.max_grad_norm)
            optimizer.step()

        with Timer(log_prefix + "-logging_total", self.time_log, active=self.log_time, use_cuda=self.use_cuda):
            # Logging
            # We hard-coded the 20-step interval for now. Profiling analysis shows it is about 1/10 on time cost
            #   comparing to autograd happening above.
            if self.use_wandb and (self.total_num_steps + current_step) % 100 == 0:
                if self.layerwise_logging:
                    for j, layer in enumerate(q_net.parameters()):
                        self.log_writer.add_scalar(
                            f"{log_prefix}/model/layer-{j}/avg_wt", layer.mean().item(), self.total_num_steps + current_step
                        )
                        self.log_writer.add_scalar(
                            f"{log_prefix}/model/layer-{j}/std", layer.std().item(), self.total_num_steps + current_step
                        )
                        self.log_writer.add_scalar(
                            f"{log_prefix}/gradient/layer-{j}/grad_avg",
                            layer.grad.mean().item(),
                            self.total_num_steps + current_step,
                        )
                        self.log_writer.add_scalar(
                            f"{log_prefix}/gradient/layer-{j}/grad_std",
                            layer.grad.std().item(),
                            self.total_num_steps + current_step,
                        )
                self.log_writer.add_scalar(f"{log_prefix}/loss", loss.item(), self.total_num_steps + current_step)

        return loss

    def _train(
        self, steps: int, evaluation_interval: int = 1000, players: Iterable[str] = ("host", "agent"), prefix: str = ""
    ):
        losses = []
        param = {}
        net_param = {}

        for role in players:
            if role not in self.trained_roles:
                self.logger.error(
                    f"{role} is not trainable. Either it is not present in config or it was set as a dummy module."
                )
                return

        for role in players:
            net_param[role] = self.get_net(role), self.get_net_target(role), self.get_optim(role), self.get_replay_buffer(role)
            param[role] = self.get_all_role_specific_param(role)
        model_prefix = f"{prefix}-{self.version_string}-{self.device_num}"

        for i in range(steps):
            self.set_learning_rate()
            for role in players:
                # Updating target network with a delay helps the stability.
                with Timer(f"update_{role}_target_net", self.time_log, active=self.log_time, use_cuda=self.use_cuda):
                    if (self.total_num_steps + i) % param[role]["steps_before_update_target"] == 0:
                        self._update_target_net(net_param[role][0], net_param[role][1], param[role]["tau"])

                with Timer(f"fit_{role}_net_total", self.time_log, active=self.log_time, use_cuda=self.use_cuda):
                    losses.append(
                        self._fit_network(*net_param[role], **param[role], log_prefix=f"{role}-{model_prefix}", current_step=i)
                    )

                with Timer(f"collect_{role}_rollouts_total", self.time_log, active=self.log_time, use_cuda=self.use_cuda):
                    if (self.total_num_steps + i) % param[role]["steps_before_rollout"] == 0:
                        with self.inference_mode():
                            self.collect_rollout(role, param[role]["rollout_size"])

            with Timer("evaluate_total", self.time_log, active=self.log_time, use_cuda=self.use_cuda):
                if i % evaluation_interval == 0 and i != 0:
                    with self.inference_mode():
                        rhos = self.evaluate_rho()
                        for role in players:
                            actions = self.count_actions(role, 100, er=0.0)
                            self.logger.info(actions)
                        self.logger.info(rhos)

                        if self.use_wandb:
                            self.log_writer.add_scalar(
                                f"{model_prefix}/rhos/host-agent", rhos[0].item(), self.total_num_steps + i
                            )
                            self.log_writer.add_scalar(
                                f"{model_prefix}/rhos/host-random", rhos[1].item(), self.total_num_steps + i
                            )
                            self.log_writer.add_scalar(
                                f"{model_prefix}/rhos/host-choosefirst", rhos[2].item(), self.total_num_steps + i
                            )
                            self.log_writer.add_scalar(
                                f"{model_prefix}/rhos/random-agent", rhos[3].item(), self.total_num_steps + i
                            )
                            self.log_writer.add_scalar(
                                f"{model_prefix}/rhos/allcoord-agent", rhos[4].item(), self.total_num_steps + i
                            )

        self.total_num_steps += steps

    @staticmethod
    def _update_target_net(q_net: nn.Module, q_net_target: nn.Module, tau: int):
        polyak_update(q_net.parameters(), q_net_target.parameters(), tau)

        # Also copy batch norm running mean/var
        modules = []
        modules_target = []
        for module, module_target in zip(q_net.modules(), q_net_target.modules()):
            for key in ["running_mean", "running_var"]:
                if hasattr(module, key):
                    modules.append(getattr(module, key))
                    modules_target.append(getattr(module_target, key))
        polyak_update(modules, modules_target, tau)

    # ------- Role specific getters -------
    def get_net_target(self, role):
        return getattr(self, f"{role}_net_target", None)
