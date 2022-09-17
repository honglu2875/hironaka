from typing import List

from torch.nn import DataParallel

from hironaka.trainer.FusedGame import FusedGame
from hironaka.trainer.Trainer import Trainer


def activate_dp(trainer: Trainer, device_ids: List[int]):
    net_list = ['host_net', 'agent_net', 'host_net_target', 'agent_net_target']
    for net_str in net_list:
        if hasattr(trainer, net_str):
            setattr(trainer, net_str, DataParallel(getattr(trainer, net_str), device_ids))

    trainer.fused_game = ParallelFusedGame(FusedGame(trainer.host_net, trainer.agent_net, device=trainer.device,
                                                     log_time=trainer.log_time, reward_func=trainer.reward_func,
                                                     dtype=trainer.dtype), device_ids=device_ids)


class ParallelFusedGame:
    """
    A wrapper making FusedGame run on multiple GPU. It only exposes `step` method.
    """

    def __init__(self, fused_game: FusedGame, device_ids: List[int]):
        self.fused_game = fused_game
        self.device_ids = device_ids
