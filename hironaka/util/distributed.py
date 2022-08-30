from typing import List

from torch.nn import DataParallel

from hironaka.trainer.FusedGame import FusedGame
from hironaka.trainer.Trainer import Trainer


def activate_dp(trainer: Trainer, device_ids: List[int]):
    trainer.host_net = DataParallel(trainer.host_net, device_ids)
    trainer.agent_net = DataParallel(trainer.host_net, device_ids)
    trainer.fused_game = FusedGame(trainer.host_net, trainer.agent_net, device=trainer.device,
                                   log_time=trainer.log_time, reward_func=trainer.reward_func, dtype=trainer.dtype)
