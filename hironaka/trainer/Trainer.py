import abc
import contextlib
import logging
from copy import deepcopy
from typing import List, Any, Dict, Union, Callable, Optional, Type

import torch
import yaml
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from hironaka.core import TensorPoints
from .FusedGame import FusedGame
from .ReplayBuffer import ReplayBuffer
from .Scheduler import ConstantScheduler, ExponentialLRScheduler, ExponentialERScheduler, InverseLRScheduler, Scheduler
from .Timer import Timer
from .nets import create_mlp, AgentFeatureExtractor, HostFeatureExtractor
from .player_modules import DummyModule, RandomHostModule, AllCoordHostModule, RandomAgentModule, ChooseFirstAgentModule
from ..src import HostActionEncoder


class Trainer(abc.ABC):
    """
        Build all the facilities and handle training. Largely inspired by stable-baselines3, but we fuse everything
            together for better clarity and easier modifications.
        To maximize performances and lay the foundation for distributed training, we skip gym environments and game
            wrappers (Host, Agent, Game, etc.).

        A Trainer (and its subclasses) is responsible for training either a single player (either host or agent), or
            a pair of (host, agent) altogether.
        A few important points before using/inheriting:
          - All hyperparameters come from one single nested dict `config` as the positional argument in the constructor.
            A sample config should be given in YAML format for every implementation.
          - NONE of the keys in config may have default values in the class. Not having a lazy mode means the user
            is educated/reminded about every parameter that goes into the RL training. Also avoids messy parameter
            passing when there are crazy chains of configs from subclasses with clashing keys.
            It is okay to have optional config keys though (default: None).
          - Please include role-specific hyperparameters in `role_specific_hyperparameters`. The rest is taken care of.
            You can find the parameters in the dict returned by `get_all_role_specific_param()`
          - The `__init__` of this base class is supposed to handle EVERYTHING about the reading of config. You will be
            given all the attributes after calling `super().__init__(config, **kwargs)` on which your `_train()`
            implementation should be based. The most commonly used attributes in `_train()` are:
                self.host_net, self.host_optim, self.host_replay_buffer,
                self.agent_net, self.agent_optim, self.agent_replay_buffer,
                self.fused_game
            Also use `self.get_all_role_specific_param(role)` to get role-specific attributes defined in
            `role_specific_hyperparameters`.

        Please implement:
            _train()
        Feel free to override:
            _make_network()  # override if one wants to involve more complicated network structures (CNN, GNN, ...).
            _update_learning_rate()
            _generate_rollout()
            copy()  # override if a subclass needs to copy other models/variables.
            save()  # override if a subclass needs to save other models/variables.
            load()

    """
    optim_dict = {'adam': torch.optim.Adam,
                  'sgd': torch.optim.SGD}
    lr_scheduler_dict = {'constant': ConstantScheduler,
                         'exponential': ExponentialLRScheduler,
                         'inverse': InverseLRScheduler}
    er_scheduler_dict = {'constant': ConstantScheduler,
                         'exponential': ExponentialERScheduler}
    replay_buffer_dict = {'base': ReplayBuffer}

    # Please include role-specific hyperparameters that only require simple assignments to attributes.
    #   (except exploration_rate due to more complex nature).
    # Note that all the parameters defined here can be obtained by calling `get_all_role_specific_param()`.
    role_specific_hyperparameters = ['batch_size', 'initial_rollout_size', 'max_rollout_step']

    def __init__(self,
                 config: Union[Dict[str, Any], str],  # Either the config dict or the path to the YAML file
                 node: int = 0,  # For distributed training: the number of node
                 device_num: int = 0,  # For distributed training: the number of cuda device
                 host_net: Optional[nn.Module] = None,  # Pre-assigned host_net. Will ignore host config if set.
                 agent_net: Optional[nn.Module] = None,  # Pre-assigned agent_net. Will ignore agent config if set.
                 reward_func: Optional[Callable] = None,
                 point_cls: Optional[Type[TensorPoints]] = TensorPoints,  # the class to construct points
                 dtype: Optional[Union[Type, torch.dtype]] = torch.float32,
                 ):
        self.logger = logging.getLogger(__class__.__name__)

        self.node = node
        self.device_num = device_num

        if isinstance(config, str):
            self.config = self.load_yaml(config)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError(f"config must be either a str or dict. Got{type(config)}.")

        # Initialize persistent variables
        self.total_num_steps = 0  # Record the total number of training steps

        # -------- Handle Configurations -------- #

        # Highest-level mandatory configs:
        self.use_tensorboard = self.config['use_tensorboard']
        self.layerwise_logging = self.config['layerwise_logging']
        self.log_time = self.config['log_time']
        self.use_cuda = self.config['use_cuda']
        self.scale_observation = self.config['scale_observation']
        self.version_string = self.config['version_string']

        self.dimension = self.config['dimension']
        self.host_action_encoder = HostActionEncoder(self.dimension)
        self.max_num_points = self.config['max_num_points']
        self.max_value = self.config['max_value']
        self.point_cls = point_cls
        self.dtype = dtype

        # Get feature dimension by constructing a dummy tensor
        pts = self.point_cls(torch.rand(1, self.max_num_points, self.dimension))
        self.feature_dim = pts.get_features().reshape(-1).shape[0]
        self.feature_shape = pts.get_features().shape[1:]

        # Add networks. The designed behavior should be the following:
        #   if 'host' is present in config:
        #       if `host_net` is a DummyModule, ignore in `self.trained_roles` but `self.host_net` will be set.
        #       otherwise, it will be included in `self.trained_roles` and initialized if `host_net` is None.
        #   if 'host' is not present in config:
        #       'host_net' MUST be passed during initialization.
        #       But it will not be included in `self.trained_roles`.
        # The same goes for 'agent'...

        self.host_net = host_net
        self.agent_net = agent_net
        assert any([role in self.config for role in ['host', 'agent']]), \
            f"Must have at least one role out of ['host', 'agent'] in the config."

        self.trained_roles = []
        for role, net in zip(['host', 'agent'], [host_net, agent_net]):
            if role in self.config and (not isinstance(net, DummyModule)):
                self.trained_roles.append(role)
            elif role not in self.config:
                assert net is not None, f"{role} is not present in config. A network must be given during init."

        # Create time log
        self.time_log = dict()

        # The suffix string used for logging and saving
        self.string_suffix = f"-{self.version_string}-node_{node}-cuda_{device_num}"

        # Initialize TensorBoard settings
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(comment=self.string_suffix)

        # Set torch device
        self.node = node
        self.device = torch.device(f'cuda:{device_num}') if self.use_cuda else torch.device('cpu')

        # Initialize host and agent parameters
        heads = {'host': HostFeatureExtractor, 'agent': AgentFeatureExtractor}
        input_dims = {'host': self.feature_dim,
                      'agent': self.feature_dim + self.dimension}
        output_dims = {'host': 2 ** self.dimension - self.dimension - 1, 'agent': self.dimension}
        pretrained_nets = {'host': host_net, 'agent': agent_net}

        # Set the reward function
        self.reward_func = reward_func

        for role in ['host', 'agent']:
            if role not in self.config:
                continue

            head_cls, input_dim, output_dim, pretrained_net = \
                heads[role], input_dims[role], output_dims[role], pretrained_nets[role]

            # Initialize hyperparameters
            for key in self.role_specific_hyperparameters:
                setattr(self, f'{role}_{key}', self.config[role][key])

            # Construct networks
            if pretrained_net is not None:
                # The pretrained network should have been set. Or there must be a broken update.
                assert self.get_net(role) is not None
            else:
                net_arch = self.config[role]['net_arch']
                assert isinstance(net_arch, list), f"'net_arch' must be a list. Got {type(net_arch)}."

                head = head_cls(input_dim)
                input_dim = head.feature_dim
                setattr(self, f'{role}_net', self._make_network(head, net_arch, input_dim, output_dim).to(self.device))

            # Construct exploration rate scheduler
            self._make_er_scheduler(role)

            # Construct optimizer, lr scheduler and replay buffer
            setattr(self, f'{role}_optim_config', self.config[role]['optim'])
            if not isinstance(self.get_net(role), DummyModule):
                self._make_optimizer_and_lr(role)
                self._make_replay_buffer(role, output_dim)

        # -------- Initialize states -------- #

        # At the end of the loop, both host_net and agent_net should be set.
        assert self.host_net is not None and self.agent_net is not None
        # Construct FusedGame.
        self._make_fused_game()
        # Generate initial collections of replays if 'deactivate' is not set or not True.
        if self.use_replay_buffer:
            for role in self.trained_roles:
                self.collect_rollout(role, getattr(self, f'{role}_initial_rollout_size'))

    def replace_nets(self, host_net: nn.Module = None, agent_net: nn.Module = None) -> None:
        """
            Override the internal host_net and agent_net with custom networks.
            It is the user's responsibility to make sure the input dimension and the output dimension are correct.
        """
        for role, net in zip(['host', 'agent'], [host_net, agent_net]):
            if net is not None:
                setattr(self, f'{role}_net', net.to(self.device))
                if isinstance(net, DummyModule):
                    self.trained_roles.remove(role)
                elif role in self.config:
                    self._set_optim(role)

        self._make_fused_game()

    def set_trainable(self, players: List[str]):
        for role in players:
            if role not in ['host', 'agent']:
                continue

            assert not isinstance(self.get_net(role), DummyModule), f"{role} net cannot be DummyModule."

            if role not in self.trained_roles:
                self.trained_roles.append(role)

    def replace_reward_func(self, reward_func: Callable):
        """
            Replace the reward function by a new one, and reconstruct the self.fused_game object
        """
        self.reward_func = reward_func
        self._make_fused_game()

    def train(self, steps: int, evaluation_interval: int = 1000, **kwargs):
        """
            Train the networks for a number of steps.
            The definition of 'step' is up to the subclasses. Ideally, each step is one unit that updates both host
                and agent together (but, for example, could already be many epochs of gradient descent.)
        """
        self.set_training(True)
        # The subclass will implement the training logic in _train()
        # Note: `self.total_num_steps` is left for _train() to control.
        with Timer('train_total', self.time_log, active=self.log_time, use_cuda=self.use_cuda):
            self._train(steps, evaluation_interval=evaluation_interval, **kwargs)
        # We always reset training mode to False outside training.
        self.set_training(False)

    def collect_rollout(self, role: str, num_of_games: int):
        """
            Play random games `num_of_games` number of times, and add all the outputs into the replay buffer.
            Parameters:
                role: str. Either 'host' or 'agent'.
                num_of_games: int. Number of steps to play.
        """
        net = self.get_net(role)

        if net is None:  # Might have a broken update if this is triggered.
            self.logger.error(f'{role} network is not set.')
            return

        if isinstance(net, DummyModule):  # Ignore DummyModule silently (as it does not have replay buffer).
            return

        param = self.get_all_role_specific_param(role)
        replay_buffer = self.get_replay_buffer(role)

        exps = self.get_rollout(role, num_of_games, param['max_rollout_step'])
        for exp in exps:
            replay_buffer.add(*exp, clone=True)

    def get_rollout(self, role: str, num_of_games: int, steps: int, er: float = None) -> List:
        """
            Generate roll-out `step` number of times, and return the experiences as a tuple
                (obs, act, rew, done, next_obs).
            Parameters:
                role: str. Either 'host' or 'agent'.
                num_of_games: int. Number of games to play.
                steps: int. Number of maximal steps to play.
                er: int. Learning rate.
        """
        er = self.get_er(role) if er is None else er
        points = self._generate_random_points(num_of_games)
        exps = []

        for i in range(steps):
            if not points.ended:
                exps.append(self.fused_game.step(points, role,
                                                 scale_observation=self.scale_observation,
                                                 exploration_rate=er))
        return exps

    @torch.inference_mode()
    def evaluate_rho(self, num_samples: int = 100, max_steps: int = 100) -> List[torch.Tensor]:
        """
            Estimate the rho value for pairs:
                host_net vs (agent_net, RandomAgent, ChooseFirstAgent)
                (RandomHost, AllCoordHost) vs agent_net
        """
        # TODO: add count_actions into this evaluation and rename
        if self.host_net.training or self.agent_net.training:
            self.logger.error("Host net or agent net is still in training mode. Eval will not be executed")
            return []

        result = []
        dummy_param = (self.dimension, self.max_num_points, self.device)
        hosts = [self.host_net] * 3 + [RandomHostModule(*dummy_param), AllCoordHostModule(*dummy_param)]
        agents = [self.agent_net, RandomAgentModule(*dummy_param), ChooseFirstAgentModule(*dummy_param)] + \
                 [self.agent_net] * 2

        for host, agent in zip(hosts, agents):
            result.append(self.get_rho_for_pair(host, agent, num_samples, max_steps))
        return result

    @torch.inference_mode()
    def get_rho_for_pair(self, host_candidate: nn.Module, agent_candidate: nn.Module,
                         num_samples: int, max_steps: int) -> torch.Tensor:
        points = self._generate_random_points(num_samples)
        fused_game = FusedGame(host_candidate, agent_candidate, device=self.device, reward_func=self.reward_func,
                               dtype=self.dtype)
        initial = sum(points.ended_batch_in_tensor)
        previous = initial
        total_steps = 0
        for i in range(max_steps):
            host_move, _ = fused_game.host_move(points, exploration_rate=0.)
            fused_game.agent_move(points, host_move,
                                  scale_observation=self.scale_observation,
                                  inplace=True,
                                  exploration_rate=0.)
            new_ended = sum(points.ended_batch_in_tensor) - previous
            total_steps += new_ended * (i + 1)
            previous = sum(points.ended_batch_in_tensor)
        total_steps += sum(~points.ended_batch_in_tensor) * max_steps
        return (num_samples - initial) / total_steps

    @torch.inference_mode()
    def count_actions(self, role: str, games: int, max_steps: int = 100, er: float = None) -> torch.Tensor:
        """
            Get samples of rollout and count the action distributions.
        """
        if self.get_net(role).training:
            self.logger.error("Host net or agent net is still in training mode. Eval will not be executed")
            return None

        rollouts = self.get_rollout(role, games, max_steps, er=er)
        if role == 'host':
            max_num = 2 ** self.dimension
        elif role == 'agent':
            max_num = self.dimension
        else:
            raise Exception(f'role must be either host or agent. Got {role}.')

        count = torch.bincount(rollouts[0][1].flatten(), minlength=max_num)
        for i in range(1, len(rollouts)):
            count += torch.bincount(rollouts[i][1].flatten(), minlength=max_num)
        return count

    def copy(self):
        """
            Copy the models and the config to create a new object. (Caution: ReplayBuffer is NOT copied).
            If a subclass would like to copy other models or variables, it MUST be overridden.
        """
        return self.__class__(self.config, node=self.node, device_num=self.device_num,
                              host_net=deepcopy(self.get_net('host')), agent_net=deepcopy(self.get_net('agent')),
                              reward_func=self.reward_func, point_cls=self.point_cls)

    def save(self, path: str):
        """
            Save only models and config as a dict (Caution: ReplayBuffer is NOT saved).
            If a subclass creates extra models (e.g., DQNTrainer.{role}_q_net_target), it MUST be overridden.
        """
        saved = {'host_net': self.get_net('host'), 'agent_net': self.get_net('agent'), 'config': self.config,
                 'reward_func': self.reward_func, 'point_cls': self.point_cls, 'dtype': self.dtype}
        torch.save(saved, path)

    def save_replay_buffer(self, path: str):
        """
            Save replay buffers as a dict.
        """
        saved = {'host_replay_buffer': self.get_replay_buffer('host'),
                 'agent_replay_buffer': self.get_replay_buffer('agent')}
        torch.save(saved, path)

    def load_replay_buffer(self, path: str):
        """
            Load replay buffers from file.
        """
        saved = torch.load(path)
        for key in ['host_replay_buffer', 'agent_replay_buffer']:
            if key not in saved or saved[key] is None:
                continue

            if saved[key].actions.shape != getattr(self, key).actions.shape:
                self.logger.warning(
                    f"The shape of the replay buffers might be different! Got {saved[key].actions.shape} \
                    and {getattr(self, key).actions.shape} on action attributes.")
            setattr(self, key, saved[key])

    @classmethod
    def load(cls, path: str, node: int = 0, device_num: int = 0):
        """
            Load from the model-config dict and reconstruct the Trainer object.
        """
        saved = torch.load(path)
        new_trainer = cls(saved['config'], node=node, device_num=device_num,
                          host_net=saved['host_net'], agent_net=saved['agent_net'],
                          reward_func=saved.get('reward_func'), point_cls=saved.get('point_cls', TensorPoints),
                          dtype=saved.get('dtype', torch.float32))
        return new_trainer

    @abc.abstractmethod
    def _train(self, steps: int, evaluation_interval: int = 1000, **kwargs):
        pass

    # -------- Role specific getters -------- #

    def get_all_role_specific_param(self, role):
        if not hasattr(self, f'{role}_all_param'):
            result = {}
            for key in self.role_specific_hyperparameters:
                result[key] = getattr(self, f'{role}_{key}')
            setattr(self, f'{role}_all_param', result)
        return getattr(self, f'{role}_all_param')

    def get_net(self, role) -> torch.nn.Module:
        return getattr(self, f'{role}_net')

    def get_optim(self, role):
        return getattr(self, f'{role}_optimizer', None)

    def get_lr_scheduler(self, role) -> Scheduler:
        return getattr(self, f'{role}_lr_scheduler', None)

    def get_er_scheduler(self, role) -> Scheduler:
        return getattr(self, f'{role}_er_scheduler', None)

    def get_er(self, role) -> float:
        return self.get_er_scheduler(role)(self.total_num_steps)

    def get_replay_buffer(self, role) -> ReplayBuffer:
        return getattr(self, f'{role}_replay_buffer', None)

    def get_batch_size(self, role) -> int:
        return getattr(self, f'{role}_batch_size')

    def is_dummy(self, role) -> bool:
        return getattr(self, f'{role}_is_dummy')

    # -------- Set internal parameters (used outside initialization) -------- #

    def set_learning_rate(self):
        for role in ['host', 'agent']:
            optimizer = self.get_optim(role)
            scheduler = self.get_lr_scheduler(role)
            if scheduler is None:
                return
            self._update_learning_rate(optimizer, scheduler(self.total_num_steps))

    def set_training(self, training_mode: bool):
        for role in ['host', 'agent']:
            self.get_net(role).train(training_mode)

    @contextlib.contextmanager
    def inference_mode(self):
        self.set_training(False)
        try:
            yield
        finally:
            self.set_training(True)

    # -------- Private utility methods -------- #

    @staticmethod
    def _update_learning_rate(optimizer, new_lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

    def _make_fused_game(self):
        self.fused_game = FusedGame(self.host_net, self.agent_net, device=self.device, log_time=self.log_time,
                                    reward_func=self.reward_func, dtype=self.dtype)

    def _set_optim(self, role: str):
        """
            Use `self.{role}_optim_config` to (re-)build `self.{role}_optimizer` using `self.{role}_net.parameters()`
        """
        cfg = getattr(self, f'{role}_optim_config')
        net = getattr(self, f'{role}_net')
        setattr(self, f'{role}_optimizer', self.optim_dict[cfg['name']](net.parameters(), **cfg['args']))

    def _make_replay_buffer(self, role: str, output_dim: int):
        """
            Create replay buffer and learning rate scheduler inside __init__. Should only be called during initialization.
        """
        cfg = self.config['replay_buffer']
        self.use_replay_buffer = not cfg.get('deactivate', False)
        if self.use_replay_buffer:  # Ignore if `deactivate` is True
            if cfg['use_cuda']:
                device = torch.device('cuda') if self.use_cuda else self.device
            else:
                device = torch.device('cpu')

            if role == 'host':
                input_shape = self.feature_shape
            elif role == 'agent':
                input_shape = {'points': self.feature_shape,
                               'coords': (self.dimension,)}
            else:
                raise Exception('Impossible code path.')

            replay_buffer = self.replay_buffer_dict[cfg['type']](
                input_shape=input_shape,
                output_dim=output_dim,
                device=device,
                dtype=self.dtype,
                **cfg)
            setattr(self, f'{role}_replay_buffer', replay_buffer)

    def _make_optimizer_and_lr(self, role: str):
        """
            Create optimizer inside __init__. Should only be called during initialization.
        """
        optim = self.config[role]['optim']['name']
        assert optim in self.optim_dict, f"'optim' must be one of {self.optim_dict.keys()}. Got {optim}."

        cfg = self.config[role]['optim'].copy()

        # Create learning rate scheduler
        lr = cfg['args']['lr']
        if 'lr_schedule' in cfg:
            lr_scheduler = self.lr_scheduler_dict[cfg['lr_schedule']['mode']](lr, **cfg['lr_schedule'])
        else:
            lr_scheduler = None
        setattr(self, f'{role}_lr_scheduler', lr_scheduler)

        self._set_optim(role)

    def _make_er_scheduler(self, role: str):
        """
            Create exploration rate scheduler inside __init__. Should only be called during initialization.
        """
        cfg = self.config[role]
        er = cfg['er']
        if 'er_schedule' in cfg:
            er_scheduler = self.er_scheduler_dict[cfg['er_schedule']['mode']](er, **cfg['er_schedule'])
        else:
            er_scheduler = ConstantScheduler(er)
        setattr(self, f'{role}_er_scheduler', er_scheduler)

    @staticmethod
    def _make_network(head: nn.Module, net_arch: list, input_dim: int, output_dim: int) -> nn.Module:
        return create_mlp(head, net_arch, input_dim, output_dim)

    def _generate_random_points(self, samples: int) -> TensorPoints:
        pts = torch.randint(self.max_value + 1, (samples, self.max_num_points, self.dimension), dtype=self.dtype,
                            device=self.device)
        points = self.point_cls(pts, device=self.device, dtype=self.dtype)
        points.get_newton_polytope()
        if self.scale_observation:
            points.rescale()
        return points

    # -------- Public static helpers -------- #

    @staticmethod
    def load_yaml(file_path: str) -> dict:
        with open(file_path, "r") as stream:
            config = yaml.safe_load(stream)
        return config
