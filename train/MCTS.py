import logging
import random
import sys
from typing import List, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from hironaka.core import TensorPoints
from hironaka.agent import RandomAgent, ChooseFirstAgent
from hironaka.src import _snippets as snip
from hironaka import host
from hironaka.validator import HironakaValidator
from hironaka.trainer import Trainer
from hironaka.trainer.player_modules import ChooseFirstAgentModule

import collections as col

ITERATIONS = 1000


# WARNING: this only works for 1st batch, dim = 3 and maximal 10 points for now!

class HironakaNet(nn.Module):

    def __init__(self, dim=3):
        super(HironakaNet, self).__init__()
        self.dim = dim
        self.choices = 2 ** dim - dim - 1
        self.fc1 = nn.Linear(dim * 10, 64)  # input: all coordinates of Points
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32,
                             2 ** dim - dim)

    def forward(self, x):
        # # Max pooling over a (2, 2) window
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # # If the size is a square, you can specify with a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x_prob = torch.narrow(x, 0, 0, self.choices)
        x_reward = torch.narrow(x, 0, self.choices, 1)
        x_prob = F.softmax(x_prob, dim=0)
        x = torch.cat((x_prob, x_reward), dim=0)
        return x

class TempAgentNet(nn.Module):
    def __init__(self, dim = 3):
        super(TempAgentNet, self).__init__()
        self.dim = dim
        self.choices = dim
        self.fc1 = nn.Linear(dim * 11, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, dim+1)

    def forward(self, x):
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x_prob = torch.narrow(x, 0, 0, self.choices)
        x_reward = torch.narrow(x, 0, self.choices, 1)
        x_prob = F.softmax(x_prob, dim=0)
        x = torch.cat((x_prob, x_reward), dim=0)

        return x

class trained_host(host.Host):
    def __init__(self, net: HironakaNet):
        self.net = net
        self.dim = net.dim
        super().__init__()

        self.action_translate = []
        for i in range(1, 2 ** self.dim):
            if not (i & (i - 1) == 0):  # Check if i is NOT a power of 2
                self.action_translate.append(i)

    def _action_to_coords(self, action: int):

        current_coord = 0
        coords = []
        action = self.action_translate[action]
        while action != 0:
            if action % 2:
                coords.append(current_coord)
            current_coord += 1
            action = action // 2
        return coords

    def _select_coord(self, points: TensorPoints, debug=False):
        answer = []
        if not isinstance(points, TensorPoints):
            points = TensorPoints(points.points, max_num_points=10)
        for i in range(points.batch_size):
            x = points.points[0]
            result = self.net(x)
            prob_vector = torch.narrow(result, 0, 0, self.net.choices)
            prob_vector = prob_vector.tolist()
            reward_vector = torch.narrow(result, 0, self.net.choices, 1)
            reward = reward_vector.item()
            current_prob, choice = -float("inf"), -1
            for _, prob in enumerate(prob_vector):
                if prob > current_prob:
                    current_prob = prob
                    choice = _

            coords = self._action_to_coords(choice)
            answer.append(coords)

        return answer


# todo: decide how to deal with hash. No inverse hash is needed anymore, so any hash can work. Maybe use default hash
#  of tensors.
def hashify(s: TensorPoints):
    hashed_str = ""
    current_points = s.points[0].tolist()
    for point in current_points:
        for coord in point:
            hashed_str += '%.8f' % coord
            hashed_str += ','

    return hashed_str


class MCTS:
    def __init__(self, state, host_nn, **config):
        options = {
            'env': ChooseFirstAgent(),
            'max_depth': 15,
            'c_puct': 0.5
        }
        options.update(config)

        self.initial_state = state if isinstance(state, TensorPoints) else TensorPoints(state.points,
                                                                                        max_num_points=10)
        self.dim = state.dimension
        self.nn = host_nn
        self.env = options['env']
        self.max_depth = options['max_depth']
        self.c_puct = options['c_puct']
        # The dicts will have value type as float tensors.
        self.P = col.defaultdict()
        self.Q = col.defaultdict()
        self.N = col.defaultdict()

        self.reward = col.defaultdict()
        self.visited = col.defaultdict()

        self.coder = snip.HostActionEncoder(dim=self.dim)

    def run(self, iteration=100, state=None):
        for _ in range(iteration):
            if not state:
                this_state = self.initial_state.copy()
            else:
                this_state = state.copy()
            self._search(this_state)

    def get_sample(self, state: TensorPoints):
        this_key = hashify(state)

        return torch.softmax(self.N[this_key], 0)

    def _search(self, s: TensorPoints, depth=0):
        hashed_s = hashify(s)

        if s.ended:
            current_reward = 1
            self.reward[hashed_s] = current_reward
            return 1

        if depth >= self.max_depth:
            self.reward[hashed_s] = -1
            return -1

        if not (hashed_s in self.visited):
            self.visited[hashed_s] = 1
            result = self.nn(s.points[0])
            self.P[hashed_s] = result[:self.nn.choices]
            current_reward = result[self.nn.choices].item()
            self.Q[hashed_s] = torch.zeros(self.nn.choices)
            self.N[hashed_s] = torch.zeros(self.nn.choices)
            self.reward[hashed_s] = current_reward
            return current_reward

        # For agent, the tree node includes host actions. Change hash and stuff to that.

        if torch.count_nonzero(self.N[hashed_s]).item() == self.nn.choices:
            this_action = random.randint(0, self.nn.choices - 1)
        else:
            u = self.Q[hashed_s] + self.c_puct * self.P[hashed_s] * torch.div(torch.sqrt(torch.sum(self.N[hashed_s])),
                                                                              1 + self.N[hashed_s])
            this_action = torch.argmax(u)

        coords = [self.coder.decode(this_action)]

        next_s = s.copy()  # Since I need to run the same MCTS multiple times, I don't alter the original game state.
        self.env.move(next_s, coords)
        next_s.rescale()
        next_s = TensorPoints(next_s.get_features())

        current_reward = self._search(next_s, depth + 1)

        self.Q[hashed_s] = torch.div(self.N[hashed_s] * self.Q[hashed_s] + current_reward, self.N[hashed_s] + 1)

        self.N[hashed_s][this_action] += 1
        self.reward[hashed_s] = current_reward
        return current_reward


class MCTS2:
    def __init__(self, state, host_net, agent_net, train_target='host', **config):
        options = {
            'env': ChooseFirstAgent(),
            'max_depth': 15,
            'c_puct': 0.5
        }
        options.update(config)
        self.initial_state = state if isinstance(state, TensorPoints) else TensorPoints(state.points,
                                                                                        max_num_points=10)
        self.dim = state.dimension
        self.host_net = host_net
        self.agent_net = agent_net
        self.max_depth = options['max_depth']
        self.c_puct = options['c_puct']

        assert train_target in ['host', 'agent']
        self.train_target = train_target
        # The dicts will have value type as float tensors.
        self.P = col.defaultdict()
        self.Q = col.defaultdict()
        self.N = col.defaultdict()

        self.reward = col.defaultdict()
        self.visited = col.defaultdict()

        self.coder = snip.HostActionEncoder(dim=self.dim)

    def run(self, iteration=100, state=None):
        for _ in range(iteration):
            if not state:
                this_state = self.initial_state.copy()
            else:
                this_state = state.copy()
            self._search(this_state)

    def get_sample(self, state: TensorPoints):
        this_key = hashify(state)
        if self.train_target == 'agent':
            host_action = self.coder.decode(torch.argmax(self.host_net(state.points[0])).item())
            host_action_tensor = torch.zeros(self.dim)
            for coord in host_action:
                host_action_tensor[coord] = 1
            net_input = torch.cat((torch.flatten(state.points[0]), host_action_tensor), 0)

            for coord in host_action:
                this_key += str(coord)
        else:
            net_input = state.points[0]

        prob_vector = torch.softmax(self.N[this_key], 0)
        sample = (net_input, prob_vector)

        return sample

    def _search(self, s: TensorPoints, depth=0):
        hashed_s = hashify(s)

        # End the game if one point left or it goes too deep.
        if s.ended:
            current_reward = 1
            self.reward[hashed_s] = current_reward
            return self._reward(1)

        if depth >= self.max_depth:
            self.reward[hashed_s] = -1
            return self._reward(-1)

        training_net = getattr(self, f'{self.train_target}_net')

        # If we are training agent, we require host to make a move first, since it is a part of input for agent net.
        if self.train_target == 'agent':
            host_action = self.coder.decode(torch.argmax(self.host_net(s.points[0])).item())
            host_action_tensor = torch.zeros(self.dim)
            for coord in host_action:
                host_action_tensor[coord] = 1

            agent_net_input = torch.cat((torch.flatten(s.points[0],0), host_action_tensor), 0)
            result = self.agent_net(agent_net_input)

            for coord in host_action:
                hashed_s += str(coord)
        else:
            result = self.host_net(s.points[0])

        # If we haven't been here before, create the node info, then go back.
        if not (hashed_s in self.visited):
            self.visited[hashed_s] = 1
            self.P[hashed_s] = result[:training_net.choices]
            current_reward = result[training_net.choices].item()
            self.Q[hashed_s] = torch.zeros(training_net.choices)
            self.N[hashed_s] = torch.zeros(training_net.choices)
            self.reward[hashed_s] = current_reward
            return current_reward

        if torch.count_nonzero(self.N[hashed_s]).item() == training_net.choices:
            this_action = random.randint(0, training_net.choices - 1)
        else:
            u = self.Q[hashed_s] + self.c_puct * self.P[hashed_s] * torch.div(torch.sqrt(torch.sum(self.N[hashed_s])),
                                                                              1 + self.N[hashed_s])
            this_action = torch.argmax(u)

        if self.train_target == 'host':
            host_action = self.coder.decode(this_action)
            host_action_tensor = torch.zeros(self.dim)
            for coord in host_action:
                host_action_tensor[coord] = 1
            agent_input = torch.cat((torch.flatten(s.points[0]), torch.tensor(host_action_tensor)),0)
            agent_action = torch.argmax(self.agent_net(agent_input))
        else:
            agent_action = this_action

        next_s = s.copy()  # Since I need to run the same MCTS multiple times, I don't alter the original game state.

        next_s.shift([host_action],[agent_action])
        next_s.rescale()
        next_s = TensorPoints(next_s.get_features())

        current_reward = self._search(next_s, depth + 1)

        self.Q[hashed_s] = torch.div(self.N[hashed_s] * self.Q[hashed_s] + current_reward, self.N[hashed_s] + 1)

        self.N[hashed_s][this_action] += 1
        self.reward[hashed_s] = current_reward
        return current_reward

    def _reward(self, ended: int) -> int:
        if self.train_target == 'agent':
            ended = -ended
        return ended


class MCTSTrainer2(Trainer.Trainer):
    role_specific_hyperparameters = ['batch_size', 'initial_rollout_size', 'max_rollout_step', 'c_puct', 'lr',
                                     'MSE_coefficient']
    model_specific_hyperparameters = ['max_depth']

    def __init__(self, config: Union[Dict[str, Any], str], host_module, agent_module):

        if isinstance(config, str):
            self.options = self.load_yaml(config)
        else:
            self.options = config

        super().__init__(config=self.options, host_net=host_module, agent_net=agent_module)

        self.coder = snip.HostActionEncoder(dim=self.dimension)

    def _make_network(self, head: nn.Module, net_arch: list, input_dim: int, output_dim: int) -> nn.Module:
        return HironakaNet(dim=self.dimension)

    def _policy_iter(self, state: TensorPoints, c_puct=0.5, max_depth=20, train_target = 'host'):
        # This method returns samples of a single complete game.
        examples = ([], [])

        mcts_instance = MCTS2(state=state, host_net=self.host_net, agent_net=self.agent_net, train_target= train_target,
                              max_depth=max_depth, c_puct=c_puct)

        depth = 0

        while True:
            mcts_instance.run(iteration=20, state=state)
            current_sample = mcts_instance.get_sample(state)
            examples[0].append(current_sample[0])
            examples[1].append(current_sample[1])
            if train_target == 'host':
                best_action = torch.argmax(current_sample[1], 0).item()
                host_action = self.coder.decode(best_action)
                host_action_tensor = torch.zeros(self.dimension)
                for coord in host_action:
                    host_action_tensor[coord] = 1
                agent_input = torch.cat((torch.flatten(state.points[0]), torch.tensor(host_action_tensor)), 0)
                agent_action = torch.argmax(self.agent_net(agent_input))
            else:
                host_action = self.coder.decode(torch.argmax(self.host_net(state.points[0])))
                agent_action = torch.argmax(current_sample[1], 0).item()

            state.shift([host_action], [agent_action])
            state.get_newton_polytope()
            state.rescale()
            state = TensorPoints(state.get_features())

            depth += 1

            if state.ended:
                for i, sample in enumerate(examples[1]):
                    examples[1][i] = torch.cat((sample, torch.zeros(1) + 1), 0)
                break
            elif depth >= max_depth:
                for i, sample in enumerate(examples[1]):
                    examples[1][i] = torch.cat((sample, torch.zeros(1) - 1), 0)
                break

        return examples

    def _loss_function(self, pred, y: List[torch.FloatTensor], train_target ='host'):
        # loss function is a linear combination of MSE on winning/lossing prediction and cross entropy on probability
        # vectors.

        choices = getattr(self, f'{train_target}_net').choices

        loss = torch.zeros(1)
        for i, pred in enumerate(pred):
            choice_pred = torch.narrow(pred, 0, 0, choices)
            reward_pred = torch.narrow(pred, 0, choices, 1)
            choice_y = torch.narrow(y[i], 0, 0, choices)
            choice_y = F.softmax(choice_y, dim=0)
            reward_y = torch.narrow(y[i], 0, choices, 1)
            loss = loss + self.host_MSE_coefficient * torch.square((reward_pred - reward_y)) - torch.dot(choice_y,
                                                                                                            torch.log(
                                                                                                             choice_pred))

            # If it is changed to the build-in cross entropy, remove softmax both in the network and in the MCTS.get_sample().

        return loss

    def _train(self, steps=100, **config):

        evaluation_interval = config['evaluation_interval']
        train_target = config['train_target']

        for i in range(steps):
            while True:
                test_points = TensorPoints(snip.generate_batch_points(n=10, dimension=self.dimension, max_value=50),
                                           max_num_points=10)
                test_points.get_newton_polytope()
                test_points.rescale()
                test_points = TensorPoints(test_points.get_features())
                if not test_points.ended:
                    break

            losses = []

            c_puct = self.host_c_puct

            examples = self._policy_iter(state=test_points, c_puct=c_puct, max_depth=self.max_depth, train_target = train_target)

            data = [torch.FloatTensor(_) for _ in examples[0]]

            y = [torch.FloatTensor(_) for _ in examples[1]]
            pred = []
            lr = getattr(self, f'{train_target}_lr')
            if train_target == 'host':
                optimizer = torch.optim.Adam(self.host_net.parameters(), lr=lr)

                for batch, x in enumerate(data):
                    this_pred = self.host_net(x)
                    pred.append(this_pred)

                loss = self._loss_function(pred, y, train_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            else:
                optimizer = torch.optim.Adam(self.agent_net.parameters(), lr=lr)

                for batch, x in enumerate(data):
                    this_pred = self.agent_net(x)
                    pred.append(this_pred)

                loss = self._loss_function(pred, y, train_target = train_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())


            if len(losses) > evaluation_interval:
                losses.pop(0)
            if i % evaluation_interval == 0:
                self.logger.info(
                    "The MA of last " + str(evaluation_interval) + " iterations is: " + str(sum(losses) / len(losses)))
                self.logger.info("Current iteration: " + str(i) + '/' + str(steps))

            print("training ", train_target)
            print("we are in interation ",i)

    def save(self, path='test_model.pth'):
        torch.save(self.host_net, path)
        self.logger.info("Saved model as: " + path)


class MCTSTrainer:
    # todo: cooperate with the abstract trainer class.
    def __init__(self, **config):
        """
        Config is a dictionary that contains parameters.
        Currently there are two:
        dim: The dimension of the board (default = 3)
        nn_parameters = a list of strings to build the neural network (not realized yet).
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.hasHandlers():
            logger.addHandler(logging.StreamHandler(sys.stdout))

        self.log = logger

        self.dim = config["dim"]
        self.net = HironakaNet(dim=self.dim)
        self.coder = snip.HostActionEncoder(dim=self.dim)

    def _loss_function(self, pred, y: List[torch.FloatTensor]):
        # loss function is a linear combination of MSE on winning/lossing prediction and cross entropy on probability
        # vectors.

        loss = torch.zeros(1)
        for i, pred in enumerate(pred):
            choice_pred = torch.narrow(pred, 0, 0, self.net.choices)
            reward_pred = torch.narrow(pred, 0, self.net.choices, 1)
            choice_y = torch.narrow(y[i], 0, 0, self.net.choices)
            choice_y = F.softmax(choice_y, dim=0)
            reward_y = torch.narrow(y[i], 0, self.net.choices, 1)
            loss = loss + self.MSE_coefficient * torch.square((reward_pred - reward_y)) - torch.dot(choice_y, torch.log(
                choice_pred))

            # If it is changed to the build-in cross entropy, remove softmax both in the network and in the MCTS.get_sample().

        return loss

    def host_from_nn(self) -> trained_host:
        return trained_host(self.net)

    def _arena(self, new_host, old_host: host.Host, steps=1000, agent=ChooseFirstAgent()) -> bool:
        test_validator = HironakaValidator(new_host, agent, dimension=self.dim)
        new_host_record = test_validator.playoff(num_steps=steps, verbose=0)
        test_validator = HironakaValidator(old_host, agent, dimension=self.dim)
        old_host_record = test_validator.playoff(num_steps=steps, verbose=0)
        return (len(new_host_record) > len(old_host_record))

    def _symmertic_sample_generator(self, examples):
        # todo:This method generate new samples from old one by rotating valid points.
        pass

    def _policy_iter(self, state: TensorPoints, c_puct=0.5, max_depth=20):
        # This method returns samples of a single complete game.
        examples = ([], [])

        mcts_instance = MCTS(state=state, host_nn=self.net, env=self.agent, max_depth=max_depth, c_puct=c_puct)

        depth = 0

        while True:
            mcts_instance.run(iteration=20, state=state)
            current_sample = mcts_instance.get_sample(state)
            examples[0].append(state.points[0])
            examples[1].append(current_sample)
            best_action = torch.argmax(current_sample, 0).item()

            coords = self.coder.decode(best_action)

            self.agent.move(state, [coords])

            state.get_newton_polytope()
            state.rescale()
            state = TensorPoints(state.get_features())

            depth += 1

            if state.ended:
                for i, sample in enumerate(examples[1]):
                    examples[1][i] = torch.cat((sample, torch.zeros(1) + 1), 0)
                break
            elif depth >= max_depth:
                for i, sample in enumerate(examples[1]):
                    examples[1][i] = torch.cat((sample, torch.zeros(1) - 1), 0)
                break

        return examples

    def train(self, **config):

        """
        config contains all parameters we need for training.
        It currently has the following parameters:
        ITERATIONS: How many training iterations do we do.
        c_puct: exploration parameter for training.
        lr: learning rate.
        max_depth: The maximal depth of the MCTS.
        MSE_coefficient: the coefficient of the reward part in loss function.
        agent: The agent used for training.
        """

        options = {
            "ITERATIONS": 200,
            "c_puct": 0.5,
            "lr": 1e-4,
            "max_depth": 20,
            "MSE_coefficient": 1.0,
            "agent": ChooseFirstAgent()
        }

        options.update(config)
        self.ITERATIONS = options["ITERATIONS"]
        self.c_puct = options["c_puct"]
        self.lr = options["lr"]
        self.max_depth = options["max_depth"]
        self.MSE_coefficient = options["MSE_coefficient"]
        self.agent = options["agent"]

        for i in range(self.ITERATIONS):
            while True:
                test_points = TensorPoints(snip.generate_batch_points(n=10, dimension=self.dim, max_value=50),
                                           max_num_points=10)
                test_points.get_newton_polytope()
                test_points.rescale()
                test_points = TensorPoints(test_points.get_features())
                if not test_points.ended:
                    break

            losses = []

            examples = self._policy_iter(state=test_points, c_puct=self.c_puct, max_depth=self.max_depth)

            data = [torch.FloatTensor(_) for _ in examples[0]]

            y = [torch.FloatTensor(_) for _ in examples[1]]
            pred = []
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

            for batch, x in enumerate(data):
                this_pred = self.net(x)
                pred.append(this_pred)

            loss = self._loss_function(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if len(losses) > 10:
                losses.pop(0)
            if i % 10 == 0:
                self.log.info("The MA of last 10 iterations is: " + str(sum(losses) / len(losses)))
                self.log.info("Current iteration: " + str(i) + '/' + str(self.ITERATIONS))

    def save_model(self, path='test_model.pth'):
        torch.save(self.net, path)
        self.log.info("Saved model as: " + path)


if __name__ == '__main__':
    host_net = HironakaNet(dim=3)

    # agent_net = ChooseFirstAgentModule(dimension=3, max_num_points=10, device='CPU')
    agent_net = TempAgentNet(dim = 3)

    test = MCTSTrainer2("mcts_config.yml", host_module=host_net, agent_module=agent_net)

    test.train(steps=10, evaluation_interval=10,train_target = 'agent')

    test.save("test_model.pth")

    # test_trainer = MCTSTrainer(dim = 3)
    # test_trainer.train(ITERATIONS = 5000, c_puct = 0.5, lr = 1e-7, max_depth = 20, MSE_coefficient = 0.35, agent = ChooseFirstAgent())
    # test_trainer.save_model("test_model.pth")
