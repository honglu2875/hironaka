
import logging
import random
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from hironaka.core import TensorPoints
from hironaka.agent import RandomAgent, ChooseFirstAgent
from hironaka.src import _snippets as snip
from hironaka import host
from hironaka.validator import HironakaValidator

import collections as col
import math

ITERATIONS = 1000

#WARNING: this only works for 1st batch, dim = 3 and maximal 10 points for now!

class HironakaNet(nn.Module):

    def __init__(self, dim = 3):
        super(HironakaNet, self).__init__()
        self.dim = dim
        self.choices = 2 ** dim - dim - 1
        # # 1 input image channel, 6 output channels, 5x5 square convolution
        # # kernel
        #self.conv1 = nn.Conv2d(1, 6, 5)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        # # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(dim * 10, 64)  # input: all coordinates of Points
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2**dim - dim) # output: there are 8 = 2^3 action of choosing subsets, and the last number is the expected steps

    def forward(self, x):
        # # Max pooling over a (2, 2) window
        #x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # # If the size is a square, you can specify with a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x_prob = torch.narrow(x,0,0,self.choices)
        x_reward = torch.narrow(x,0,self.choices,1)
        x_prob = F.softmax(x_prob,dim = 0)
        x = torch.cat((x_prob,x_reward), dim = 0)
        return x

class trained_host(host.Host):
    def __init__(self,net: HironakaNet):
        self.net = net
        self.dim = net.dim
        super().__init__()

        self.action_translate = []
        for i in range(1,2**self.dim):
            if not ((i & (i-1) == 0)): # Check if i is NOT a power of 2
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

    def _select_coord(self, points: TensorPoints, debug = False):
        answer = []
        if not isinstance(points, TensorPoints):
            points = TensorPoints(points.points, max_num_points = 10)
        for i in range(points.batch_size):
            x = points.points[0]
            result = self.net(x)
            prob_vector = torch.narrow(result,0,0,self.net.choices)
            prob_vector = prob_vector.tolist()
            reward_vector = torch.narrow(result,0,self.net.choices,1)
            reward = reward_vector.item()
            current_prob, choice = -float("inf"), -1
            for _,prob in enumerate(prob_vector):
                if prob > current_prob:
                    current_prob = prob
                    choice = _

            coords = self._action_to_coords(choice)
            print(coords)
            answer.append(coords)

        return answer
#todo: decide how to deal with hash. Either put them as a method in training, or as a method in Points classes.
def hashify(s:TensorPoints):
    hashed_str = ""
    current_points = s.points[0].tolist()
    for point in current_points:
        for coord in point:
            hashed_str += '%.5f' % coord
            hashed_str += ','

    return hashed_str

def inverse_hash(s:str, dim = 3):
    x = s.split(",")
    x.pop()
    pt = []
    temp = []
    for i, piece in enumerate(x):
        if i%dim == 0:
            temp = []
        temp.append(float(piece))
        if i%dim == dim - 1:
            pt.append(temp)
    while len(pt) < 10:
        pt.append([0. for _ in range(dim)])

    return pt


class MCTS:
    def __init__(self, state, env, nn, max_depth = 15, c_puct = 0.5):

        self.initial_state = state if isinstance(state, TensorPoints) else TensorPoints(state.points, max_num_points = 10) #Turn points into PointsTensor
        self.dim = state.dimension
        self.env = env
        self.nn = nn
        self.max_depth = max_depth
        self.P = col.defaultdict()
        self.Q = col.defaultdict()
        self.N = col.defaultdict()
        self.reward = col.defaultdict()
        self.visited = col.defaultdict()
        self.c_puct = c_puct

        #A pre-calculated table to convert action to coordinates

        self.action_translate = []
        for i in range(1,2**self.dim):
            if not ((i & (i-1) == 0)): # Check if i is NOT a power of 2
                self.action_translate.append(i)

        #self.action_translate gives a table, whose value at index i is the 10-digits convertion of the i-th valid action.

    def run(self, iteration = 100, state = None):
        for _ in range(iteration):
            if not state:
                this_state = self.initial_state.copy()
            else:
                this_state = state.copy()
            self._search(this_state)

    def get_sample(self,state:TensorPoints):
        this_key = hashify(state)

        prob_vector = []
        total_attempt = float(sum(self.N[this_key]))
        if total_attempt == 0:
            print("Total attempt is zero! The state is ", state)
        for attemp in self.N[this_key]:
            prob_vector.append(attemp/total_attempt)

        return prob_vector

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

    def _search(self,s:TensorPoints,depth = 0):
        if torch.isnan(s.points[0][0][0]):
            logging.warning("nan coordinate detected!")
            print(s.points)
        hashed_s = hashify(s)

        if s.ended:
            current_reward = 1
            self.reward[hashed_s] = current_reward
            return 1

        if depth >= self.max_depth:
            return -1

        if not (hashed_s in self.visited):
            self.visited[hashed_s] = 1
            result = self.nn(s.points[0])
            self.P[hashed_s] = result[:self.nn.choices].tolist()
            current_reward = result[self.nn.choices].item()
            self.Q[hashed_s] = [0 for _ in range(self.nn.choices)]
            self.N[hashed_s] = [0 for _ in range(self.nn.choices)]
            self.reward[hashed_s] = current_reward
            return current_reward

        max_u, best_action = -float("inf"), -1

        if all(v == 0 for v in self.Q[hashed_s]):
            best_action = random.randint(0,self.nn.choices-1)
        else:
            for action in range(0, self.nn.choices):
                u = self.Q[hashed_s][action] + self.c_puct * self.P[hashed_s][action] * math.sqrt(sum(self.N[hashed_s])) / (
                        1 + self.N[hashed_s][action])

                if math.isnan(u):
                    logging.warning("u is nan!")

                if u > max_u:
                    max_u, best_action = u, action

        this_action = best_action

        coords = [self._action_to_coords(this_action)]

        next_s = s.copy()
        self.env.move(next_s, coords)
        next_s.rescale()
        next_s = TensorPoints(next_s.get_features())

        current_reward = self._search(next_s, depth + 1)

        self.Q[hashed_s][this_action] = (self.N[hashed_s][this_action] * self.Q[hashed_s][this_action] + current_reward) / (
                    self.N[hashed_s][this_action] + 1)

        assert not math.isnan(self.Q[hashed_s][this_action])

        self.N[hashed_s][this_action] += 1
        self.reward[hashed_s] = current_reward
        return current_reward

class MCTSTrainer:
    def __init__(self, dim = 3, agent = ChooseFirstAgent()):
        self.dim = dim
        self.net = HironakaNet(dim = dim)
        self.agent = agent

    def _loss_function(self,x,y : List[torch.FloatTensor]):
        loss = torch.zeros(1)
        for i, pred in enumerate(x):
            choice_x = torch.narrow(pred, 0, 0, self.net.choices)
            reward_x = torch.narrow(pred, 0, self.net.choices, 1)
            choice_y = torch.narrow(y[i], 0, 0, self.net.choices)
            choice_y = F.softmax(choice_y, dim=0)
            reward_y = torch.narrow(y[i], 0, self.net.choices, 1)
            loss = loss + torch.square((reward_x - reward_y)) - torch.dot(choice_y, (choice_x))

        return loss

    def host_from_nn(self)->trained_host:
        return trained_host(self.net)

    def _arena(self, new_host, old_host:host.Host, steps = 1000, agent = ChooseFirstAgent()) -> bool:
        test_validator = HironakaValidator(new_host, agent, dimension = self.dim)
        new_host_record = test_validator.playoff(num_steps=steps, verbose=0)
        test_validator = HironakaValidator(old_host, agent, dimension = self.dim)
        old_host_record = test_validator.playoff(num_steps=steps, verbose=0)
        return (len(new_host_record) > len(old_host_record))

    def _policy_iter(self, state:TensorPoints, c_puct = 0.5, max_depth = 20):
        #This method returns samples of a single complete game.
        examples = ([],[])

        mcts_instance = MCTS(state = state, env = self.agent, nn=self.net, max_depth = max_depth, c_puct= c_puct)

        depth = 0

        while True:
            mcts_instance.run(iteration = 20, state = state)
            if sum(mcts_instance.N[hashify(state)]) == 0:
                print("No attempt made! Current state:", state)
                print("Current depth:", depth)

            current_sample = mcts_instance.get_sample(state)
            examples[0].append(inverse_hash(hashify(state), dim = self.dim))
            examples[1].append(current_sample)
            best_prob , best_action = 0,-1
            for i,prob in enumerate(current_sample):
                if prob > best_prob:
                    best_prob = prob
                    best_action = i

            coords = mcts_instance._action_to_coords(best_action) #todo: This is ugly. Write a global action decoder.

            self.agent.move(state,[coords])

            state.get_newton_polytope()
            state.rescale()
            state = TensorPoints(state.get_features())

            depth += 1

            if state.ended:
                for sample in examples[1]:
                    sample.append(1)
                break
            elif depth >= max_depth:
                for sample in examples[1]:
                    sample.append(-1)
                break

        return examples

    def train(self, ITERATIONS = 1000, c_puct = 0.5, lr = 1e-6,  use_arena = False):
        for i in range(ITERATIONS):
            while True:
                test_points = TensorPoints(snip.generate_batch_points(n=10, dimension=self.dim, max_value=50),max_num_points=10)
                test_points.get_newton_polytope()
                test_points.rescale()
                test_points = TensorPoints(test_points.get_features())
                if not test_points.ended:
                    break

            examples = self._policy_iter(state = test_points, c_puct = 0.5, max_depth = 20)

            data = [torch.FloatTensor(_) for _ in examples[0]]

            y = [torch.FloatTensor(_) for _ in examples[1]]
            pred = []
            optimizer = torch.optim.SGD(self.net.parameters(), lr = lr, momentum= 0.9)

            for batch, x in enumerate(data):
                this_pred = self.net(x)
                pred.append(this_pred)

            if pred:
                loss = self._loss_function(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i % 10 == 0:
                    print("The current loss is: ", loss.item())

            print("We are in iteration ", i)

    def save_model(self,path:str):
        torch.save(self.net, path)
        print("Saved model as ", path)


if __name__ == '__main__':

    trainer = MCTSTrainer(dim=4, agent = RandomAgent())
    trainer.train(ITERATIONS=100, c_puct=0.5, lr=1e-4)

    path = 'test_model.pth'
    trainer.save_model(path = path)
