import copy
import logging
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
        # # 1 input image channel, 6 output channels, 5x5 square convolution
        # # kernel
        #self.conv1 = nn.Conv2d(1, 6, 5)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        # # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(dim * 10, 64)  # input: all coordinates of Points
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2**dim + 1) # output: there are 8 = 2^3 action of choosing subsets, and the last number is the expected steps

    def forward(self, x):
        # # Max pooling over a (2, 2) window
        #x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # # If the size is a square, you can specify with a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x_prob = torch.narrow(x,0,0,8)
        x_reward = torch.narrow(x,0,8,1)
        x_prob = F.softmax(x_prob,dim = 0)
        x = torch.cat((x_prob,x_reward), dim = 0)
        return x

class trained_host(host.Host):
    def __init__(self,net):
        self.net = net
        super().__init__()

    def _select_coord(self, points: TensorPoints, debug = False):
        answer = []
        if not isinstance(points, TensorPoints):
            points = TensorPoints(points.points, max_num_points = 10)
        for i in range(points.batch_size):
            x = points.points[0]
            result = self.net(x)
            prob_vector = torch.narrow(result,0,0,8)
            prob_vector = prob_vector.tolist()
            reward_vector = torch.narrow(result,0,8,1)
            reward = reward_vector.item()
            current_prob, choice = -float("inf"), -1
            for _,prob in enumerate(prob_vector):
                if prob > current_prob and not _ in {0,1,3,7}:
                    current_prob = prob
                    choice = _

            coords = action_to_coords(choice)
            answer.append(coords)

        return answer

def hashify(s:TensorPoints):
    hashed_str = ""
    current_points = s.points[0].tolist()
    for point in current_points:
        for coord in point:
            hashed_str += '%.10f' % coord
            hashed_str += ','

    return hashed_str

def inverse_hash(s:str):
    x = s.split(",")
    x.pop()
    pt = []
    temp = []
    for i, piece in enumerate(x):
        if i%3 == 0:
            temp = []
        temp.append(float(piece))
        if i%3 == 2:
            pt.append(temp)
    while len(pt) < 10:
        pt.append([0.,0.,0.])

    return pt


def points_to_tensor(s:TensorPoints):
    state = s.points[0]
    coords = []
    for point in state:
        for coord in point:
            coords.append(coord)

    while len(coords) < 30:
        coords.append(0)

    converted = torch.FloatTensor(coords)
    return converted

def action_to_coords(action: int):
    #action is an integer, and coords is a choice of coordinates. We compute the binary expansion of action, and 1 means it choose the corresponding coordidnate.

    current_coord = 0
    coords = []
    action += 1
    while action != 0:
        if action%2:
            coords.append(current_coord)
        current_coord += 1
        action = action // 2
    return coords


class MCTS:
    #todo: add dimensions into mcts.
    def __init__(self, state, env, nn, max_depth = 20, c_puct = 0.5):

        self.initial_state = state if isinstance(state, TensorPoints) else TensorPoints(state.points, max_num_points = 10) #Turn points into PointsTensor
        self.env = env
        self.nn = nn
        self.max_depth = max_depth
        self.P = col.defaultdict()
        self.Q = col.defaultdict()
        self.N = col.defaultdict()
        self.reward = col.defaultdict()
        self.visited = col.defaultdict()
        self.c_puct = c_puct

    def run(self, iteration = 100):
        for _ in range(iteration):
            state = self.initial_state.copy()
            self._search(state)

    def get_examples(self):
        examples = ([],[])
        for _ in self.Q.keys():
            examples[0].append(inverse_hash(_))
            examples[1].append(self.Q[_] + [self.reward[_]])

        return examples


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
            # if it goes too deep, give the host a penalty and move back.
            return 0

        if not (hashed_s in self.visited):
            self.visited[hashed_s] = 1
            result = self.nn(s.points[0])
            # if torch.isnan(result[0]):
            #     logging.warning("nan probablity detected! The current points are")
            #     print(s.points[0])
            self.P[hashed_s] = result[:8]
            current_reward = result[8].item()
            self.Q[hashed_s] = [0 for _ in range(8)]
            self.N[hashed_s] = [0 for _ in range(8)]
            self.reward[hashed_s] = current_reward
            return current_reward

        max_u, best_action = -float("inf"), -1
        # change this part to argmax.
        for action in range(7):
            if (action in {0,1,3}):
                continue
            #Abstract the action choosing function later.
            u = self.Q[hashed_s][action] + self.c_puct * self.P[hashed_s][action] * math.sqrt(sum(self.N[hashed_s])) / (
                        1 + self.N[hashed_s][action])
            if u > max_u:
                max_u, best_action = u, action

        this_action = best_action

        #if I can actively avoid choosing a single coordinate in an elegant way, then it should speed up training dramatically.

        coords = [action_to_coords(this_action)]


        next_s = s.copy()
        self.env.move(next_s, coords)
        next_s.rescale()
        # print("action taken, current length is ", len(s.points[0]))

        current_reward = self._search(next_s, depth + 1)

        self.Q[hashed_s][this_action] = (self.N[hashed_s][this_action] * self.Q[hashed_s][this_action] + current_reward) / (
                    self.N[hashed_s][this_action] + 1)

        self.N[hashed_s][this_action] += 1
        self.reward[hashed_s] = current_reward
        return current_reward

class MCTSTrainer:
    def __init__(self, dim = 3):
        self.dim = dim
        self.net = HironakaNet(dim = dim)

    def _loss_function(self,x,y : List[torch.FloatTensor]):
        loss = torch.zeros(1)
        for i, pred in enumerate(x):
            choice_x = torch.narrow(pred, 0, 0, 8)
            reward_x = torch.narrow(pred, 0, 8, 1)
            choice_y = torch.narrow(y[i], 0, 0, 8)
            choice_y = F.softmax(choice_y, dim=0)
            reward_y = torch.narrow(y[i], 0, 8, 1)
            loss = loss + torch.square((reward_x - reward_y)) - torch.dot(choice_y, (choice_x))

        return loss

    def host_from_nn(self)->trained_host:
        return trained_host(self.net)

    def _arena(self, new_host, old_host:host.Host, steps = 1000, agent = ChooseFirstAgent()) -> bool:
        test_validator = HironakaValidator(new_host, agent)
        new_host_record = test_validator.playoff(num_steps=steps, verbose=0)
        test_validator = HironakaValidator(old_host, agent)
        old_host_record = test_validator.playoff(num_steps=steps, verbose=0)
        return (len(new_host_record) > len(old_host_record))

    def train(self, ITERATIONS = 1000, c_puct = 0.5, lr = 1e-6, agent = ChooseFirstAgent()):
        for i in range(ITERATIONS):
            examples = ([], [])
            for _ in range(10):
                test_points = TensorPoints(snip.generate_batch_points(n=10, dimension=self.dim, max_value=50),
                                           max_num_points=10)
                test_points.get_newton_polytope()
                test_points.rescale()
                mcts_instance = MCTS(state=test_points, env=agent, nn=self.net, max_depth=50, c_puct = c_puct)
                mcts_instance.run(iteration=20)
                new_examples = mcts_instance.get_examples()
                examples[0].extend(new_examples[0])
                examples[1].extend(new_examples[1])
            data, y = examples
            new_data = []
            for points in data:
                temp = []
                for point in points:
                    temp += point
                new_data.append(temp)

            data = [torch.FloatTensor(_) for _ in new_data]
            y = [torch.FloatTensor(_) for _ in y]
            pred = []
            old_model = copy.deepcopy(self.net)
            optimizer = torch.optim.SGD(self.net.parameters(), lr = lr)

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

            if not self._arena(trained_host(self.net), trained_host(old_model), steps = 1000, agent = agent):
                self.net = copy.deepcopy(old_model)
                print("Old model wins!")
            else:
                print("New model wins!")
            print("We are in iteration ", i)

    def save_model(self,path:str):
        torch.save(self.net, path)
        print("Saved model as ", path)


if __name__ == '__main__':

    trainer = MCTSTrainer(dim = 3)
    trainer.train(ITERATIONS= 10, agent = ChooseFirstAgent())

    path = 'test_model.pth'
    trainer.save_model(path = path)