import unittest
from typing import List

from train import MCTS

from hironaka.validator import HironakaValidator

from hironaka import host

from hironaka.src import _snippets as geom

import torch
from hironaka.core import Points
from hironaka.agent import RandomAgent, ChooseFirstAgent
from train.MCTS import HironakaNet, MCTS

ITERATIONS = 50

def loss_function(x,y : List[torch.FloatTensor])->torch.Tensor:
    loss = torch.zeros(1)
    for i,pred in enumerate(x):
        choice_x = torch.narrow(pred,0,0,8)
        reward_x = torch.narrow(pred,0,8,1)
        choice_y = torch.narrow(y[i],0,0,8)
        reward_y = torch.narrow(y[i],0,8,1)
        loss = loss + torch.square((reward_x - reward_y)) - torch.dot(choice_y,(choice_x))

    return loss

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

#TODO: Use PointTensor in both MCTS and test.

def points_to_tensor(s:Points):
    state = s.points[0]
    coords = []
    for point in state:
        for coord in point:
            coords.append(coord)

    while len(coords) < 30:
        coords.append(0)

    converted = torch.FloatTensor(coords)
    return converted

class trained_host(host.Host):
    def __init__(self,net):
        self.net = net

    def select_coord(self, points: Points, debug = False):
        answer = []
        for i in range(points.batch_size):
            x = points_to_tensor(Points([points.points[i]]))
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


def train_network():
    net = HironakaNet()

    coords = action_to_coords(3)
    agent = ChooseFirstAgent()
    agent2 = RandomAgent()

    #train a network here.

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-6)
    for i in range(ITERATIONS):
        examples = ([], [])
        for _ in range(10):
            test_points = Points(geom.generate_batch_points(n=10, batch_num=1, dimension=3, max_value=50))
            test_points.get_newton_polytope()
            test_points.rescale()
            mcts_instance = MCTS(state=test_points, env=agent, nn=net, max_depth=50)
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
        for batch, x in enumerate(data):
            this_pred = net(x)
            pred.append(this_pred)

        if pred:
            loss = loss_function(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print("The current loss is: ", loss.item())

    return net

class MctsTest(unittest.TestCase):
    def test_training(self):
        net = train_network()
        this_host = trained_host(net)
        agent = ChooseFirstAgent()
        test_validator = HironakaValidator(this_host, agent)
        history = test_validator.playoff(num_steps=1000,verbose= 1)
        print(len(history))


if __name__ == '__main__':
    path = 'test_model.pth'
    net = torch.load(path)
    this_host = trained_host(net)

    agent = ChooseFirstAgent()

    test_validator = HironakaValidator(this_host,agent)
    #Type check failed. Need some minor change on my host class.

    history = test_validator.playoff(num_steps=1000,verbose= 1)

    print(len(history))

    random_host = host.RandomHost()

    test_validator = HironakaValidator(random_host, agent)

    history = test_validator.playoff(num_steps=1000,verbose=1)

    print(len(history))

    Zeillinger = host.Zeillinger()

    test_validator = HironakaValidator(Zeillinger, agent)

    history = test_validator.playoff(num_steps=1000,verbose=1)

    print(len(history))