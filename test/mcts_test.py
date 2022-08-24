import unittest
from typing import List

from train import MCTS

from hironaka.validator import HironakaValidator

from hironaka import host

from hironaka.src import _snippets as snip

import torch
from hironaka.core import TensorPoints
from hironaka.agent import RandomAgent, ChooseFirstAgent
from train.MCTS import HironakaNet, MCTS, trained_host

ITERATIONS = 2

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


# class MctsTest(unittest.TestCase):
#     def test_training(self):
#         net = train_network()
#         this_host = trained_host(net)
#         agent = ChooseFirstAgent()
#         test_validator = HironakaValidator(this_host, agent)
#         history = test_validator.playoff(num_steps=1000,verbose= 1)
#         print(len(history))


if __name__ == '__main__':
    path = 'train/test_model.pth'
    net = torch.load(path)
    this_host = trained_host(net)

    agent = ChooseFirstAgent()

    test_validator = HironakaValidator(this_host,agent, dimension = net.dim, step_threshold = 50)
    #Type check failed. Need some minor change on my host class.

    history = test_validator.playoff(num_steps=2000,verbose= 1)

    print(len(history))

    random_host = host.RandomHost()

    test_validator = HironakaValidator(random_host, agent, dimension = net.dim, step_threshold = 50)

    history = test_validator.playoff(num_steps=2000,verbose=1)

    print(len(history))

    Zeillinger = host.Zeillinger()

    test_validator = HironakaValidator(Zeillinger, agent, dimension = net.dim, step_threshold = 50)

    history = test_validator.playoff(num_steps=2000,verbose=1)

    print(len(history))