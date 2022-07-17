from hironaka.mcts import MCTS

from hironaka.validator import HironakaValidator

from hironaka import host

hironaka_net = MCTS.hironaka_net

import torch
from hironaka.core import Points
from hironaka.agent import RandomAgent, ChooseFirstAgent

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

class trained_host:
    def __init__(self,path):
        self.net = torch.load(path)

    def select_coord(self, points: Points, debug = False):
        answer = []
        for i in range(points.batch_size):
            x = MCTS.points_to_tensor(Points([points.points[i]]))
            result = self.net(x)
            prob_vector = torch.narrow(result,0,0,8)
            prob_vector = prob_vector.tolist()
            reward_vector = torch.narrow(result,0,8,1)
            reward = reward_vector.item()
            print(prob_vector)
            current_prob, choice = -float("inf"), -1
            for _,prob in enumerate(prob_vector):
                if prob > current_prob and not _ in {0,1,3,7}:
                    current_prob = prob
                    choice = _

            coords = action_to_coords(choice)
            print(coords)
            answer.append(coords)

        return answer

if __name__ == '__main__':
    path = 'test_model.pth'
    this_host = trained_host(path)

    agent = RandomAgent()

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