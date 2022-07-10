import torch
import torch.nn as nn
import torch.nn.functional as F
from hironaka.abs import Points
from hironaka.agent import RandomAgent
from hironaka.envs import HironakaAgentEnv

import collections as col
import math

c_puct = 0.5 #explore hyperparameter
current_enivronment = RandomAgent

#WARNING: this only works for 1st batch, dim = 3 and maximal 10 points for now!


class hironaka_net(nn.Module):

    def __init__(self):
        super(hironaka_net, self).__init__()
        # # 1 input image channel, 6 output channels, 5x5 square convolution
        # # kernel
        # self.conv1 = nn.Conv2d(1, 6, 5)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(3 * 10, 20)  # input: all coordinates of Points
        self.fc2 = nn.Linear(20, 9) # output: there are 8 = 2^3 action of choosing subsets, and the last number is the expected steps

    def forward(self, x):
        # # Max pooling over a (2, 2) window
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # # If the size is a square, you can specify with a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x

def hashify(s:Points):
    hashed_str = ""
    for point in s.points[0]:
        for coord in point:
            hashed_str += '%.3f' % coord
        hashed_str+= ','

    return hashed_str

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

def mcts(s:Points, env, nn:hironaka_net, visited,P,Q,N):
    # s: game state, env: game environment, nn: the fixed neural network, visited,P,Q,N are parameters computed by MCTS.
    if len(s.points[0]) == 1:
        return 1

    hashed_s = hashify(s)
    if not (hashed_s in visited):
        visited[hashed_s] = 1
        result = nn(points_to_tensor(s))
        P[hashed_s] = result[:8]
        v = result[8]
        Q[s] = [0 for _ in range(8)]
        return v + 1

    max_u, best_action = -float("inf"), -1
    for action in range(8):
        u = Q[hashed_s][action] + c_puct*P[hashed_s][action]*math.sqrt(sum(N[hashed_s]))/(1+N[hashed_s][action])
        if u > max_u:
            max_u, best_action = u, action

    this_action = best_action

    new_state = env.step(s,this_action) #This part is still completely broken!!!
    v = mcts(new_state,env, nn, visited, P,Q,N)

    Q[hashed_s][this_action] = (N[hashed_s][this_action]*Q[hashed_s][this_action] + v)/(N[hashed_s][this_action] + 1)
    N[hashed_s][this_action] += 1

    return v+1

if __name__ == '__main__':
    net = hironaka_net()

    test_points = Points([[[1,1,1]]])
    P = col.defaultdict()
    Q = col.defaultdict()
    N = col.defaultdict()
    visited = col.defaultdict()
    v = mcts(test_points, RandomAgent, net, visited = visited, P = P,Q = Q,N = N)
    print(visited)

    print(v)