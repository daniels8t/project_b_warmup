import gym
import sys
import torch
from train_system import *
from arguments import get_args
from ppo import PPO
from network import FeedForwardNN
from eval_policy import eval_policy

T = np.array([[41002, 43133, 45466, 47962],
              [76220, 79164, 82408, 85859],
              [82260, 83589, 84864, 86039],
              [82917, 84022, 85122, 86219],
              [83125, 84216, 85307, 86399]])

L = np.array([[1940, 3209, 4096, 4770],
              [3521, 5716, 7123, 8063],
              [603, 804, 728, 455],
              [65, 82, 75, 63],
              [20, 31, 37, 40]])

P = np.array([[1940, 2045, 2170, 2312],
              [3521, 3603, 3694, 3789],
              [603, 442, 245, 18],
              [65, 43, 25, 18],
              [20, 19, 18, 18]])


T = np.array([[82044, 86398]])

L = np.array([[6044, 9998]])

P = np.array([[6044, 6371]])

g = Generator(
trains=1,
stations=2,
t_alight_per_person=3,
t_board_per_person=4,
platform_arrivals_per_t=0.1,
alight_fraction=0.4,
number_of_carts=10,
km_between_stations=30,
speed_kmh=100,
stop_t=0,
tmin=180,
train_capacity=10000,
platform_capacity=100000,
var=0
)

env = GymTrainSystem(T, L, P, g)

state = env.reset()
total_reward = 0
for step in range(10):
    action = env.action_space.sample()
    state, reward, done = env.step(action)
    total_reward += reward
    print(state)


