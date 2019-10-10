##################################
##### 1. Check numpy, torch  #####
##################################

import numpy
import torch

print('numpy' + numpy.__version__)
print('torch' + torch.__version__)

##################################################################

########################
##### 2. Check gym #####
########################

import gym

env = gym.make('CartPole-v1')

for episode in range(10000):
    done = False
    state = env.reset()

    while not done:
        env.render()

        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

        print('state: {} | action: {} | reward: {} | next_state: {} | done: {}'.format(
                state, action, reward, next_state, done))
        
        state = next_state