import os
import gym
import argparse
import numpy as np
import torch
from model import *

# Configurations
parser = argparse.ArgumentParser()
parser.add_argument('--load', type=str, default=None,
                    help='load the saved model')
parser.add_argument('--render', action="store_true", default=True,
                    help='if you want to render, set this to True')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    env = gym.make('Pendulum-v0')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    mlp = GaussianPolicy(obs_dim, act_dim).to(device)

    if args.load is not None:
        pretrained_model_path = os.path.join('./save_model/' + str(args.load))
        pretrained_model = torch.load(pretrained_model_path)
        mlp.load_state_dict(pretrained_model)

    sum_returns = 0.
    num_episodes = 0

    for episode in range(1, 10001):
        total_reward = 0.

        obs = env.reset()
        done = False

        while not done:
            if args.render:
                env.render()
            
            action, _, _ = mlp(torch.Tensor(obs).to(device))
            action = action.detach().cpu().numpy()
            next_obs, reward, done, _ = env.step(action)
            
            total_reward += reward
            obs = next_obs
        
        sum_returns += total_reward
        num_episodes += 1

        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0

        if episode % 10 == 0:
            print('---------------------------------------')
            print('Episodes:', num_episodes)
            print('AverageReturn:', average_return)
            print('---------------------------------------')

if __name__ == "__main__":
    main()
