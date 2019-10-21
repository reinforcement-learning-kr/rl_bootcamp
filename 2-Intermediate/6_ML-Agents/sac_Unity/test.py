import os
import gym
import argparse
import numpy as np
import torch
from model import *
from mlagents.envs import UnityEnvironment

# Configurations
parser = argparse.ArgumentParser()
parser.add_argument('--load', type=str, default="Hopper_ep_480_rt_26.04_t_10116.pt",
                    help='load the saved model')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    env = UnityEnvironment(file_name='../env/Hopper/Hopper')

    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]

    env_info = env.reset(train_mode=False)[default_brain]

    obs_dim = env_info.vector_observations[0].shape[0]
    act_dim = brain.vector_action_space_size[0]

    mlp = GaussianPolicy(obs_dim, act_dim).to(device)

    if args.load is not None:
        pretrained_model_path = os.path.join('./save_model/' + str(args.load))
        pretrained_model = torch.load(pretrained_model_path)
        mlp.load_state_dict(pretrained_model)

    sum_returns = 0.
    num_episodes = 0

    for episode in range(1, 10001):
        total_reward = 0.

        obs = env_info.vector_observations[0]
        done = False

        while not done:           
            action, _, _ = mlp(torch.Tensor(obs).to(device))
            action = action.detach().cpu().numpy()
            env_info = env.step(action)[default_brain]

            next_obs = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]         

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
