import os
import time
import argparse
from collections import deque
from mlagents.envs import UnityEnvironment

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter

from model import MLP
from replay_buffer import ReplayBuffer

parser = argparse.ArgumentParser()
parser.add_argument('--episode_num', type=int, default=500)
parser.add_argument('--threshold_return', type=int, default=495)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--epsilon', type=float, default=1.0)
parser.add_argument('--epsilon_decay', type=float, default=0.995)
parser.add_argument('--buffer_size', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--target_update_steps', type=int, default=100)
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def select_action(obs, act_num, qf):
    # Decaying epsilon
    args.epsilon *= args.epsilon_decay
    args.epsilon = max(args.epsilon, 0.01)

    if np.random.rand() <= args.epsilon:
        # Choose a random action with probability epsilon
        return np.random.randint(act_num)
    else:
        # Choose the action with highest Q-value at the current state
        action = qf(obs).argmax()
        return action.detach().cpu().numpy()

def train_model(qf, qf_target, qf_optimizer, batch, step_count):
    obs1 = batch['obs1']
    obs2 = batch['obs2']
    acts = batch['acts']
    rews = batch['rews']
    done = batch['done']

    if 0: # Check shape of experiences
        print("obs1", obs1.shape)
        print("obs2", obs2.shape)
        print("acts", acts.shape)
        print("rews", rews.shape)
        print("done", done.shape)

    # Prediction Q(s), Q‾(s')
    q = qf(obs1).gather(1, acts.long()).squeeze(1)
    q_target = qf_target(obs2)

    # Target for Q regression
    q_backup = rews + args.gamma*(1-done)*q_target.max(1)[0]
    q_backup.to(device)

    if 0: # Check shape of prediction and target
        print("q", q.shape)
        print("q_backup", q_backup.shape)

    # Update perdiction network parameter
    qf_loss = F.mse_loss(q, q_backup.detach())
    qf_optimizer.zero_grad()
    qf_loss.backward()
    qf_optimizer.step()

    # Synchronize target parameters 𝜃‾ as 𝜃 every N steps
    if step_count % args.target_update_steps == 0:
        qf_target.load_state_dict(qf.state_dict())

def main():
    # Initialize environment
    env = UnityEnvironment(file_name='../../env/Pong/Pong')

    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]

    env_info = env.reset(train_mode=True)[default_brain]

    obs_dim = env_info.vector_observations[0].shape[0]
    act_num = brain.vector_action_space_size[0]
    print('State dimension:', obs_dim)
    print('Action number:', act_num)

    # Set a random seed
    np.random.seed(0)
    torch.manual_seed(0)

    # Create a SummaryWriter object by TensorBoard
    dir_name = 'runs/' + 'dqn/' + 'CartPole-v1' + '_' + time.ctime()
    # writer = SummaryWriter(log_dir=dir_name)

    # Main network
    qf = MLP(obs_dim, act_num).to(device)
    # Target network
    qf_target = MLP(obs_dim, act_num).to(device)

    # Initialize target parameters to match main parameters
    qf_target.load_state_dict(qf.state_dict())

    # Create an optimizer
    qf_optimizer = optim.Adam(qf.parameters(), lr=1e-3)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim, 1, args.buffer_size)
    
    step_count = 0
    sum_returns = 0.
    num_episodes = 0
    recent_returns = deque(maxlen=10)

    start_time = time.time()
    
    for episode in range(1, args.episode_num+1):
        total_reward = 0.

        env_info = env.reset(train_mode=True)[default_brain]
        obs = env_info.vector_observations[0]
        done = False

        # Keep interacting until agent reaches a terminal state.
        while not done:
            step_count += 1 

            # Collect experience (s, a, r, s') using some policy
            action = select_action(torch.Tensor(obs).to(device), act_num, qf)

            env_info = env.step(int(action))[default_brain]

            next_obs = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            # Add experience to replay buffer
            replay_buffer.add(obs, action, reward, next_obs, done)

            # Start training when the number of experience is greater than batch size
            if step_count > args.batch_size:
                batch = replay_buffer.sample(args.batch_size)
                train_model(qf, qf_target, qf_optimizer, batch, step_count)
            
            total_reward += reward
            obs = next_obs
        
        recent_returns.append(total_reward)
        sum_returns += total_reward
        num_episodes += 1
        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0

        # # Log experiment result for training episodes
        # writer.add_scalar('Train/AverageReturns', average_return, episode)
        # writer.add_scalar('Train/EpisodeReturns', sum_returns, episode)
        
        if episode % 10 == 0:
            print('---------------------------------------')
            print('Episodes:', episode)
            print('Steps:', step_count)
            print('AverageReturn:', round(average_return, 2))
            print('RecentReturn:', np.mean(recent_returns))
            print('Time:', int(time.time() - start_time))
            print('---------------------------------------')

        # Save a training model
        if (np.mean(recent_returns)) >= args.threshold_return:
            print('Recent returns {} exceed threshold return. So end'.format(np.mean(recent_returns)))
            if not os.path.exists('./save_model'):
                os.mkdir('./save_model')

            ckpt_path = os.path.join('./save_model/' + 'Pong_dqn' + '_rt_' + str(round(average_return, 2)) \
                                                                  + '_ep_' + str(episode) \
                                                                  + '_t_' + str(int(time.time() - start_time)) + '.pt')
            torch.save(qf.state_dict(), ckpt_path)
            break  

if __name__ == '__main__':
    main()