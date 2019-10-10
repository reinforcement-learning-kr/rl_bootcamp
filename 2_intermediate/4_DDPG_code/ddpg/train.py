import os
import gym
import time
import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import *
from replay_buffer import ReplayBuffer

parser = argparse.ArgumentParser()
parser.add_argument('--training_eps', type=int, default=500)
parser.add_argument('--eval_per_train', type=int, default=50)
parser.add_argument('--evaluation_eps', type=int, default=100)
parser.add_argument('--threshold_return', type=int, default=-230)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--act_noise', type=float, default=0.2)
parser.add_argument('--buffer_size', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--actor_lr', type=float, default=1e-4)
parser.add_argument('--critic_lr', type=float, default=3e-3)
parser.add_argument('--gradient_clip_ac', type=float, default=0.5)
parser.add_argument('--gradient_clip_cr', type=float, default=1.0)
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def select_action(obs, act_dim, act_limit, actor):
    action = actor(obs).detach().cpu().numpy()
    action += args.act_noise * np.random.randn(act_dim)
    return np.clip(action, -act_limit, act_limit)

def hard_target_update(main, target):
    target.load_state_dict(main.state_dict())

def soft_target_update(main, target, tau=0.005):
    for main_param, target_param in zip(main.parameters(), target.parameters()):
        target_param.data.copy_(tau*main_param.data + (1.0-tau)*target_param.data)

def train_model(actor, actor_target, critic, critic_target, 
                actor_optimizer, critic_optimizer, batch):
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

    # Actor prediction Q(s,π(s))
    pi = actor(obs1)
    q_pi = critic(obs1, pi)

    # Critic prediction Q(s,a), Q‾(s',π‾(s'))
    q = critic(obs1, acts).squeeze(1)
    pi_target = actor_target(obs2)
    q_pi_target = critic_target(obs2, pi_target).squeeze(1)

    # Target for Q regression
    q_backup = rews + args.gamma*(1-done)*q_pi_target
    q_backup.to(device)

    if 0: # Check shape of prediction and target
        print("q", q.shape)
        print("q_backup", q_backup.shape)

    # DDPG losses
    actor_loss = -q_pi.mean()
    critic_loss = F.mse_loss(q, q_backup.detach())

    # Update critic network parameter
    critic_optimizer.zero_grad()
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), args.gradient_clip_cr)
    critic_optimizer.step()

    # Update actor network parameter
    actor_optimizer.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), args.gradient_clip_ac)
    actor_optimizer.step()

    # Polyak averaging for target parameter
    soft_target_update(actor, actor_target)
    soft_target_update(critic, critic_target)

def main():
    # Initialize environment
    env = gym.make('Pendulum-v0')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    print('State dimension:', obs_dim)
    print('Action dimension:', act_dim)

    # Set a random seed
    env.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Create a SummaryWriter object by TensorBoard
    dir_name = 'runs/' + 'Pendulum-v0' + '_' + time.ctime()
    writer = SummaryWriter(log_dir=dir_name)

    # Main network
    actor = MLP(obs_dim, act_dim, output_activation=torch.tanh).to(device)
    critic = FlattenMLP(obs_dim+act_dim, 1).to(device)
    # Target network
    actor_target = MLP(obs_dim, act_dim, output_activation=torch.tanh).to(device)
    critic_target = FlattenMLP(obs_dim+act_dim, 1).to(device)

    # Initialize target parameters to match main parameters
    hard_target_update(actor, actor_target)
    hard_target_update(critic, critic_target)

    # Create optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.critic_lr)
    
    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim, act_dim, args.buffer_size)
    
    def run_one_episode(steps, eval_mode):
        total_reward = 0.

        obs = env.reset()
        done = False

        # Keep interacting until agent reaches a terminal state.
        while not done:
            steps += 1 

            if eval_mode:
                action = actor(torch.Tensor(obs).to(device))
                action = action.detach().cpu().numpy()
                next_obs, reward, done, _ = env.step(action)
            else:
                # Collect experience (s, a, r, s') using some policy
                action = select_action(torch.Tensor(obs).to(device), act_dim, act_limit, actor)
                next_obs, reward, done, _ = env.step(action)

                # Add experience to replay buffer
                replay_buffer.add(obs, action, reward, next_obs, done)
            
                # Start training when the number of experience is greater than batch size
                if steps > args.batch_size:
                    batch = replay_buffer.sample(args.batch_size)
                    train_model(actor, actor_target, critic, critic_target, 
                                actor_optimizer, critic_optimizer, batch)
            
            total_reward += reward
            obs = next_obs
        return steps, total_reward

    train_sum_returns = 0.
    train_num_episodes = 0

    start_time = time.time()
    steps = 0
    
    for episode in range(1, args.training_eps+1):
        # Perform the training phase, during which the agent learns
        eval_mode = False

        # Run one episode
        steps, train_episode_return = run_one_episode(steps, eval_mode)

        train_sum_returns += train_episode_return
        train_num_episodes += 1

        train_average_return = train_sum_returns / train_num_episodes if train_num_episodes > 0 else 0.0

        # Log experiment result for training episodes
        writer.add_scalar('Train/AverageReturns', train_average_return, episode)
        writer.add_scalar('Train/EpisodeReturns', train_sum_returns, episode)
        
        # Perform the evaluation phase -- no learning
        if episode > 0 and episode % args.eval_per_train == 0:
            eval_mode = True
            
            eval_sum_returns = 0.
            eval_num_episodes = 0

            for _ in range(args.evaluation_eps):
                # Run one episode
                steps, eval_episode_return = run_one_episode(steps, eval_mode)

                eval_sum_returns += eval_episode_return
                eval_num_episodes += 1

                eval_average_return = eval_sum_returns / eval_num_episodes if eval_num_episodes > 0 else 0.0

                # Log experiment result for evaluation episodes
                writer.add_scalar('Eval/AverageReturns', eval_average_return, episode)
                writer.add_scalar('Eval/EpisodeReturns', eval_episode_return, episode)

            print('---------------------------------------')
            print('Episodes:', episode)
            print('AverageReturn:', round(train_average_return, 2))
            print('EvalEpisodes:', eval_num_episodes)
            print('EvalAverageReturn:', round(eval_average_return, 2))
            print('Time:', int(time.time() - start_time))
            print('---------------------------------------')

            # Save a training model
            if eval_average_return >= args.threshold_return:
                if not os.path.exists('./save_model'):
                    os.mkdir('./save_model')

                ckpt_path = os.path.join('./save_model/' + 'Pendulum-v0_ddpg' + '_ep_' + str(episode) \
                                                                              + '_rt_' + str(round(eval_average_return, 2)) \
                                                                              + '_t_' + str(int(time.time() - start_time)) + '.pt')
                torch.save(actor.state_dict(), ckpt_path)
                break  

if __name__ == '__main__':
    main()