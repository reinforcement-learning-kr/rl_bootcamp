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
parser.add_argument('--alpha', type=float, default=0.05)
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True)
parser.add_argument('--buffer_size', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--actor_lr', type=float, default=1e-4)
parser.add_argument('--qf_lr', type=float, default=3e-3)
parser.add_argument('--alpha_lr', type=float, default=1e-4)
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hard_target_update(main, target):
    target.load_state_dict(main.state_dict())

def soft_target_update(main, target, tau=0.005):
    for main_param, target_param in zip(main.parameters(), target.parameters()):
        target_param.data.copy_(tau*main_param.data + (1.0-tau)*target_param.data)

def train_model(actor, qf1, qf2, qf1_target, qf2_target,
                actor_optimizer, qf1_optimizer, qf2_optimizer, 
                batch, target_entropy, log_alpha, alpha_optimizer):
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

    # Prediction π(s), logπ(s), π(s'), logπ(s'), Q1(s,a), Q2(s,a)
    _, pi, log_pi = actor(obs1)
    _, next_pi, next_log_pi = actor(obs2)
    q1 = qf1(obs1, acts).squeeze(1)
    q2 = qf2(obs1, acts).squeeze(1)

    # Min Double-Q: min(Q1(s,π(s)), Q2(s,π(s))), min(Q1‾(s',π(s')), Q2‾(s',π(s')))
    min_q_pi = torch.min(qf1(obs1, pi), qf2(obs1, pi)).squeeze(1).to(device)
    min_q_next_pi = torch.min(qf1_target(obs2, next_pi), qf2_target(obs2, next_pi)).squeeze(1).to(device)

    # Targets for Q and V regression
    v_backup = min_q_next_pi - args.alpha*next_log_pi
    q_backup = rews + args.gamma*(1-done)*v_backup
    q_backup.to(device)

    if 0: # Check shape of prediction and target
        print("log_pi", log_pi.shape)
        print("next_log_pi", next_log_pi.shape)
        print("q1", q1.shape)
        print("q2", q2.shape)
        print("min_q_pi", min_q_pi.shape)
        print("min_q_next_pi", min_q_next_pi.shape)
        print("q_backup", q_backup.shape)

    # Soft actor-critic losses
    actor_loss = (args.alpha*log_pi - min_q_pi).mean()
    qf1_loss = F.mse_loss(q1, q_backup.detach())
    qf2_loss = F.mse_loss(q2, q_backup.detach())

    # Update two Q network parameter
    qf1_optimizer.zero_grad()
    qf1_loss.backward()
    qf1_optimizer.step()

    qf2_optimizer.zero_grad()
    qf2_loss.backward()
    qf2_optimizer.step()
    
    # Update actor network parameter
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # If automatic entropy tuning is True, update alpha
    if args.automatic_entropy_tuning:
        alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()

        args.alpha = log_alpha.exp()

    # Polyak averaging for target parameter
    soft_target_update(qf1, qf1_target)
    soft_target_update(qf2, qf2_target)

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
    actor = GaussianPolicy(obs_dim, act_dim).to(device)
    qf1 = FlattenMLP(obs_dim+act_dim, 1).to(device)
    qf2 = FlattenMLP(obs_dim+act_dim, 1).to(device)
    # Target network
    qf1_target = FlattenMLP(obs_dim+act_dim, 1).to(device)
    qf2_target = FlattenMLP(obs_dim+act_dim, 1).to(device)

    # Initialize target parameters to match main parameters
    hard_target_update(qf1, qf1_target)
    hard_target_update(qf2, qf2_target)

    # Create optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.actor_lr)
    qf1_optimizer = optim.Adam(qf1.parameters(), lr=args.qf_lr)
    qf2_optimizer = optim.Adam(qf2.parameters(), lr=args.qf_lr)
    
    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim, act_dim, args.buffer_size)
    
    # If automatic entropy tuning is True, initialize a target entropy, a log alpha and an alpha optimizer
    if args.automatic_entropy_tuning:
        target_entropy = -np.prod((act_dim,)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)
    else:
        target_entropy = None
        log_alpha = None
        alpha_optimizer = None

    def run_one_episode(steps, eval_mode):
        total_reward = 0.

        obs = env.reset()
        done = False

        # Keep interacting until agent reaches a terminal state.
        while not done:
            steps += 1 

            if eval_mode:
                action, _, _ = actor(torch.Tensor(obs).to(device))
                action = action.detach().cpu().numpy()
                next_obs, reward, done, _ = env.step(action)
            else:
                # Collect experience (s, a, r, s') using some policy
                _, action, _ = actor(torch.Tensor(obs).to(device))
                action = action.detach().cpu().numpy()
                next_obs, reward, done, _ = env.step(action)

                # Add experience to replay buffer
                replay_buffer.add(obs, action, reward, next_obs, done)
            
                # Start training when the number of experience is greater than batch size
                if steps > args.batch_size:
                    batch = replay_buffer.sample(args.batch_size)
                    train_model(actor, qf1, qf2, qf1_target, qf2_target, 
                                actor_optimizer, qf1_optimizer, qf2_optimizer,
                                batch, target_entropy, log_alpha, alpha_optimizer)
            
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

                ckpt_path = os.path.join('./save_model/' + 'Pendulum-v0_sac' + '_ep_' + str(episode) \
                                                                              + '_rt_' + str(round(eval_average_return, 2)) \
                                                                              + '_t_' + str(int(time.time() - start_time)) + '.pt')
                torch.save(actor.state_dict(), ckpt_path)
                break  

if __name__ == '__main__':
    main()