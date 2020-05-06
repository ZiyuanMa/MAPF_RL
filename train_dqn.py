import torch.nn as nn
from torch.optim import Adam, lr_scheduler
import math
import os
import random
import time
from collections import deque
from copy import deepcopy
 
import numpy as np
import torch
import argparse

from buffer import ReplayBuffer, PrioritizedReplayBuffer
from model_dqn import Network
from environment import Environment
from search import find_path

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def learn(  env, number_timesteps,
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), save_path='./models', save_interval=500000,
            gamma=0.99, grad_norm=10,
            exploration_final_eps=0.01, batch_size=32, train_freq=4,
            learning_starts=20000, target_network_update_freq=4000, buffer_size=50000,
            prioritized_replay=True, prioritized_replay_alpha=0.6,
            prioritized_replay_beta0=0.4):

    # create network and optimizer
    network = Network()

    # create target network
    qnet = network.to(device)


    optimizer = Adam(qnet.parameters(), lr=2.5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, 200000, gamma=0.5)

    tar_qnet = deepcopy(qnet)

    # create replay buffer
    if prioritized_replay:
        buffer = PrioritizedReplayBuffer(buffer_size, device,
                                         prioritized_replay_alpha,
                                         prioritized_replay_beta0)
    else:
        buffer = ReplayBuffer(buffer_size, device)

    generator = _generate(device, env, qnet,
                          number_timesteps,
                          exploration_final_eps)

    start_ts = time.time()
    for n_iter in range(1, number_timesteps + 1):

        if prioritized_replay:
            buffer.beta += (1 - prioritized_replay_beta0) / number_timesteps
            
        data = generator.__next__()
        buffer.add(data)


        # update qnet
        if n_iter > learning_starts and n_iter % train_freq == 0:
            b_obs, b_pos, b_action, b_reward, b_next_obs, b_next_pos, b_done, b_steps, *extra = buffer.sample(batch_size)


            with torch.no_grad():
                # choose max q index from next observation
                # double q-learning
                b_action_ = qnet(b_next_obs, b_next_pos).argmax(2).unsqueeze(2)
                b_q_ = (1 - b_done) * tar_qnet(b_next_obs, b_next_pos).gather(2, b_action_)


            b_q = qnet(b_obs, b_pos).gather(2, b_action)

            abs_td_error = (b_q - (b_reward + (gamma ** b_steps) * b_q_)).abs().mean(1)

            priorities = abs_td_error.detach().cpu().clamp(1e-6).numpy()

            if extra:
                loss = (extra[0] * huber_loss(abs_td_error)).mean()
            else:
                loss = huber_loss(abs_td_error).mean()


            optimizer.zero_grad()

            loss.backward()
            if grad_norm is not None:
                nn.utils.clip_grad_norm_(qnet.parameters(), grad_norm)

            optimizer.step()

            scheduler.step()

            # soft update
            # for tar_net, net in zip(tar_qnet.parameters(), qnet.parameters()):
            #     tar_net.data.copy_(0.001*net.data + 0.999*tar_net.data)

            if prioritized_replay:
                buffer.update_priorities(extra[1], priorities)

        # update target net and log
        if n_iter % target_network_update_freq == 0:
            tar_qnet.load_state_dict(qnet.state_dict())

            print('{} Iter {} {}'.format('=' * 10, n_iter, '=' * 10))
            fps = int(target_network_update_freq / (time.time() - start_ts))
            start_ts = time.time()
            print('FPS {}'.format(fps))

            if n_iter > learning_starts and n_iter % train_freq == 0:
                print('vloss: {:.6f}'.format(loss.item()))
            

        if save_interval and n_iter % save_interval == 0:
            torch.save(qnet.state_dict(), os.path.join(save_path, '{}.pth'.format(n_iter)))


def _generate(device, env, qnet,
              number_timesteps,
            exploration_final_eps):

    """ Generate training batch sample """
    explore_steps = (1.0-exploration_final_eps) / number_timesteps

    obs_pos = env.reset()
    done = False

    # if use imitation learning
    imitation = True if random.random() < 0.3 else False
    if imitation:
        imitation_actions = find_path(env)

    while imitation and imitation_actions is None:
        obs_pos = env.reset()
        imitation_actions = find_path(env)

    epsilon = 1.0
    for _ in range(1, number_timesteps + 1):

        if imitation:

            a = imitation_actions.pop(0)

        else:
            # sample action
            with torch.no_grad():

                q_val = qnet(torch.from_numpy(obs_pos[0]).to(device), torch.from_numpy(obs_pos[1]).to(device))

                actions = q_val.argmax(1).cpu().tolist()

                for i in len(actions):
                    if random.random() < epsilon:
                        actions[i] = np.random.randint(0, 5)

        # take action in env
        next_obs_pos, r, done, info = env.step(actions)
    

        # return data and update observation

        yield (obs_pos, actions, r, next_obs_pos, int(done), imitation, info)


        if done == False and env.steps < 200:

            obs_pos = next_obs_pos 
        else:
            obs_pos = env.reset()
            done = False

            imitation = True if random.random() < 0.3 else False
            imitation_actions = find_path(env)

            while imitation_actions is None:
                obs_pos = env.reset()
                imitation_actions = find_path(env)

        
        epsilon -= explore_steps
            


def huber_loss(abs_td_error):
    flag = (abs_td_error < 1).float()
    return flag * abs_td_error.pow(2) * 0.5 + (1 - flag) * (abs_td_error - 0.5)


if __name__ == '__main__':

    # a = np.array([[[1,2],[3,4]], [[5,6],[7,8]]])
    # print(a[[0,1],[0,1],[0,1]])
    explore_start_eps = 1.0
    explore_final_eps = 0.01

    imitation_ratio = 0.3

    env = Environment()
    learn(env, 2000000)