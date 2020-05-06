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
import config
from search import find_path

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def learn(  env, number_timesteps,
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), save_path='./models', save_interval=config.save_interval,
            gamma=config.gamma, grad_norm=config.grad_norm, double_q=config.double_q,
            exploration_final_eps=config.exploration_final_eps, batch_size=config.batch_size, train_freq=config.train_freq,
            learning_starts=config.learning_starts, target_network_update_freq=config.target_network_update_freq, buffer_size=config.buffer_size,
            prioritized_replay=config.prioritized_replay, prioritized_replay_alpha=config.prioritized_replay_alpha,
            prioritized_replay_beta0=config.prioritized_replay_beta0):

    # create network and optimizer
    network = Network()



    # optimizer = Adam(
    #     filter(lambda p: p.requires_grad, network.parameters()),
    #     lr=1e-3, eps=1e-5
    # )

    # create target network
    qnet = network.to(device)
    qnet.load_state_dict(torch.load('./model.pth'))
    # for param in qnet.encoder.parameters():
    #     param.requires_grad = False


    optimizer = Adam(qnet.parameters(), lr=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, 95000, gamma=0.5)

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
            b_obs, b_action, b_reward, b_obs_, b_done, b_steps, *extra = buffer.sample(batch_size)


            with torch.no_grad():
                # choose max q index from next observation
                # double q-learning
                b_action_ = qnet(b_obs_).argmax(1).unsqueeze(1)
                b_q_ = (1 - b_done) * tar_qnet(b_obs_).gather(1, b_action_)


            b_q = qnet(b_obs).gather(1, b_action)

            abs_td_error = (b_q - (b_reward + (gamma ** b_steps) * b_q_)).abs()

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
    explore_steps = (config.exploration_start_eps-exploration_final_eps) / number_timesteps

    obs, pos = env.reset()
    done = False

    # if use imitation learning
    imitation = True if random.random() < config.imitation_ratio else False
    if imitation:
        imitation_actions = find_path(env)

    while imitation and imitation_actions is None:
        o = env.reset()
        imitation_actions = find_path(env)

    o = torch.from_numpy(o).to(device)

    epsilon = config.exploration_start_eps
    for _ in range(1, number_timesteps + 1):

        if imitation:

            a = imitation_actions.pop(0)

        else:
            # sample action
            with torch.no_grad():

                q_val = qnet(torch.from_numpy(obs).to(device), torch.from_numpy(pos).to(device))

                a = q_val.argmax(1).cpu().tolist()

                if random.random() < epsilon:
                    a[np.random.randint(0, 2)] = np.random.randint(0, config.action_space)

        # take action in env
        (next_obs, next_pos), r, done, info = env.step(a)
    

        # return data and update observation

        yield (o[0,:,:,:], a[0], r[0].item(), o_[0,:,:,:], int(done[0]), imitation, info)


        if done[0] == False and env.steps < config.max_steps:

            o = o_ 
        else:
            o = env.reset()
            done = [False for _ in range(env.num_agents)]

            imitation = True if random.random() < config.imitation_ratio else False
            imitation_actions = find_path(env)

            while imitation_actions is None:
                o = env.reset()
                imitation_actions = find_path(env)

            o = torch.from_numpy(o).to(device)
            # if imitation:
            #     print(env.map)
            #     print(env.agents_pos)
            #     print(env.goals_pos)
            #     print(imitation_actions)
        
        epsilon -= explore_steps
            


def huber_loss(abs_td_error):
    flag = (abs_td_error < 1).float()
    return flag * abs_td_error.pow(2) * 0.5 + (1 - flag) * (abs_td_error - 0.5)


if __name__ == '__main__':

    # a = np.array([[[1,2],[3,4]], [[5,6],[7,8]]])
    # print(a[[0,1],[0,1],[0,1]])
    explore_start_eps = 1.0
    explore_final_eps = 0.01

    env = Environment()
    learn(env, 1000000)