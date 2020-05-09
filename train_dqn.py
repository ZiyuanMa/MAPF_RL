import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
import os
import random
import time
from copy import deepcopy
import numpy as np
import argparse

from buffer import PrioritizedReplayBuffer
from model_dqn import Network
from environment import Environment
from search import find_path
import config

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def learn(  env=Environment(), training_timesteps=config.training_timesteps,
            explore_start_eps=config.explore_start_eps, explore_final_eps=config.explore_final_eps,
            save_path=config.save_path, save_interval=config.save_interval,
            gamma=config.gamma, grad_norm=config.grad_norm_dqn,
            batch_size=config.batch_size_dqn, train_freq=config.train_freq,
            learning_starts=config.learning_starts, target_network_update_freq=config.target_network_update_freq,
            buffer_size=config.buffer_size, max_steps=config.max_steps, imitation_ratio=config.imitation_ratio,
            prioritized_replay_alpha=config.prioritized_replay_alpha, prioritized_replay_beta=config.prioritized_replay_beta):

    # create network
    qnet = Network().to(device)

    optimizer = Adam(qnet.parameters(), lr=2.5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, 200000, gamma=0.5)

    # create target network
    tar_qnet = deepcopy(qnet)

    # create replay buffer
    buffer = PrioritizedReplayBuffer(buffer_size, device, prioritized_replay_alpha, prioritized_replay_beta)

    generator = _generate(env, qnet, training_timesteps, max_steps, imitation_ratio, explore_start_eps, explore_final_eps)

    start_ts = time.time()
    for n_iter in range(1, training_timesteps + 1):

        buffer.beta += (1 - prioritized_replay_beta) / training_timesteps
            
        data = generator.__next__()
        buffer.add(data)

        # update qnet
        if n_iter > learning_starts and n_iter % train_freq == 0:
            b_obs, b_pos, b_action, b_reward, b_next_obs, b_next_pos, b_done, b_steps, b_bt_steps, b_next_bt_steps, *extra = buffer.sample(batch_size)


            with torch.no_grad():
                # choose max q index from next observation
                # double q-learning
                b_action_ = qnet.bootstrap(b_next_obs, b_next_pos, b_next_bt_steps).argmax(1, keepdim=True)
                b_q_ = (1 - b_done) * tar_qnet.bootstrap(b_next_obs, b_next_pos, b_next_bt_steps).gather(1, b_action_)


            b_q = qnet.bootstrap(b_obs, b_pos, b_bt_steps).gather(1, b_action)

            abs_td_error = (b_q - (b_reward + (gamma ** b_steps) * b_q_)).abs().reshape(32, 2).mean(dim=1, keepdim=True)

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
            
        # save model
        if save_interval and n_iter % save_interval == 0:
            torch.save(qnet.state_dict(), os.path.join(save_path, '{}.pth'.format(n_iter)))

    torch.save(qnet.state_dict(), os.path.join(save_path, 'model.pth'))


def _generate(env, qnet,
            training_timesteps, max_steps, imitation_ratio,
            explore_start_eps, exploration_final_eps):

    """ Generate training batch sample """
    explore_delta = (explore_start_eps-exploration_final_eps) / training_timesteps

    obs_pos = env.reset()
    done = False

    # if use imitation learning
    imitation = True if random.random() < imitation_ratio else False
    imitation_actions = find_path(env) if imitation else None

    # if no solution, reset environment
    while imitation and imitation_actions is None:
        obs_pos = env.reset()
        imitation_actions = find_path(env)

    epsilon = explore_start_eps
    for _ in range(1, training_timesteps + 1):

        if imitation:

            actions = imitation_actions.pop(0)

        else:
            # sample action
            with torch.no_grad():

                q_val = qnet.step(torch.from_numpy(obs_pos[0]).to(device), torch.from_numpy(obs_pos[1]).to(device))

                actions = q_val.argmax(1).cpu().tolist()

                # Epsilon Greedy
                for i in range(len(actions)):
                    if random.random() < epsilon:
                        actions[i] = np.random.randint(0, 5)

        # take action in env
        next_obs_pos, r, done, info = env.step(actions)
    

        # return data and update observation

        yield (obs_pos, actions, r, next_obs_pos, int(done), imitation, info)


        if done == False and env.steps < max_steps:

            obs_pos = next_obs_pos 
        else:
            obs_pos = env.reset()
            done = False
            qnet.reset()

            imitation = True if random.random() < imitation_ratio else False
            imitation_actions = find_path(env) if imitation else None

            while imitation and imitation_actions is None:
                obs_pos = env.reset()
                imitation_actions = find_path(env)

        
        epsilon -= explore_delta
            


def huber_loss(abs_td_error):
    flag = (abs_td_error < 1).float()
    return flag * abs_td_error.pow(2) * 0.5 + (1 - flag) * (abs_td_error - 0.5)


if __name__ == '__main__':

    learn()