import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
# from torch.cuda import amp
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
            prioritized_replay_alpha=config.prioritized_replay_alpha, prioritized_replay_beta=config.prioritized_replay_beta,
            double_q=config.double_q):

    # create network
    qnet = Network().to(device)

    optimizer = Adam(qnet.parameters(), lr=2.5e-4)
    # scheduler = lr_scheduler.StepLR(optimizer, 125000, gamma=0.5)
    # scaler = amp.GradScaler()

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
                if double_q:
                    b_action_ = qnet.bootstrap(b_next_obs, b_next_pos, b_next_bt_steps).argmax(1, keepdim=True)
                    b_q_ = (1 - b_done) * tar_qnet.bootstrap(b_next_obs, b_next_pos, b_next_bt_steps).gather(1, b_action_)
                else:

                    b_q_ = (1 - b_done) * tar_qnet.bootstrap(b_next_obs, b_next_pos, b_next_bt_steps).max(1, keepdim=True)[0]


            b_q = qnet.bootstrap(b_obs, b_pos, b_bt_steps).gather(1, b_action)

            abs_td_error = (b_q - (b_reward + (gamma ** b_steps) * b_q_)).abs().reshape(batch_size, config.num_agents).mean(dim=1, keepdim=True)

            priorities = abs_td_error.detach().cpu().clamp(1e-6).numpy()

            loss = (extra[0] * huber_loss(abs_td_error)).mean()

            optimizer.zero_grad()

            loss.backward()
            # scaler.scale(loss).backward()

            if grad_norm is not None:
                # scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(qnet.parameters(), grad_norm)

            optimizer.step()
            # scaler.step(optimizer)
            # scaler.update()

            # scheduler.step()

            # soft update
            for tar_net, net in zip(tar_qnet.parameters(), qnet.parameters()):
                tar_net.data.copy_(0.001*net.data + 0.999*tar_net.data)

            buffer.update_priorities(extra[1], priorities)

        # update target net and log
        if n_iter % target_network_update_freq == 0:
            # tar_qnet.load_state_dict(qnet.state_dict())

            print('{} Iter {} {}'.format('=' * 10, n_iter, '=' * 10))
            fps = int(target_network_update_freq / (time.time() - start_ts))
            start_ts = time.time()
            print('FPS {}'.format(fps))

            if n_iter > learning_starts == 0:
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

# see Appendix C of `https://arxiv.org/abs/1706.01905`
# q_dict = deepcopy(qnet.state_dict())
# for _, m in qnet.named_modules():
#     if isinstance(m, nn.Linear):
#         std = torch.empty_like(m.weight).fill_(noise_scale)
#         m.weight.data.add_(torch.normal(0, std).to(device))
#         std = torch.empty_like(m.bias).fill_(noise_scale)
#         m.bias.data.add_(torch.normal(0, std).to(device))
# q_perturb = qnet(ob)
# if atom_num > 1:
#     q_perturb = (q_perturb.exp() * vrange).sum(2)
# kl_perturb = ((log_softmax(q, 1) - log_softmax(q_perturb, 1)) *
#             softmax(q, 1)).sum(-1).mean()
# kl_explore = -math.log(1 - epsilon + epsilon / action_dim)
# if kl_perturb < kl_explore:
#     noise_scale *= 1.01
# else:
#     noise_scale /= 1.01
# qnet.load_state_dict(q_dict)
# if random.random() < epsilon:
#     a = int(random.random() * action_dim)
# else:
#     a = q_perturb.argmax(1).cpu().numpy()[0]