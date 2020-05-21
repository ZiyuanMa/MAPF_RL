import ray
import time
import random
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from copy import deepcopy
from typing import List
import threading

import config
from model_dqn import Network
from environment import Environment
from buffer import SumTree, LocalBuffer
from search import find_path

@ray.remote(num_cpus=1)
class GlobalBuffer:
    def __init__(self, capacity, beta=config.prioritized_replay_beta):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        self.buffer = [ None for _ in range(capacity) ]
        self.priority_tree = SumTree(capacity)
        self.beta = beta
        self.counter = 0
        self.data = []
        self.lock = threading.Lock()

    def __len__(self):
        return self.size

    def run(self):
        self.background_thread = threading.Thread(target=self.prepare_data, daemon=True)
        self.background_thread.start()

    def prepare_data(self):
        while True:
            if len(self.data) < 4:
                data = self.sample_batch(config.batch_size)
                data_id = ray.put(data)
                self.data.append(data_id)
            else:
                time.sleep(0.5)
    
    def get_data(self):

        if len(self.data) == 0:
            print('no prepared data')
            data = self.sample_batch(config.batch_size)
            data_id = ray.put(data)
            return data_id
        else:
            return self.data.pop(0)


    def add(self, buffer):
        # update buffer size
        with self.lock:
            if self.buffer[self.ptr] is not None:
                self.size -= len(self.buffer[self.ptr])
            self.size += len(buffer)
            self.counter += len(buffer)

            buffer.priority_tree.tree.flags.writeable = True

            self.buffer[self.ptr] = buffer

            self.priority_tree.update(self.ptr, buffer.priority)

            self.ptr = (self.ptr+1) % self.capacity

    def sample_batch(self, batch_size):
        with self.lock:
            if len(self) < config.learning_starts:
                raise Exception('buffer size is not large enough')

            total_p = self.priority_tree.sum()

            b_obs, b_pos, b_action, b_reward, b_next_obs, b_next_pos, b_done, b_steps, b_bt_steps, b_next_bt_steps = [], [], [], [], [], [], [], [], [], []
            idxes, priorities = [], []

            every_range_len = total_p / batch_size
            for i in range(batch_size):
                global_prefixsum = random.random() * every_range_len + i * every_range_len
                global_idx, local_prefixsum = self.priority_tree.find_prefixsum_idx(global_prefixsum)
                ret = self.buffer[global_idx].sample(local_prefixsum)
                obs, pos, action, reward, next_obs, next_pos, done, steps, bt_steps, next_bt_steps, local_idx, priority = ret   
                
                b_obs.append(obs)
                b_pos.append(pos)
                b_action += action
                b_reward += reward
                b_next_obs.append(next_obs)
                b_next_pos.append(next_pos)

                b_done += done
                b_steps += steps
                b_bt_steps += bt_steps
                b_next_bt_steps += next_bt_steps

                idxes.append(global_idx*config.max_steps+local_idx)
                priorities.append(priority)

            priorities = np.array(priorities, dtype=np.float32)
            min_p = np.min(priorities)
            weights = np.power(priorities/min_p, -self.beta)

            data = (
                torch.from_numpy(np.concatenate(b_obs).astype(np.float32)),
                torch.from_numpy(np.concatenate(b_pos).astype(np.float32)),
                torch.LongTensor(b_action).unsqueeze(1),
                torch.FloatTensor(b_reward).unsqueeze(1),
                torch.from_numpy(np.concatenate(b_next_obs).astype(np.float32)),
                torch.from_numpy(np.concatenate(b_next_pos).astype(np.float32)),

                torch.FloatTensor(b_done).unsqueeze(1),
                torch.FloatTensor(b_steps).unsqueeze(1),
                b_bt_steps,
                b_next_bt_steps,

                idxes,
                torch.from_numpy(weights).unsqueeze(1),
                self.ptr
            )

            return data

    def update_priorities(self, idxes:List[int], priorities:List[float], old_ptr):
        """Update priorities of sampled transitions"""
        with self.lock:
            idxes = np.asarray(idxes)
            priorities = np.asarray(priorities)

            # discard the idx that already been discarded during training
            if self.ptr > old_ptr:
                # range from [old_ptr, self.ptr)
                mask = (idxes < old_ptr*config.max_steps) | (idxes >= self.ptr*config.max_steps)
                idxes = idxes[mask]
                priorities = priorities[mask]
            elif self.ptr < old_ptr:
                # range from [0, self.ptr) & [old_ptr, self,capacity)
                mask = (idxes < old_ptr*config.max_steps) & (idxes >= self.ptr*config.max_steps)
                idxes = idxes[mask]
                priorities = priorities[mask]

            global_idxes = idxes // config.max_steps
            local_idxes = idxes % config.max_steps

            for global_idx, local_idx, priority in zip(global_idxes, local_idxes, priorities):
                assert priority > 0

                self.buffer[global_idx].update_priority(local_idx, priority)

            global_idxes = np.unique(global_idxes)
            new_p = []
            for global_idx in global_idxes:
                new_p.append(self.buffer[global_idx].priority)

            new_p = np.asarray(new_p)
            self.priority_tree.batch_update(global_idxes, new_p)

    def stats(self, interval:int):
        print('buffer update: {}'.format(self.counter/interval))
        print('buffer size: {}'.format(self.size))
        self.counter = 0

    def ready(self):
        if len(self) >= config.learning_starts:
            return True
        else:
            return False

@ray.remote(num_cpus=1, num_gpus=1)
class Learner:
    def __init__(self, buffer:GlobalBuffer):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Network()
        self.model.to(self.device)
        self.tar_model = deepcopy(self.model)
        self.optimizer = Adam(self.model.parameters(), lr=1.25e-4)
        # self.scheduler = MultiStepLR(self.optimizer, milestones=[5000,30000,40000,80000], gamma=0.5)
        self.buffer = buffer
        self.counter = 0
        self.done = False
        self.loss = 0

        self.store_weights()

    def get_weights(self):
        return self.weights_id

    def store_weights(self):
        state_dict = self.model.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        weights_id = ray.put(state_dict)
        self.weights_id = weights_id

    def run(self):
        self.learning_thread = threading.Thread(target=self.train, daemon=True)
        self.learning_thread.start()

    def train(self):

        min_value = -5
        max_value = 5
        atom_num = 51
        delta_z = 10 / 50
        z_i = torch.linspace(-5, 5, 51).to(self.device)

        for i in range(1, 200001):

            data_id = ray.get(self.buffer.get_data.remote())
            data = ray.get(data_id)
 
            b_obs, b_pos, b_action, b_reward, b_next_obs, b_next_pos, b_done, b_steps, b_bt_steps, b_next_bt_steps, idxes, weights, old_ptr = data
            b_obs, b_pos, b_action, b_reward = b_obs.to(self.device), b_pos.to(self.device), b_action.to(self.device), b_reward.to(self.device)
            b_next_obs, b_next_pos, b_done, b_steps, weights = b_next_obs.to(self.device), b_next_pos.to(self.device), b_done.to(self.device), b_steps.to(self.device), weights.to(self.device)

            if config.distributional:
                with torch.no_grad():
                    b_next_dist = self.tar_model.bootstrap(b_next_obs, b_next_pos, b_next_bt_steps).exp()
                    b_next_action = (b_next_dist * z_i).sum(-1).argmax(1)
                    b_tzj = ((0.99**b_steps) * (1 - b_done) * z_i[None, :] + b_reward).clamp(min_value, max_value)
                    b_i = (b_tzj - min_value) / delta_z
                    b_l = b_i.floor()
                    b_u = b_i.ceil()
                    b_m = torch.zeros(config.batch_size*config.num_agents, atom_num).to(self.device)
                    temp = b_next_dist[torch.arange(config.batch_size*config.num_agents), b_next_action, :]
                    b_m.scatter_add_(1, b_l.long(), temp * (b_u - b_i))
                    b_m.scatter_add_(1, b_u.long(), temp * (b_i - b_l))

                b_q = self.model.bootstrap(b_obs, b_pos, b_bt_steps)[torch.arange(config.batch_size*config.num_agents), b_action.squeeze(1), :]

                kl_error = (-b_q*b_m).sum(dim=1).reshape(config.batch_size, config.num_agents).mean(dim=1)

                priorities = kl_error.detach().cpu().clamp(1e-6).numpy()
                loss = kl_error.mean()

            else:
                with torch.no_grad():
                    # choose max q index from next observation
                    # double q-learning
                    if config.double_q:
                        b_action_ = self.model.bootstrap(b_next_obs, b_next_pos, b_next_bt_steps).argmax(1, keepdim=True)
                        b_q_ = (1 - b_done) * self.tar_model.bootstrap(b_next_obs, b_next_pos, b_next_bt_steps).gather(1, b_action_)
                    else:
                        b_q_ = (1 - b_done) * self.tar_model.bootstrap(b_next_obs, b_next_pos, b_next_bt_steps).max(1, keepdim=True)[0]

                b_q = self.model.bootstrap(b_obs, b_pos, b_bt_steps).gather(1, b_action)

                abs_td_error = (b_q - (b_reward + (0.99 ** b_steps) * b_q_)).abs().reshape(config.batch_size, config.num_agents).mean(dim=1, keepdim=True)

                priorities = abs_td_error.detach().cpu().clamp(1e-6).numpy()

                loss = (weights * self.huber_loss(abs_td_error)).mean()

            self.optimizer.zero_grad()

            loss.backward()
            self.loss = loss.item()
            # scaler.scale(loss).backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), 40)

            self.optimizer.step()
            # scaler.step(optimizer)
            # scaler.update()

            # self.scheduler.step()

            # store new weight in shared memory
            self.store_weights()

            self.buffer.update_priorities.remote(idxes, priorities, old_ptr)

            self.counter += 1

            # update target net, save model
            if i % 2000 == 0:
                self.tar_model.load_state_dict(self.model.state_dict())
                torch.save(self.model.state_dict(), os.path.join(config.save_path, '{}.pth'.format(i)))

        self.done = True
    def huber_loss(self, abs_td_error):
        flag = (abs_td_error < 1).float()
        return flag * abs_td_error.pow(2) * 0.5 + (1 - flag) * (abs_td_error - 0.5)

    def stats(self, interval:int):
        print('updates: {}'.format(self.counter))
        print('loss: {}'.format(self.loss))
        # self.counter = 0
        return self.done

@ray.remote(num_cpus=1)
class Actor:
    def __init__(self, worker_id, epsilon, learner:Learner, buffer:GlobalBuffer):
        self.id = worker_id
        self.model = Network()
        self.model.eval()
        self.env = Environment()
        self.epsilon = epsilon
        self.learner = learner
        self.buffer = buffer
        self.distributional = config.distributional
        self.imitation_ratio = config.imitation_ratio
        self.max_steps = config.max_steps

    def run(self):
        """ Generate training batch sample """
        done = False

        if self.distributional:
            vrange = torch.linspace(-5, 5, 51)

        # if use imitation learning
        imitation = True if random.random() < self.imitation_ratio else False
        if imitation:
            imitation_actions = find_path(self.env)
            while imitation_actions is None:
                self.env.reset()
                imitation_actions = find_path(self.env)
            obs_pos = self.env.observe()
            buffer = LocalBuffer(obs_pos, True)
        else:
            obs_pos = self.env.reset()
            buffer = LocalBuffer(obs_pos, False)

        while True:

            if imitation:

                actions = imitation_actions.pop(0)
                with torch.no_grad():
                    q_val = self.model.step(torch.FloatTensor(obs_pos[0]), torch.FloatTensor(obs_pos[1]))
                    if self.distributional:
                        q_val = (q_val.exp() * vrange).sum(2)

            else:
                # sample action
                with torch.no_grad():

                    q_val = self.model.step(torch.FloatTensor(obs_pos[0]), torch.FloatTensor(obs_pos[1]))

                    if self.distributional:
                        q_val = (q_val.exp() * vrange).sum(2)

                    actions = q_val.argmax(1).tolist()

                    for i in range(len(actions)):
                        if random.random() < self.epsilon:
                            actions[i] = np.random.randint(0, 5)

            # take action in env
            next_obs_pos, r, done, _ = self.env.step(actions)
        

            # return data and update observation

            buffer.add(q_val.numpy(), actions, r, next_obs_pos)


            if done == False and self.env.steps < self.max_steps:

                obs_pos = next_obs_pos 
            else:
                # finish and send buffer
                if done:
                    buffer.finish()
                else:
                    with torch.no_grad():
                        q_val = self.model.step(torch.FloatTensor(next_obs_pos[0]), torch.FloatTensor(next_obs_pos[1]))
                        if self.distributional:
                            q_val = (q_val.exp() * vrange).sum(2)
                    buffer.finish(q_val)

                self.buffer.add.remote(buffer)

                done = False
                self.model.reset()
                obs_pos = self.env.reset()

                self.update_weights()

                imitation = True if random.random() < self.imitation_ratio else False
                if imitation:
                    imitation_actions = find_path(self.env)
                    while imitation_actions is None:
                        obs_pos = self.env.reset()
                        imitation_actions = find_path(self.env)

                    buffer = LocalBuffer(obs_pos, True)
                else:
                    buffer = LocalBuffer(obs_pos, False)

    def update_weights(self):
        '''load weights from learner'''
        weights_id = ray.get(self.learner.get_weights.remote())
        weights = ray.get(weights_id)
        self.model.load_state_dict(weights)
    
