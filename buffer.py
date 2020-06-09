import random
from typing import List
import numpy as np
from numba import int32, float32
import torch
import math
from dataclasses import dataclass

import config

discounts = np.array([ 0.99**i for i in range(config.max_steps)])

def quantile_huber_loss(curr_dist, target_dist, kappa=1.0):
    curr_dist = np.expand_dims(curr_dist, 1)
    target_dist = np.expand_dims(target_dist, 0)

    td_errors = curr_dist - target_dist

    abs_td_errors = np.abs(td_errors)

    flag = (abs_td_errors < kappa).astype(np.float32)
    element_wise_huber_loss = flag * (abs_td_errors**2) * 0.5 + (1 - flag) * kappa * (abs_td_errors - 0.5*kappa)

    taus = np.expand_dims(np.arange(1/400, 1, 1/200), 1)

    element_wise_quantile_huber_loss = np.abs(taus - (td_errors < 0).astype(np.float32)) * element_wise_huber_loss / kappa

    quantile_huber_loss = np.mean(np.sum(element_wise_quantile_huber_loss, axis=0), axis=0)

    return quantile_huber_loss


class SumTree:

    def __init__(self, capacity):

        layer = 1
        while 2**(layer-1) < capacity:
            layer += 1
        assert 2**(layer-1) == capacity, 'buffer size only support power of 2 size'
        self.layer = layer
        self.tree = np.zeros(2**layer-1, dtype=np.float32)
        self.capacity = capacity
        self.size = 0

    def sum(self):
        assert np.sum(self.tree[-self.capacity:])-self.tree[0] < 0.1, 'sum is {} but root is {}'.format(np.sum(self.tree[-self.capacity:]), self.tree[0])
        return self.tree[0]

    def __getitem__(self, idx:int):
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity-1+idx]

    def find_prefixsum_idx(self, prefixsum:float):
        """
        Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i]) <= prefixsum
        """
        
        assert 0 <= prefixsum <= self.sum() + 1e-5

        idx = 0
        while idx < self.capacity-1:  # while non-leaf
            if self.tree[2*idx + 1] > prefixsum:
                idx = 2*idx + 1
            else:
                prefixsum -= self.tree[2*idx + 1]
                idx = 2*idx + 2

        return idx-self.capacity+1, prefixsum

    def batch_sample(self, batch_size:int):
        sum = self.tree[0]
        interval = sum/batch_size

        prefixsums = np.arange(0, sum, interval) + np.random.uniform(0, interval, batch_size)

        idxes = np.zeros(batch_size, dtype=np.int)

        for _ in range(self.layer-1):
            p = self.tree[idxes*2+1]
            idxes = np.where(prefixsums<p, idxes*2+1, idxes*2+2)
            prefixsums = np.where(idxes%2==0, prefixsums-self.tree[idxes-1], prefixsums)

        priorities = self.tree[idxes]
        idxes -= self.capacity-1

        assert np.all(idxes>=0) and np.all(idxes<self.capacity)

        return idxes, priorities

    def update(self, idx:int, priority:float):
        assert 0 <= idx < self.capacity
        # self.tree.flags.writeable = True

        idx += self.capacity-1

        self.tree[idx] = priority

        idx = (idx-1) // 2
        while idx >= 0:
            self.tree[idx] = self.tree[2*idx+1] + self.tree[2*idx+2]
            idx = (idx-1) // 2
        
        assert np.sum(self.tree[-self.capacity:])-self.tree[0] < 0.1, 'sum is {} but root is {}'.format(np.sum(self.tree[-self.capacity:]), self.tree[0])
        
    def batch_update(self, idxes:np.ndarray, priorities:np.ndarray):
        idxes += self.capacity-1
        self.tree[idxes] = priorities

        for _ in range(self.layer-1):
            idxes = (idxes-1) // 2
            idxes = np.unique(idxes)
            self.tree[idxes] = self.tree[2*idxes+1] + self.tree[2*idxes+2]
        
        # check
        assert np.sum(self.tree[-self.capacity:])-self.tree[0] < 0.1, 'sum is {} but root is {}'.format(np.sum(self.tree[-self.capacity:]), self.tree[0])


class LocalBuffer:
    __slots__ = ('num_agents', 'obs_buf', 'pos_buf', 'act_buf', 'rew_buf', 'q_buf',
                    'capacity', 'size', 'imitation', 'done', 'td_errors', 'comm_mask')
    def __init__(self, init_obs_pos, imitation:bool, size=config.max_steps):
        """
        Prioritized Replay buffer for each actor
        """

        self.num_agents = init_obs_pos[0].shape[0]
        # observation length should be (max steps+1)
        self.obs_buf = np.zeros((size+1, self.num_agents, *config.obs_shape), dtype=np.bool)
        self.pos_buf = np.zeros((size+1, self.num_agents, *config.pos_shape), dtype=np.uint8)
        self.act_buf = np.zeros((size, self.num_agents), dtype=np.uint8)
        self.rew_buf = np.zeros((size, self.num_agents), dtype=np.float32)

        if config.distributional:
            # quantile values
            self.q_buf = np.zeros((size+1, self.num_agents, 5, 200), dtype=np.float32)
        else:
            self.q_buf = np.zeros((size+1, self.num_agents, 5), dtype=np.float32)

        self.capacity = size
        self.size = 0
        self.imitation = imitation

        self.obs_buf[0], self.pos_buf[0] = init_obs_pos

        # self.td_errors = np.zeros(size, dtype=np.float32)
    
    def __len__(self):
        return self.size

    def __getitem__(self, idx:int):
        assert idx < self.size

        if self.imitation:
            # n-step forward = 4
            forward = min(4, self.size-idx)
            reward = np.sum(self.rew_buf[idx:idx+forward, 0]*discounts[:forward], axis=0)
        else:
            # self play
            forward = 1
            reward = self.rew_buf[idx, 0]

        if self.done and idx+forward == self.size:
            done = True
        else:
            done = False

        # obs and pos
        bt_steps = min(idx+1, config.bt_steps)
        # obs = np.swapaxes(self.obs_buf[idx+1-bt_steps:idx+1], 0, 1)
        # pos = np.swapaxes(self.pos_buf[idx+1-bt_steps:idx+1], 0, 1)
        obs = self.obs_buf[idx+1-bt_steps:idx+1].swapaxes(0,1)
        pos = self.pos_buf[idx+1-bt_steps:idx+1].swapaxes(0,1)
        comm_mask = self.comm_mask[idx+1-bt_steps:idx+1]

        # if len(adj_list)==1:
        #     obs = np.expand_dims(obs, 1)
        #     pos = np.expand_dims(pos, 1)

        # print(obs.shape)
        # print(adj_list)
        # print(type(adj_list))

        if bt_steps < config.bt_steps:
            step_pad = config.bt_steps-bt_steps
            obs = np.pad(obs, ((0,0), (0,step_pad), (0,0), (0,0), (0,0)))
            pos = np.pad(pos, ((0,0), (0,step_pad), (0,0)))
            comm_mask = np.pad(comm_mask, ((0,step_pad), (0,0), (0,0)))

        # next obs and next pos
        next_bt_steps = min(idx+1+forward, config.bt_steps)
        # next_obs = np.swapaxes(self.obs_buf[idx+1+forward-next_bt_steps:idx+1+forward], 0, 1)
        # next_pos = np.swapaxes(self.pos_buf[idx+1+forward-next_bt_steps:idx+1+forward], 0, 1)
        next_obs = self.obs_buf[idx+1+forward-next_bt_steps:idx+1+forward].swapaxes(0,1)
        next_pos = self.pos_buf[idx+1+forward-next_bt_steps:idx+1+forward].swapaxes(0,1)
        next_comm_mask = self.comm_mask[idx+1+forward-next_bt_steps:idx+1+forward]

        # if len(next_adj_list)==1:
        #     next_obs = np.expand_dims(next_obs, 1)
        #     next_pos = np.expand_dims(next_pos, 1)

        if next_bt_steps < config.bt_steps:
            step_pad = config.bt_steps-next_bt_steps
            next_obs = np.pad(next_obs, ((0,0), (0,step_pad), (0,0), (0,0), (0,0)))
            next_pos = np.pad(next_pos, ((0,0), (0,step_pad), (0,0)))
            next_comm_mask = np.pad(next_comm_mask, ((0,step_pad), (0,0), (0,0)))

        return obs, pos, self.act_buf[idx, 0], reward, next_obs, next_pos, done, forward, [bt_steps for _ in range(config.max_comm_agents)], [next_bt_steps for _ in range(config.max_comm_agents)], comm_mask, next_comm_mask

    def add(self, q_val:np.ndarray, actions:List[int], reward:List[float], next_obs_pos:np.ndarray):

        assert self.size < self.capacity

        self.act_buf[self.size] = actions
        self.rew_buf[self.size] = reward
        self.obs_buf[self.size+1], self.pos_buf[self.size+1] = next_obs_pos
        self.q_buf[self.size] = q_val

        self.size += 1

    def finish(self, last_q_val=None):
        # last q value is None if done
        if last_q_val is None:
            self.done = True
        else:
            self.done = False
            self.q_buf[self.size] = last_q_val
        
        self.obs_buf = self.obs_buf[:self.size+1]
        self.pos_buf = self.pos_buf[:self.size+1]
        self.act_buf = self.act_buf[:self.size]
        self.rew_buf = self.rew_buf[:self.size]
        self.q_buf = self.q_buf[:self.size+1]

        self.td_errors = np.zeros(self.capacity, dtype=np.float32)

        if self.imitation:
            for i in range(self.size):
                # n-steps forward = 4
                forward = min(4, self.size-i)
                # print(self.rew_buf[i:i+forward, 0])
                # print(self.rew_buf[i:i+forward, 0]*discounts[:forward])
                if config.distributional:
                    next_dist = self.q_buf[i+forward, 0]
                    next_q = np.mean(next_dist, axis=1)
                    next_action = np.argmax(next_q)
                    next_dist = next_dist[next_action]

                    target_dist = np.sum(self.rew_buf[i:i+forward, 0]*discounts[:forward], axis=0) + (0.99**forward)*next_dist

                    curr_dist = self.q_buf[i, 0, self.act_buf[i, 0]]

                    self.td_errors[i] = quantile_huber_loss(curr_dist, target_dist)

                else:
                    reward = np.sum(self.rew_buf[i:i+forward, 0]*discounts[:forward], axis=0)+(0.99**forward)*np.max(self.q_buf[i+forward, 0], axis=0)
                    q_val = self.q_buf[i, 0, self.act_buf[i, 0]]
                    self.td_errors[i] = np.abs(reward-q_val)

        else:
            for i in range(self.size):
                # forward = 1
                if config.distributional:
                    next_dist = self.q_buf[i+1, 0]
                    next_q = np.mean(next_dist, axis=1)
                    next_action = np.argmax(next_q)
                    next_dist = next_dist[next_action]

                    target_dist = self.rew_buf[i, 0]+0.99*next_dist

                    curr_dist = self.q_buf[i, 0, self.act_buf[i, 0]]

                    self.td_errors[i] = quantile_huber_loss(curr_dist, target_dist)
                else:
                    reward = self.rew_buf[i, 0]+0.99*np.max(self.q_buf[i+1, 0], axis=0)
                    q_val = self.q_buf[i, 0, self.act_buf[i, 0]]
                    self.td_errors[i] = np.abs(reward-q_val)

        # if config.distributional:
        #     raise NotImplementedError
        # else:
        #     if self.imitation:
        #         for i in range(self.size):
        #             forward = min(4, self.size-i)
        #             reward = np.sum(self.rew_buf[i:i+forward, 0]*discounts[:forward], axis=0)+(0.99**forward)*np.max(self.q_buf[i+forward, 0], axis=0)
        #             q_val = self.q_buf[i, 0, self.act_buf[i, 0]]
        #             self.td_errors[i] = np.abs(reward-q_val)
        #     else:
        #         selected_q = self.q_buf[:, 0][self.act_buf[:, 0]]
        #         max_q = np.max(self.q_buf[:, 0], axis=0)

        relative_pos = np.abs(np.expand_dims(self.pos_buf[:, :, :2], 1)-np.expand_dims(self.pos_buf[:, :, :2], 2))
        in_obs_mask = np.all(relative_pos<=config.obs_radius, axis=3)
        relative_dis = np.sqrt(relative_pos[:, :, :, 0]**2+relative_pos[:, :, :, 1]**2)
        dis_mask = np.zeros((self.size+1, self.num_agents, self.num_agents), dtype=np.bool)
        dis_mask[relative_dis.argsort() < config.max_comm_agents] = True

        self.comm_mask = np.bitwise_and(in_obs_mask, dis_mask)

        delattr(self, 'q_buf')
