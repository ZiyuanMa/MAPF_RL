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
        if prefixsums[0] == 0:
            prefixsums[0] = 1e-5

        idxes = np.zeros(batch_size, dtype=np.int)

        for _ in range(self.layer-1):
            p = self.tree[idxes*2+1]
            idxes = np.where(prefixsums<=p, idxes*2+1, idxes*2+2)
            prefixsums = np.where(idxes%2==0, prefixsums-self.tree[idxes-1], prefixsums)
        
        priorities = self.tree[idxes]
        idxes -= self.capacity-1

        assert np.all(priorities>0), 'idx: {}, priority: {}'.format(idxes, priorities)
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
    __slots__ = ('actor_id', 'map_len', 'num_agents', 'obs_buf', 'pos_buf', 'act_buf', 'rew_buf', 'hid_buf', 'q_buf',
                    'capacity', 'size', 'done', 'td_errors')
    def __init__(self, actor_id, num_agents, map_len, init_obs_pos, size=config.max_steps):
        """
        Prioritized Replay buffer for each actor
        """

        self.actor_id = actor_id
        self.num_agents = num_agents
        self.map_len = map_len
        # observation length should be (max steps+1)
        self.obs_buf = np.zeros((size+1, *config.obs_shape), dtype=np.bool)
        self.pos_buf = np.zeros((size+1, *config.pos_shape), dtype=np.uint8)
        self.act_buf = np.zeros((size), dtype=np.uint8)
        self.rew_buf = np.zeros((size), dtype=np.float32)
        self.hid_buf = np.zeros((size, 256), dtype=np.float32)

        if config.distributional:
            # quantile values
            self.q_buf = np.zeros((size+1, 5, 200), dtype=np.float32)
        else:
            self.q_buf = np.zeros((size+1, 5), dtype=np.float32)

        self.capacity = size
        self.size = 0

        self.obs_buf[0], self.pos_buf[0] = init_obs_pos

        # self.td_errors = np.zeros(size, dtype=np.float32)
    
    def __len__(self):
        return self.size

    def __getitem__(self, idx:int):
        assert idx < self.size

        # self play
        forward = 1
        reward = self.rew_buf[idx]

        if self.done and idx+forward == self.size:
            done = True
        else:
            done = False

        # obs and pos
        bt_steps = min(idx+1, config.bt_steps)
        obs = self.obs_buf[idx+1-bt_steps:idx+2]
        pos = self.pos_buf[idx+1-bt_steps:idx+2]

        if bt_steps < config.bt_steps:
            pad_len = config.bt_steps-bt_steps
            obs = np.pad(obs, ((0,pad_len),(0,0),(0,0),(0,0)))
            pos = np.pad(pos, ((0,pad_len),(0,0)))


        if idx == config.bt_steps:
            hidden = np.zeros(256, dtype=np.float32)
            next_hidden = self.hid_buf[0]
        elif idx < config.bt_steps:
            hidden = np.zeros(256, dtype=np.float32)
            next_hidden = np.zeros(256, dtype=np.float32)
        else:
            hidden = self.hid_buf[idx-config.bt_steps-1]
            next_hidden = self.hid_buf[idx-config.bt_steps]

        return obs, pos, self.act_buf[idx], reward, done, forward, bt_steps, hidden, next_hidden

    def add(self, q_val:np.ndarray, action:int, reward:float, next_obs_pos:np.ndarray, hidden):

        assert self.size < self.capacity

        self.act_buf[self.size] = action
        self.rew_buf[self.size] = reward
        self.obs_buf[self.size+1], self.pos_buf[self.size+1] = next_obs_pos
        self.q_buf[self.size] = q_val
        self.hid_buf[self.size] = hidden

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
                reward = self.rew_buf[i]+0.99*np.max(self.q_buf[i+1], axis=0)
                q_val = self.q_buf[i, self.act_buf[i]]
                self.td_errors[i] = np.abs(reward-q_val)

        delattr(self, 'q_buf')
