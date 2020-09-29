import random
from typing import List
import numpy as np
# from numba import int32, float32
import torch
import math
from dataclasses import dataclass


import config

discounts = np.array([ 0.99**i for i in range(config.max_steps)])



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

        prefixsums = np.arange(0, sum, interval, dtype=np.float64) + np.random.uniform(0, interval, batch_size)
        if prefixsums[0] == 0:
            prefixsums[0] = 1e-5

        idxes = np.zeros(batch_size, dtype=np.int)

        for _ in range(self.layer-1):
            p = self.tree[idxes*2+1]
            idxes = np.where(prefixsums<=p, idxes*2+1, idxes*2+2)
            prefixsums = np.where(idxes%2==0, prefixsums-self.tree[idxes-1], prefixsums)
            prefixsums = np.where(prefixsums==0, 1e-5, prefixsums)
        
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
    __slots__ = ('actor_id', 'map_len', 'num_agents', 'obs_buf', 'act_buf', 'rew_buf', 'hid_buf', 'comm_buf', 'q_buf',
                    'capacity', 'size', 'done', 'td_errors')
    def __init__(self, actor_id, num_agents, map_len, init_obs, size=config.max_steps):
        """
        Prioritized Replay buffer for each actor
        """

        self.actor_id = actor_id
        self.num_agents = num_agents
        self.map_len = map_len
        # observation length should be (max steps+1)
        self.obs_buf = np.zeros((size+1, self.num_agents, *config.obs_shape), dtype=np.bool)
        self.act_buf = np.zeros((size), dtype=np.uint8)
        self.rew_buf = np.zeros((size), dtype=np.float16)
        self.hid_buf = np.zeros((size,  self.num_agents, config.latent_dim), dtype=np.float16)
        self.comm_buf = np.zeros((size+1, num_agents, num_agents), dtype=np.bool)

        self.q_buf = np.zeros((size+1, 5), dtype=np.float32)

        self.capacity = size
        self.size = 0

        self.obs_buf[0] = init_obs


        # self.td_errors = np.zeros(size, dtype=np.float32)
    
    def __len__(self):
        return self.size


    def add(self, q_val:np.ndarray, action:int, reward, next_obs:np.ndarray, hidden, comm_mask):

        assert self.size < self.capacity

        self.act_buf[self.size] = action
        self.rew_buf[self.size] = reward
        self.obs_buf[self.size+1] = next_obs
        self.q_buf[self.size] = q_val
        self.hid_buf[self.size] = hidden
        self.comm_buf[self.size] = comm_mask

        self.size += 1

    def finish(self, last_q_val=None, comm_mask=None):
        # last q value is None if done
        if last_q_val is None:
            self.done = True
        else:
            self.done = False
            self.q_buf[self.size] = last_q_val
            self.comm_buf[self.size] = comm_mask
        
        self.obs_buf = self.obs_buf[:self.size+1]
        self.act_buf = self.act_buf[:self.size]
        self.rew_buf = self.rew_buf[:self.size]
        self.hid_buf = self.hid_buf[:self.size]
        self.comm_buf = self.comm_buf[:self.size+1]
        self.q_buf = self.q_buf[:self.size+1]


        self.td_errors = np.zeros(self.capacity, dtype=np.float32)


        q_max = np.max(self.q_buf[:self.size], axis=1)
        ret = self.rew_buf.tolist() + [ 0 for _ in range(config.forward_steps-1)]
        reward = np.convolve(ret, [0.99**(config.forward_steps-1-i) for i in range(config.forward_steps)],'valid')+q_max
        q_val = self.q_buf[np.arange(self.size), self.act_buf]
        self.td_errors[:self.size] = np.abs(reward-q_val)

        return  self.actor_id, self.num_agents, self.map_len, self.obs_buf, self.act_buf, self.rew_buf, self.hid_buf, self.td_errors, self.done, self.size, self.comm_buf
