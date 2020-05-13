import operator
import random
from typing import List
import numpy as np
import torch
import math

import config

obs_pad = np.zeros((config.obs_dim, 9, 9), dtype=config.dtype)
pos_pad = np.zeros((config.pos_dim), dtype=config.dtype)

discounts = np.array([[0.99**i] for i in range(config.max_steps)])

class SumTree:
    def __init__(self, capacity):

        layer = 1
        while 2**(layer-1) < capacity:
            layer += 1
        assert 2**(layer-1) == capacity, 'buffer size only support power of 2 size'
        self.tree = np.zeros(2**layer-1)
        self.capacity = capacity
        self.size = 0
        self.ptr = capacity-1

    def sum(self):
        assert int(np.sum(self.tree[-self.capacity:])) == int(self.tree[0]), 'sum is {} but root is {}'.format(np.sum(self.tree[-self.capacity:]), self.tree[0])
        return self.tree[0]

    def min(self):
        if self.size < self.capacity:
            return np.min(self.tree[-self.capacity:-self.capacity+self.size])
        else:
            return np.min(self.tree[-self.capacity:])

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

        return idx - self.capacity + 1

    def update(self, idx:int, priority:float):
        assert 0 <= idx < self.capacity

        idx += self.capacity-1

        self.tree[idx] = priority

        idx = (idx-1) // 2
        while idx >= 0:
            self.tree[idx] = self.tree[2*idx+1] + self.tree[2*idx+2]
            idx = (idx-1) // 2

    def add(self, priority:float):
        self.tree[self.ptr] = priority

        idx = (self.ptr-1) // 2
        while idx >= 0:
            self.tree[idx] = self.tree[2*idx+1] + self.tree[2*idx+2]
            idx = (idx-1) // 2

        self.ptr = self.ptr+1 if self.ptr < 2*self.capacity-2 else self.capacity-1
        self.size += 1


class ReplayBuffer:
    def __init__(self, size, device):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """

        self.num_agents = 2
        self.obs_buf = np.zeros((size, self.num_agents, 2, 9, 9), dtype=np.float32)
        self.pos_buf = np.zeros((size, self.num_agents, 4), dtype=np.float32)
        self.act_buf = np.zeros((size, self.num_agents), dtype=np.long)
        self.rew_buf = np.zeros((size, self.num_agents), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, self.num_agents, 2, 9, 9), dtype=np.float32)
        self.next_pos_buf = np.zeros((size, self.num_agents, 4), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.int)
        self.imitat_buf = np.zeros(size, dtype=np.bool)
        self.step_buf = np.zeros(size, dtype=np.int)

        self.capacity = size
        self.size = 0
        self.ptr = 0
        self.device = device


        self.n_step = 1
        self.counter = 0

    def __len__(self):

        return self.size

    def add(self, args):
        obs_pos, actions, reward, next_obs_pos, done, imitation, info = args

        self.obs_buf[self.ptr], self.pos_buf[self.ptr] = obs_pos
        self.act_buf[self.ptr] = actions
        self.rew_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr], self.next_pos_buf[self.ptr] = next_obs_pos
        self.done_buf[self.ptr] = done
        self.imitat_buf[self.ptr] = imitation
        self.step_buf[self.ptr] = info['step']

        self.ptr = (self.ptr + 1) % self.capacity

        if self.size < self.capacity:
            self.size += 1

    def _encode_sample(self, idxes):
        b_obs, b_pos, b_action, b_reward, b_next_obs, b_next_pos, b_done, b_steps, b_bt_steps, b_next_bt_steps = [], [], [], [], [], [], [], [], [], []
        
        for i in idxes:
            forward = 1
            if self.imitat_buf[i]:
                # use Monte Carlo method if it's imitation

                while not self.done_buf[(i+forward-1)%self.capacity] and (i+forward)%self.capacity != self.ptr:
                    forward += 1

                if i+forward > self.capacity:
                    reward = np.concatenate((self.rew_buf[i:], self.rew_buf[:(i+forward)%self.capacity]))
                    reward = np.sum(reward*discounts[:forward], axis=0)
                else:
                    reward = np.sum(self.rew_buf[i:i+forward]*discounts[:forward], axis=0)

            else:
                reward = self.rew_buf[i]

            bt_steps = min(self.step_buf[self.ptr]+1, config.bt_steps)


            if i+1-bt_steps >= 0:
                obs = np.swapaxes(self.obs_buf[i+1-bt_steps:i+1], 0, 1)
                pos = np.swapaxes(self.pos_buf[i+1-bt_steps:i+1], 0, 1)
            else:
                obs = np.swapaxes(np.concatenate((self.obs_buf[i+1-bt_steps:], self.obs_buf[:i+1])), 0, 1)
                pos = np.swapaxes(np.concatenate((self.pos_buf[i+1-bt_steps:], self.pos_buf[:i+1])), 0, 1)

            if bt_steps < config.bt_steps:
                pad_len = config.bt_steps-bt_steps
                obs = np.pad(obs, ((0,0),(0,pad_len),(0,0),(0,0),(0,0)))
                pos = np.pad(pos, ((0,0),(0,pad_len),(0,0)))

                next_bt_steps = bt_steps+1

                next_obs = np.copy(obs)
                next_obs[:,bt_steps] = self.next_obs_buf[i]

                next_pos = np.copy(pos)
                next_pos[:,bt_steps] = self.next_pos_buf[i]

            else:
                next_bt_steps = bt_steps
                next_obs = np.concatenate((obs[:,1:], np.expand_dims(self.next_obs_buf[i], axis=1)), axis=1)
                next_pos = np.concatenate((pos[:,1:], np.expand_dims(self.next_pos_buf[i], axis=1)), axis=1)


            b_obs.append(obs)
            b_pos.append(pos)
            b_action += self.act_buf[i].tolist()
            b_reward += reward.tolist()
            b_next_obs.append(next_obs)
            b_next_pos.append(next_pos)

            b_done += [ self.done_buf[i] for _ in range(self.num_agents) ]
            b_steps += [ forward for _ in range(self.num_agents) ]
            b_bt_steps += [ bt_steps for _ in range(self.num_agents) ]
            b_next_bt_steps += [ next_bt_steps for _ in range(self.num_agents) ]


        res = (
            torch.from_numpy(np.concatenate(b_obs)).to(self.device),
            torch.from_numpy(np.concatenate(b_pos)).to(self.device),
            torch.LongTensor(b_action).unsqueeze(1).to(self.device),
            torch.FloatTensor(b_reward).unsqueeze(1).to(self.device),
            torch.from_numpy(np.concatenate(b_next_obs)).to(self.device),
            torch.from_numpy(np.concatenate(b_next_pos)).to(self.device),
            torch.FloatTensor(b_done).unsqueeze(1).to(self.device),
            torch.FloatTensor(b_steps).unsqueeze(1).to(self.device),
            torch.LongTensor(b_bt_steps).to(self.device),
            torch.LongTensor(b_next_bt_steps).to(self.device),
        )

        return res

    def sample(self, batch_size):
        """Sample a batch of experiences."""
        indexes = range(self.size)
        idxes = []
        for _ in range(batch_size):
            idx = random.choice(indexes)
            idxes.append(idx)

        return self._encode_sample(idxes)
    
    def step(self):
        self.counter += 1
        if self.counter == 350000:
            self.counter = 0
            self.n_step += 1

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, device, alpha, beta):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size, device)
        assert alpha >= 0
        self.alpha = alpha

        self.priority_tree = SumTree(size)
        self.max_priority = 1.0
        self.beta = beta

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""

        super().add(*args, **kwargs)

        self.priority_tree.add(self.max_priority**self.alpha)


    def _sample_proportional(self, batch_size):
        res = []
        p_total = self.priority_tree.sum()

        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self.priority_tree.find_prefixsum_idx(mass)

            res.append(idx)
            
        return res

    def sample(self, batch_size):
        """Sample a batch of experiences"""
        idxes = self._sample_proportional(batch_size)

        min_p = self.priority_tree.min()
        
        samples_p = np.asarray([self.priority_tree[idx] for idx in idxes])
        weights = np.power(samples_p/min_p, -self.beta)
        weights = torch.from_numpy(weights.astype('float32'))
        weights = weights.unsqueeze(1).to(self.device)
        encoded_sample = self._encode_sample(idxes)

        super().step()

        return encoded_sample + (weights, idxes)

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions"""

        for idx, priority in zip(idxes, priorities):
            assert (priority > 0).all()
            assert 0 <= idx < self.size

            self.priority_tree.update(idx, priority**self.alpha)

            self.max_priority = max(self.max_priority, priority)



