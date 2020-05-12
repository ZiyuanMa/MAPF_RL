"""
This file is adapted from following files in openai/baselines.
deepq/replay_buffer.py
baselines/acer/buffer.py
"""
import operator
import random
from typing import List
import numpy as np
import torch
import math

import config

obs_pad = np.zeros((config.obs_dim, 9, 9), dtype=config.dtype)
pos_pad = np.zeros((config.pos_dim), dtype=config.dtype)


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

    def __setitem__(self, idx, val):

        assert val != 0, 'val is 0'
        idx += self.capacity-1
        if self.tree[idx] == 0:
            self.size += 1

        self.tree[idx] = val
        
        idx = (idx-1) // 2
        while idx >= 0:
            self.tree[idx] = self.tree[2*idx+1] + self.tree[2*idx+2]
            idx = (idx-1) // 2

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
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self._device = device
        self.n_step = 1
        self.counter = 0

    def __len__(self):
        return len(self._storage)

    def add(self, args):
        if self._next_idx >= len(self._storage):
            self._storage.append(args)
        else:
            self._storage[self._next_idx] = args
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        b_obs, b_pos, b_action, b_reward, b_next_obs, b_next_pos, b_done, b_steps, b_bt_steps, b_next_bt_steps = [], [], [], [], [], [], [], [], [], []
        
        for i in idxes:
            obs_pos, action, reward, next_obs_pos, done, imitation, info = self._storage[i]

            # caculate forward steps and cummulated rewards
            forward = 1
            sum_reward = np.array(reward, dtype=np.float32)

            if imitation:
                # use Monte Carlo method if it's imitation
                for j in range(1, config.max_steps):
                    next_idx = (i+j) % self._maxsize
                    if next_idx != self._next_idx and not done:
                        _, _, next_reward, _, done, _, _ = self._storage[next_idx]

                        sum_reward += np.array(next_reward, dtype=np.float32) * (0.99 ** j)
                        forward += 1

                    else:
                        break
            else:
                # n-steps forward
                for j in range(1, self.n_step):
                    next_idx = (i+j) % self._maxsize
                    if next_idx != self._next_idx and not done:
                        _, _, next_reward, _, next_done, _, next_info = self._storage[next_idx]

                        if next_info['step'] == 0:
                            break

                        sum_reward += np.array(next_reward, dtype=np.float32) * (0.99 ** j)
                        done = next_done
                        forward += 1

                    else:
                        break
            
            # obs_pos
            bt_steps = min(info['step']+1, config.bt_steps)
            num_agents = obs_pos[1].shape[0]
            obs = [ [] for agent_id in range(num_agents) ]
            pos = [ [] for agent_id in range(num_agents) ]

            for step in range(bt_steps):
                bt_obs_pos, _, _, _, _, _, _ = self._storage[(i-step)%self._maxsize]
                for agent_id in range(num_agents):
                    obs[agent_id].append(bt_obs_pos[0][agent_id])
                    pos[agent_id].append(bt_obs_pos[1][agent_id])

            # reverse sequence of states
            for agent_id in range(num_agents):
                obs[agent_id].reverse()
                pos[agent_id].reverse()


            if bt_steps < config.bt_steps:
                pad_len = config.bt_steps-bt_steps
                pad_obs = [ obs_pad for _ in range(pad_len) ]
                pad_pos = [ pos_pad for _ in range(pad_len) ]
                for agent_id in range(num_agents):
                    obs[agent_id] += pad_obs
                    pos[agent_id] += pad_pos


            # next obs_pos
            next_bt_steps = min(bt_steps+forward, config.bt_steps)
            next_i = (i+forward-1)%self._maxsize

            _, _, _, bt_next_obs_pos, _, _, _ = self._storage[next_i]
            for agent_id in range(num_agents):
                next_obs = [ [bt_next_obs_pos[0][agent_id]] for agent_id in range(num_agents) ]
                next_pos = [ [bt_next_obs_pos[1][agent_id]] for agent_id in range(num_agents) ]

            for step in range(next_bt_steps-1):
                bt_next_obs_pos, _, _, _, _, _, _ = self._storage[(next_i-step)%self._maxsize]
                for agent_id in range(num_agents):
                    next_obs[agent_id].append(bt_next_obs_pos[0][agent_id])
                    next_pos[agent_id].append(bt_next_obs_pos[1][agent_id])

            for agent_id in range(num_agents):
                next_obs[agent_id].reverse()
                next_pos[agent_id].reverse()

            if next_bt_steps < config.bt_steps:
                pad_len = config.bt_steps-next_bt_steps
                pad_obs = [ obs_pad for _ in range(pad_len) ]
                pad_pos = [ pos_pad for _ in range(pad_len) ]
                for agent_id in range(num_agents):
                    next_obs[agent_id] += pad_obs
                    next_pos[agent_id] += pad_pos

            b_obs.append(obs)
            b_pos.append(pos)
            b_action += action
            b_reward += sum_reward.tolist()
            b_next_obs.append(next_obs)
            b_next_pos.append(next_pos)

            b_done += [ done for _ in range(num_agents) ]
            b_steps += [ forward for _ in range(num_agents) ]
            b_bt_steps += [ bt_steps for _ in range(num_agents) ]
            b_next_bt_steps += [ next_bt_steps for _ in range(num_agents) ]


        res = (
            torch.from_numpy(np.concatenate(b_obs)).to(self._device),
            torch.from_numpy(np.concatenate(b_pos)).to(self._device),
            torch.LongTensor(b_action).unsqueeze(1).to(self._device),
            torch.FloatTensor(b_reward).unsqueeze(1).to(self._device),
            torch.from_numpy(np.concatenate(b_next_obs)).to(self._device),
            torch.from_numpy(np.concatenate(b_next_pos)).to(self._device),
            torch.FloatTensor(b_done).unsqueeze(1).to(self._device),
            torch.FloatTensor(b_steps).unsqueeze(1).to(self._device),
            torch.LongTensor(b_bt_steps).to(self._device),
            torch.LongTensor(b_next_bt_steps).to(self._device),
        ) 

        return res

    def sample(self, batch_size):
        """Sample a batch of experiences."""
        indexes = range(len(self._storage))
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
        weights = weights.unsqueeze(1).to(self._device)
        encoded_sample = self._encode_sample(idxes)

        super().step()

        return encoded_sample + (weights, idxes)

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions"""

        for idx, priority in zip(idxes, priorities):
            assert (priority > 0).all()
            assert 0 <= idx < len(self._storage)

            self.priority_tree.update(idx, priority**self.alpha)
            self.max_priority = max(self.max_priority, priority)
