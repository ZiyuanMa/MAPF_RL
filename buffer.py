"""
This file is adapted from following files in openai/baselines.
deepq/replay_buffer.py
baselines/acer/buffer.py
"""
import operator
import random

import numpy as np
import torch
import math

import config

obs_pad = np.zeros((config.obs_dim, 9, 9), dtype=config.dtype)
pos_pad = np.zeros((config.pos_dim), dtype=config.dtype)

# class SegmentTree(object):
#     def __init__(self, capacity, operation, neutral_element):
#         """Build a Segment Tree data structure.

#         https://en.wikipedia.org/wiki/Segment_tree

#         Can be used as regular array, but with two
#         important differences:

#             a) setting item's value is slightly slower.
#                It is O(lg capacity) instead of O(1).
#             b) user has access to an efficient ( O(log segment size) )
#                `reduce` operation which reduces `operation` over
#                a contiguous subsequence of items in the array.

#         Paramters
#         ---------
#         capacity: int
#             Total size of the array - must be a power of two.
#         operation: lambda obj, obj -> obj
#             and operation for combining elements (eg. sum, max)
#             must form a mathematical group together with the set of
#             possible values for array elements (i.e. be associative)
#         neutral_element: obj
#             neutral element for the operation above. eg. float('-inf')
#             for max and 0 for sum.
#         """
#         assert capacity > 0 and capacity & (capacity - 1) == 0, \
#             "capacity must be positive and a power of 2."
#         self._capacity = capacity
#         self._value = [neutral_element for _ in range(2 * capacity)]
#         self._operation = operation

#     def _reduce_helper(self, start, end, node, node_start, node_end):
#         if start == node_start and end == node_end:
#             return self._value[node]
#         mid = (node_start + node_end) // 2
#         if end <= mid:
#             return self._reduce_helper(start, end, 2 * node, node_start, mid)
#         else:
#             if mid + 1 <= start:
#                 return self._reduce_helper(start, end,
#                                            2 * node + 1, mid + 1, node_end)
#             else:
#                 return self._operation(
#                     self._reduce_helper(start, mid,
#                                         2 * node, node_start, mid),
#                     self._reduce_helper(mid + 1, end,
#                                         2 * node + 1, mid + 1, node_end)
#                 )

#     def reduce(self, start=0, end=None):
#         """Returns result of applying `self.operation`
#         to a contiguous subsequence of the array.

#         Parameters
#         ----------
#         start: int
#             beginning of the subsequence
#         end: int
#             end of the subsequences

#         Returns
#         -------
#         reduced: obj
#             result of reducing self.operation over the specified range of array.
#         """
#         if end is None:
#             end = self._capacity
#         if end < 0:
#             end += self._capacity
#         end -= 1
#         return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

#     def __setitem__(self, idx, val):
#         # index of the leaf
#         idx += self._capacity
#         self._value[idx] = val
#         idx //= 2
#         while idx >= 1:
#             self._value[idx] = self._operation(
#                 self._value[2 * idx],
#                 self._value[2 * idx + 1]
#             )
#             idx //= 2

#     def __getitem__(self, idx):
#         assert 0 <= idx < self._capacity
#         return self._value[self._capacity + idx]


# class SumSegmentTree(SegmentTree):
#     def __init__(self, capacity):
#         super(SumSegmentTree, self).__init__(
#             capacity=capacity,
#             operation=operator.add,
#             neutral_element=0.0
#         )

#     def sum(self, start=0, end=None):
#         """Returns arr[start] + ... + arr[end]"""
#         return super(SumSegmentTree, self).reduce(start, end)

#     def find_prefixsum_idx(self, prefixsum):
#         """Find the highest index `i` in the array such that
#             sum(arr[0] + arr[1] + ... + arr[i]) <= prefixsum

#         if array values are probabilities, this function
#         allows to sample indexes according to the discrete
#         probability efficiently.

#         Parameters
#         ----------
#         perfixsum: float
#             upperbound on the sum of array prefix

#         Returns
#         -------
#         idx: int
#             highest index satisfying the prefixsum constraint
#         """
#         assert 0 <= prefixsum <= self.sum() + 1e-5
#         idx = 1
#         while idx < self._capacity:  # while non-leaf
#             if self._value[2 * idx] > prefixsum:
#                 idx = 2 * idx
#             else:
#                 prefixsum -= self._value[2 * idx]
#                 idx = 2 * idx + 1
#         return idx - self._capacity


# class MinSegmentTree(SegmentTree):
#     def __init__(self, capacity):
#         super(MinSegmentTree, self).__init__(
#             capacity=capacity,
#             operation=min,
#             neutral_element=float('inf')
#         )

#     def min(self, start=0, end=None):
#         """Returns min(arr[start], ...,  arr[end])"""

#         return super(MinSegmentTree, self).reduce(start, end)

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
        if self.size < self.capacity:
            return np.sum(self.tree[-self.capacity:-self.capacity+self.size])
        else:
            np.sum(self.tree[-self.capacity:])

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
                        _, _, next_reward, _, done, _, info = self._storage[next_idx]

                        if info['step'] == 0:
                            break

                        sum_reward += np.array(next_reward, dtype=np.float32) * (0.99 ** j)
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
            next_bt_steps = min(bt_steps+forward-1, config.bt_steps)
            next_i = (i+forward-1)%self._maxsize

            _, _, _, bt_next_obs_pos, _, _, _ = self._storage[next_i]
            for agent_id in range(num_agents):
                    next_obs = [ [bt_next_obs_pos[0][agent_id]] for agent_id in range(num_agents) ]
                    next_pos = [ [bt_next_obs_pos[1][agent_id]] for agent_id in range(num_agents) ]

            for step in range(next_bt_steps):
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


        self.counter += 1
        if self.counter == 125000:
            self.counter = 0
            self.n_step += 1

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
            _, _, _, _, _, _, info = self._storage[idx]
            while info['step'] == 0:
                idx = random.choice(indexes)
                _, _, _, _, _, _, info = self._storage[idx]
            idxes.append(idx)

        return self._encode_sample(idxes)


# class PrioritizedReplayBuffer(ReplayBuffer):
#     def __init__(self, size, device, alpha, beta):
#         """Create Prioritized Replay buffer.

#         Parameters
#         ----------
#         size: int
#             Max number of transitions to store in the buffer. When the buffer
#             overflows the old memories are dropped.
#         alpha: float
#             how much prioritization is used
#             (0 - no prioritization, 1 - full prioritization)

#         See Also
#         --------
#         ReplayBuffer.__init__
#         """
#         super(PrioritizedReplayBuffer, self).__init__(size, device)
#         assert alpha >= 0
#         self._alpha = alpha

#         it_capacity = 1
#         while it_capacity < size:
#             it_capacity *= 2

#         self._it_sum = SumSegmentTree(it_capacity)
#         self._it_min = MinSegmentTree(it_capacity)
#         self._max_priority = 1.0
#         self.beta = beta

#     def add(self, *args, **kwargs):
#         """See ReplayBuffer.store_effect"""
#         idx = self._next_idx
#         super().add(*args, **kwargs)
#         self._it_sum[idx] = self._max_priority ** self._alpha
#         self._it_min[idx] = self._max_priority ** self._alpha

#     def _sample_proportional(self, batch_size):
#         res = []
#         p_total = self._it_sum.sum(0, len(self._storage)-1)
#         every_range_len = p_total / batch_size
#         for i in range(batch_size):
#             mass = random.random() * every_range_len + i * every_range_len
#             idx = self._it_sum.find_prefixsum_idx(mass)

#             res.append(idx)
            
#         return res

#     def sample(self, batch_size):
#         """Sample a batch of experiences"""
#         idxes = self._sample_proportional(batch_size)

#         it_sum = self._it_sum.sum()
#         p_min = self._it_min.min() / it_sum
#         max_weight = (p_min * len(self._storage)) ** (-self.beta)

#         p_samples = np.asarray([self._it_sum[idx] for idx in idxes]) / it_sum
#         weights = (p_samples * len(self._storage)) ** (-self.beta) / max_weight
#         weights = torch.from_numpy(weights.astype('float32'))
#         weights = weights.to(self._device).unsqueeze(1)
#         encoded_sample = self._encode_sample(idxes)
#         return encoded_sample + (weights, idxes)

#     def update_priorities(self, idxes, priorities):
#         """Update priorities of sampled transitions"""
#         assert len(idxes) == len(priorities)
#         for idx, priority in zip(idxes, priorities):
#             assert (priority > 0).all()
#             assert 0 <= idx < len(self._storage)
#             self._it_sum[idx] = priority ** self._alpha
#             self._it_min[idx] = priority ** self._alpha

#             self._max_priority = max(self._max_priority, priority)



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
        idx = self._next_idx
        super().add(*args, **kwargs)

        self.priority_tree[idx] = self.max_priority ** self.alpha


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
        return encoded_sample + (weights, idxes)

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions"""

        for idx, priority in zip(idxes, priorities):
            assert (priority > 0).all()
            assert 0 <= idx < len(self._storage)

            self.priority_tree[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)
