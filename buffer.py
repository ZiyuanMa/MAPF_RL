import operator
import random
from typing import List
import numpy as np
import torch
import math
import threading

import config

discounts = np.array([[0.99**i] for i in range(config.max_steps)])

class SumTree:
    def __init__(self, capacity, priorities=None):

        layer = 1
        while 2**(layer-1) < capacity:
            layer += 1
        assert 2**(layer-1) == capacity, 'buffer size only support power of 2 size'
        self.layer = layer
        self.tree = np.zeros(2**layer-1)
        self.capacity = capacity
        if priorities is not None:
            self.size = len(priorities)
        else:
            self.size = 0

        if priorities is not None:
            idx = np.array([ self.capacity-1+i for i in range(len(priorities)) ], dtype=np.int)
            self.tree[idx] = priorities

            for _ in range(layer-1):
                idx = (idx-1) // 2
                idx = np.unique(idx)
                self.tree[idx] = self.tree[2*idx+1] + self.tree[2*idx+2]
            
            # check
            assert int(np.sum(self.tree[-self.capacity:])) == int(self.tree[0]), 'sum is {} but root is {}'.format(np.sum(self.tree[-self.capacity:]), self.tree[0])


    def sum(self):
        assert int(np.sum(self.tree[-self.capacity:])) == int(self.tree[0]), 'sum is {} but root is {}'.format(np.sum(self.tree[-self.capacity:]), self.tree[0])
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

    def update(self, idx:int, priority:float):
        assert 0 <= idx < self.capacity
        # self.tree.flags.writeable = True

        idx += self.capacity-1

        self.tree[idx] = priority

        idx = (idx-1) // 2
        while idx >= 0:
            self.tree[idx] = self.tree[2*idx+1] + self.tree[2*idx+2]
            idx = (idx-1) // 2
        
        assert int(np.sum(self.tree[-self.capacity:])) == int(self.tree[0]), 'sum is {} but root is {}'.format(np.sum(self.tree[-self.capacity:]), self.tree[0])
        
    def batch_update(self, idxes:np.ndarray, priorities:np.ndarray):
        idxes += self.capacity-1
        self.tree[idxes] = priorities

        for _ in range(self.layer-1):
            idxes = (idxes-1) // 2
            idxes = np.unique(idxes)
            self.tree[idxes] = self.tree[2*idxes+1] + self.tree[2*idxes+2]
        
        # check
        assert int(np.sum(self.tree[-self.capacity:])) == int(self.tree[0]), 'sum is {} but root is {}'.format(np.sum(self.tree[-self.capacity:]), self.tree[0])


class LocalBuffer:
    def __init__(self, init_obs_pos, imitation:bool, size=config.max_steps, alpha=config.prioritized_replay_alpha, beta=config.prioritized_replay_beta):
        """
        Prioritized Replay buffer for each actor

        """
        assert alpha >= 0
        self.alpha = alpha
        self.beta = beta

        self.num_agents = init_obs_pos[0].shape[0]
        # observation length should be (max steps+1)
        self.obs_buf = np.zeros((size+1, self.num_agents, *config.obs_shape), dtype=np.bool)
        self.pos_buf = np.zeros((size+1, self.num_agents, 4), dtype=np.uint8)
        self.act_buf = np.zeros((size, self.num_agents), dtype=np.long)
        self.rew_buf = np.zeros((size, self.num_agents), dtype=np.float32)
        self.q_buf = np.zeros((size+1, self.num_agents, 5), dtype=np.float32)

        self.capacity = size
        self.size = 0
        self.imitation = imitation

        self.obs_buf[0], self.pos_buf[0] = init_obs_pos
    
    def __len__(self):
        return self.size

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


        if self.imitation:
            assert self.done, 'size {}'.format(self.size)

        priorities = np.zeros((self.size, self.num_agents), dtype=np.float32)

        if self.imitation:
            for i in range(self.size):
                # n-steps forward = 4
                forward = min(4, self.size-i)
                reward = np.sum(self.rew_buf[i:i+forward]*discounts[:forward], axis=0)+(0.99**forward)*np.max(self.q_buf[i+forward], axis=1)
                q_val = self.q_buf[i,[0,1],self.act_buf[i]]
                priorities[i] = np.abs(reward-q_val)
            priorities = np.mean(priorities, axis=1)

        else:
            for i in range(self.size):
                # forward = 1
                reward = self.rew_buf[i]+0.99*np.max(self.q_buf[i+1], axis=1)
                q_val = self.q_buf[i,[0,1],self.act_buf[i]]
                priorities[i] = np.abs(reward-q_val)
            priorities = np.mean(priorities, axis=1)

        self.priority_tree = SumTree(self.capacity, priorities)
        self.priority = self.priority_tree.sum()
        
    def sample(self, prefixsum):

        idx, _ = self.priority_tree.find_prefixsum_idx(prefixsum)

        assert 0 <= idx < self.size

        priority = self.priority_tree[idx]

        encoded_sample = self._encode_sample(idx)

        return encoded_sample + (idx, priority)

    def update_priority(self, idx, priority):
        assert 0 <= idx < self.size, 'idx {} out of size {}'.format(idx, self.size)

        self.priority_tree.update(idx, priority**self.alpha)
        self.priority = self.priority_tree.sum()


    def _encode_sample(self, idx):

        if self.imitation:
            # n-step forward = 4
            forward = min(4, self.size-idx)
            reward = np.sum(self.rew_buf[idx:idx+forward]*discounts[:forward], axis=0)

        else:
            # self play
            forward = 1
            reward = self.rew_buf[idx]

        if self.done and idx+forward == self.size:
            done = True
        else:
            done = False

        # obs and pos
        bt_steps = min(idx+1, config.bt_steps)
        obs = np.swapaxes(self.obs_buf[idx+1-bt_steps:idx+1], 0, 1)
        # obs = obs.reshape(self.num_agents*bt_steps, *config.obs_shape)
        pos = np.swapaxes(self.pos_buf[idx+1-bt_steps:idx+1], 0, 1)
        # pos = pos.reshape(self.num_agents*bt_steps, 4)
        if bt_steps < config.bt_steps:
            pad_len = config.bt_steps-bt_steps
            obs = np.pad(obs, ((0,0),(0,pad_len),(0,0),(0,0),(0,0)))
            pos = np.pad(pos, ((0,0),(0,pad_len),(0,0)))

        # next obs and next pos
        next_bt_steps = min(idx+1+forward, config.bt_steps)
        next_obs = np.swapaxes(self.obs_buf[idx+1+forward-next_bt_steps:idx+1+forward], 0, 1)
        # next_obs = next_obs.reshape(self.num_agents*next_bt_steps, *config.obs_shape)
        next_pos = np.swapaxes(self.pos_buf[idx+1+forward-next_bt_steps:idx+1+forward], 0, 1)
        # next_pos = next_pos.reshape(self.num_agents*next_bt_steps, 4)
        if next_bt_steps < config.bt_steps:
            pad_len = config.bt_steps-next_bt_steps
            next_obs = np.pad(next_obs, ((0,0),(0,pad_len),(0,0),(0,0),(0,0)))
            next_pos = np.pad(next_pos, ((0,0),(0,pad_len),(0,0)))

        # define other part
        dones = [ done for _ in range(self.num_agents) ]
        steps = [ forward for _ in range(self.num_agents) ]
        bt_steps = [ bt_steps for _ in range(self.num_agents) ]
        next_bt_steps = [ next_bt_steps for _ in range(self.num_agents) ]

        return obs, pos, self.act_buf[idx].tolist(), reward.tolist(), next_obs, next_pos, dones, steps, bt_steps, next_bt_steps



class GlobalBuffer:
    def __init__(self, capacity, beta=config.prioritized_replay_beta):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        self.buffer = [ None for _ in range(capacity) ]
        self.priority_tree = SumTree(capacity)
        self.beta = beta
        self.counter = 0
        self.lock = threading.Lock()

    def __len__(self):
        return self.size

    def add(self, buffer:LocalBuffer):
        with self.lock:
            # update buffer size
            if self.buffer[self.ptr] is not None:
                self.size -= len(self.buffer[self.ptr])
            self.size += len(buffer)
            self.counter += len(buffer)

            buffer.priority_tree.tree.flags.writeable = True

            self.buffer[self.ptr] = buffer

            # print('tree add 0')
            self.priority_tree.update(self.ptr, buffer.priority)
            # print('tree add 1')
            # print('ptr: {}, current size: {}, add priority: {}, current: {}'.format(self.ptr, self.size, buffer.priority, self.priority_tree.sum()))

            self.ptr = (self.ptr+1) % self.capacity
    
    def batch_add(self, buffers:List[LocalBuffer]):
        for buffer in buffers:
            self.add(buffer)

    def sample_batch(self, batch_size):
        with self.lock:
            if len(self) < config.learning_starts:
                return None

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
                torch.from_numpy(weights).unsqueeze(1)
            )

            self.temp = self.ptr

        return data

    def update_priorities(self, idxes:List[int], priorities:List[float]):
        """Update priorities of sampled transitions"""
        with self.lock:
            idxes = np.asarray(idxes)
            priorities = np.asarray(priorities)

            # discard the idx that already been discarded during training
            if self.ptr > self.temp:
                # range from [self.temp, self.ptr)
                mask = (idxes < self.temp*config.max_steps) | (idxes >= self.ptr*config.max_steps)
                idxes = idxes[mask]
                priorities = priorities[mask]
            elif self.ptr < self.temp:
                # range from [0, self.ptr) & [self.temp, self,capacity)
                mask = (idxes < self.temp*config.max_steps) & (idxes >= self.ptr*config.max_steps)
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