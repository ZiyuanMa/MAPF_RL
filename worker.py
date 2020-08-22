import ray
import time
import random
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.benchmark=True
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from copy import deepcopy
from typing import List, Tuple
import threading

import config
from model import Network
from environment import Environment
from buffer import SumTree, LocalBuffer

@ray.remote(num_cpus=1)
class GlobalBuffer:
    def __init__(self, capacity, alpha=config.prioritized_replay_alpha, beta=config.prioritized_replay_beta):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        self.buffer = [ None for _ in range(capacity) ]
        self.priority_tree = SumTree(capacity*config.local_buffer_size)
        self.alpha = alpha
        self.beta = beta
        self.counter = 0
        self.data = []
        self.stat_dict = {config.init_set:[]}
        self.lock = threading.Lock()
        self.level = ray.put([config.init_set])

        self.obs_buf = np.zeros(((config.max_steps+1)*capacity, config.max_num_agetns, *config.obs_shape), dtype=np.bool)
        self.act_buf = np.zeros((config.max_steps*capacity), dtype=np.uint8)
        self.rew_buf = np.zeros((config.max_steps*capacity), dtype=np.float32)
        self.hid_buf = np.zeros((config.max_steps*capacity, config.max_num_agetns, config.latent_dim), dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.bool)
        self.size_buf = np.zeros(capacity, dtype=np.uint)
        self.comm_mask = np.zeros(((config.max_steps+1)*capacity, config.max_num_agetns, config.max_num_agetns), dtype=np.bool)

    def __len__(self):
        return self.size

    def run(self):
        self.background_thread = threading.Thread(target=self.prepare_data, daemon=True)
        self.background_thread.start()

    def prepare_data(self):
        while True:
            if len(self.data) <= 4:
                data = self.sample_batch(config.batch_size)
                data_id = ray.put(data)
                self.data.append(data_id)
            else:
                time.sleep(0.1)
    
    def get_data(self):

        if len(self.data) == 0:
            print('no prepared data')
            data = self.sample_batch(config.batch_size)
            data_id = ray.put(data)
            return data_id
        else:
            return self.data.pop(0)


    def add(self, buffer_list:List):
        # actor_id 0, num_agents 1, map_len 2, obs_buf 3, act_buf 4, rew_buf 5, hid_buf 6, td_errors 7, done 8, size 9, comm_mask 10
        for buffer in buffer_list:
            if buffer[0] >= 10:
                stat_key = (buffer[1], buffer[2])

                if stat_key in self.stat_dict:
                    if len(self.stat_dict[stat_key]) < 200:
                        self.stat_dict[stat_key].append(buffer[8])
                    else:
                        self.stat_dict[stat_key].pop(0)
                        self.stat_dict[stat_key].append(buffer[8])

        with self.lock:

            for buffer in buffer_list:
                idxes = np.arange(self.ptr*config.local_buffer_size, (self.ptr+1)*config.local_buffer_size)
                start_idx = self.ptr*config.max_steps
                # update buffer size
                self.size -= self.size_buf[self.ptr].item()
                self.size += buffer[9]
                self.counter += buffer[9]

                self.priority_tree.batch_update(idxes, buffer[7]**self.alpha)

                self.obs_buf[start_idx+self.ptr:start_idx+self.ptr+buffer[9]+1, :buffer[1]] = buffer[3]
                self.act_buf[start_idx:start_idx+buffer[9]] = buffer[4]
                self.rew_buf[start_idx:start_idx+buffer[9]] = buffer[5]
                self.hid_buf[start_idx:start_idx+buffer[9], :buffer[1]] = buffer[6]
                self.done_buf[self.ptr] = buffer[8]
                self.size_buf[self.ptr] = buffer[9]
                self.comm_mask[start_idx+self.ptr:start_idx+self.ptr+buffer[9]+1, :buffer[1], :buffer[1]] = buffer[10]

                self.ptr = (self.ptr+1) % self.capacity

    def sample_batch(self, batch_size:int) -> Tuple:

        b_obs, b_action, b_reward, b_done, b_steps, b_bt_steps, b_comm_mask = [], [], [], [], [], [], []
        idxes, priorities = [], []
        b_hidden = []

        with self.lock:

            idxes, priorities = self.priority_tree.batch_sample(batch_size)
            global_idxes = idxes // config.local_buffer_size
            local_idxes = idxes % config.local_buffer_size

            for idx, global_idx, local_idx in zip(idxes, global_idxes, local_idxes):
                
                assert local_idx < self.size_buf[global_idx]

                steps = int(min(config.forward_steps, (self.size_buf[global_idx]-local_idx).item()))

                if local_idx < config.bt_steps-1:
                    obs = self.obs_buf[global_idx*(config.max_steps+1):idx+global_idx+1+steps]
                    comm_mask = self.comm_mask[global_idx*(config.max_steps+1):idx+global_idx+1+steps]
                    hidden = np.zeros((config.max_num_agetns, config.latent_dim), dtype=np.float32)

                elif local_idx == config.bt_steps-1:
                    obs = self.obs_buf[idx+global_idx+1-config.bt_steps:idx+global_idx+1+steps]
                    comm_mask = self.comm_mask[global_idx*(config.max_steps+1):idx+global_idx+1+steps]
                    hidden = np.zeros((config.max_num_agetns, config.latent_dim), dtype=np.float32)

                else:
                    obs = self.obs_buf[idx+global_idx+1-config.bt_steps:idx+global_idx+1+steps]
                    comm_mask = self.comm_mask[idx+global_idx+1-config.bt_steps:idx+global_idx+1+steps]
                    hidden = self.hid_buf[idx-config.bt_steps-1]

                if obs.shape[0] < config.bt_steps+config.forward_steps:
                    pad_len = config.bt_steps+config.forward_steps-obs.shape[0]
                    obs = np.pad(obs, ((0,pad_len),(0,0),(0,0),(0,0),(0,0)))
                    comm_mask = np.pad(comm_mask, ((0,pad_len),(0,0),(0,0)))

                action = self.act_buf[idx]
                reward = self.rew_buf[idx]
                if local_idx==self.size_buf[global_idx]-1 and self.done_buf[global_idx]:
                    done = True
                else:
                    done = False

                bt_steps = min(local_idx+1, config.bt_steps)
                
                b_obs.append(obs)
                b_action.append(action)
                b_reward.append(reward)

                b_done.append(done)
                b_steps.append(steps)
                b_bt_steps.append(bt_steps)
                b_comm_mask.append(comm_mask)

                b_hidden.append(hidden)

            # importance sampling weights
            min_p = np.min(priorities)
            weights = np.power(priorities/min_p, -self.beta)

            data = (
                torch.from_numpy(np.stack(b_obs).astype(np.float32)),
                torch.LongTensor(b_action).unsqueeze(1),
                torch.FloatTensor(b_reward).unsqueeze(1),

                torch.FloatTensor(b_done).unsqueeze(1),
                torch.FloatTensor(b_steps).unsqueeze(1),
                torch.LongTensor(b_bt_steps),
                torch.from_numpy(np.concatenate(b_hidden)),
                torch.from_numpy(np.stack(b_comm_mask)),

                idxes,
                torch.from_numpy(weights).unsqueeze(1),
                self.ptr
            )

            return data

    def update_priorities(self, idxes:np.ndarray, priorities:np.ndarray, old_ptr:int):
        """Update priorities of sampled transitions"""
        with self.lock:
            idxes.flags.writeable = True
            priorities.flags.writeable = True
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

            self.priority_tree.batch_update(idxes, priorities**self.alpha)

    def stats(self, interval:int):
        print('buffer update speed: {}/s'.format(self.counter/interval))
        print('buffer size: {}'.format(self.size))

        for key, val in self.stat_dict.copy().items():
            print('{}: {}/{}'.format(key, sum(val), len(val)))
            if len(val) == 200 and sum(val) >= 200*config.pass_rate:
                # add number of agents
                add_agent_key = (key[0]+1, key[1]) 
                if add_agent_key[0] <= config.max_num_agetns and add_agent_key not in self.stat_dict:
                    self.stat_dict[add_agent_key] = []
                
                if key[1] < config.max_map_lenght:
                    add_map_key = (key[0], key[1]+5) 
                    if add_map_key not in self.stat_dict:
                        self.stat_dict[add_map_key] = []
                
                    del self.stat_dict[key]

        self.level = ray.put(list(self.stat_dict.keys()))

        self.counter = 0

    def ready(self):
        if len(self) >= config.learning_starts:
            return True
        else:
            return False
    
    def get_level(self):
        return self.level

    def check_done(self):

        for i in range(config.max_num_agetns):
            if (i+1, config.max_map_lenght) not in self.stat_dict:
                return False
        
            l = self.stat_dict[(i+1, config.max_map_lenght)]
            
            if len(l) < 200:
                return False
            elif sum(l) < 200*config.pass_rate:
                return False
            
        return True

@ray.remote(num_cpus=1, num_gpus=1)
class Learner:
    def __init__(self, buffer:GlobalBuffer):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Network()
        self.model.to(self.device)
        self.tar_model = deepcopy(self.model)
        self.optimizer = Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[100000, 300000], gamma=0.5)
        self.buffer = buffer
        self.counter = 0
        self.last_counter = 0
        self.done = False
        self.loss = 0
        taus = torch.arange(0, 200+1, device=self.device, dtype=torch.float32) / 200
        taus = ((taus[1:] + taus[:-1]) / 2.0).view(1, 200, 1)
        self.taus = taus.expand(config.batch_size, 200, 200)

        self.store_weights()

    def get_weights(self):
        return self.weights_id

    def store_weights(self):
        state_dict = self.model.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        self.weights_id = ray.put(state_dict)

    def run(self):
        self.learning_thread = threading.Thread(target=self.train, daemon=True)
        self.learning_thread.start()

    def train(self):
        while not ray.get(self.buffer.check_done.remote()):
            for i in range(1, 10001):

                data_id = ray.get(self.buffer.get_data.remote())
                data = ray.get(data_id)
    
                b_obs, b_action, b_reward, b_done, b_steps, b_bt_steps, b_hidden, b_comm_mask, idxes, weights, old_ptr = data
                b_obs, b_action, b_reward = b_obs.to(self.device), b_action.to(self.device), b_reward.to(self.device)
                b_done, b_steps, weights = b_done.to(self.device), b_steps.to(self.device), weights.to(self.device)
                b_hidden = b_hidden.to(self.device)
                b_comm_mask = b_comm_mask.to(self.device)

                b_next_bt_steps = [ bt_steps+steps.item() for bt_steps, steps in zip(b_bt_steps, b_steps) ]

                if config.distributional:
                    raise NotImplementedError
                    # with torch.no_grad():
                    #     b_next_dist = self.tar_model.bootstrap(b_obs[:, 1:], b_pos[:, 1:], b_next_bt_steps, b_next_hidden)
                    #     b_next_action = b_next_dist.mean(dim=2).argmax(dim=1)
                    #     b_next_dist = b_next_dist[batch_idx, b_next_action, :]

                    # b_dist = self.model.bootstrap(b_obs, b_pos, b_bt_steps, b_hidden)
                    # b_dist = b_dist[batch_idx, torch.squeeze(b_action), :]

                    # b_target_dist = b_reward + (1-b_done)*(config.gamma**b_steps)*b_next_dist

                    # # batch_size * N * 1
                    # b_dist = b_dist.unsqueeze(2)
                    # # batch_size * 1 * N
                    # b_target_dist = b_target_dist.unsqueeze(1)

                    # td_errors = b_target_dist-b_dist
                    # priorities, loss = self.quantile_huber_loss(td_errors, weights=weights)

                else:
                    with torch.no_grad():
                        # choose max q index from next observation
                        # double q-learning
                        if config.double_q:
                            b_action_ = self.model.bootstrap(b_obs, b_next_bt_steps, b_hidden, b_comm_mask).argmax(1, keepdim=True)
                            b_q_ = (1 - b_done) * self.tar_model.bootstrap(b_obs, b_next_bt_steps, b_hidden, b_comm_mask).gather(1, b_action_)
                        else:
                            b_q_ = (1 - b_done) * self.tar_model.bootstrap(b_obs, b_next_bt_steps, b_hidden, b_comm_mask).max(1, keepdim=True)[0]

                    b_q = self.model.bootstrap(b_obs[:, :-config.forward_steps], b_bt_steps, b_hidden, b_comm_mask[:, :-config.forward_steps]).gather(1, b_action)

                    td_error = (b_q - (b_reward + (0.99 ** b_steps) * b_q_))

                    priorities = td_error.detach().squeeze().abs().cpu().clamp(1e-6).numpy()

                    loss = (weights * self.huber_loss(td_error)).mean()

                self.optimizer.zero_grad()

                loss.backward()
                self.loss = loss.item()
                # scaler.scale(loss).backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), 40)

                self.optimizer.step()
                # scaler.step(optimizer)
                # scaler.update()
                self.scheduler.step()


                # store new weights in shared memory
                if i % 5  == 0:
                    self.store_weights()

                self.buffer.update_priorities.remote(idxes, priorities, old_ptr)

                self.counter += 1

                # update target net, save model
                if i % config.target_network_update_freq == 0:
                    self.tar_model.load_state_dict(self.model.state_dict())
                    torch.save(self.model.state_dict(), os.path.join(config.save_path, '{}.pth'.format(i)))
                
                # if i == 10000:
                #     config.imitation_ratio = 0

        self.done = True
    def huber_loss(self, td_error, kappa=1.0):
        abs_td_error = td_error.abs()
        flag = (abs_td_error < kappa).float()
        return flag * abs_td_error.pow(2) * 0.5 + (1 - flag) * (abs_td_error - 0.5)

    def quantile_huber_loss(self, td_errors, weights=None, kappa=1.0):

        element_wise_huber_loss = self.huber_loss(td_errors, kappa)
        assert element_wise_huber_loss.shape == (config.batch_size, 200, 200)

        element_wise_quantile_huber_loss = torch.abs(self.taus - (td_errors.detach() < 0).float()) * element_wise_huber_loss / kappa
        assert element_wise_quantile_huber_loss.shape == (config.batch_size, 200, 200)

        batch_quantile_huber_loss = element_wise_quantile_huber_loss.sum(dim=1).mean(dim=1, keepdim=True)
        assert batch_quantile_huber_loss.shape == (config.batch_size, 1)

        priorities = batch_quantile_huber_loss.detach().cpu().clamp(1e-6).numpy()

        if weights is not None:
            quantile_huber_loss = (batch_quantile_huber_loss * weights).mean()
        else:
            quantile_huber_loss = batch_quantile_huber_loss.mean()

        return priorities, quantile_huber_loss

    def stats(self, interval:int):
        print('number of updates: {}'.format(self.counter))
        print('update speed: {}/s'.format((self.counter-self.last_counter)/interval))
        print('loss: {}'.format(self.loss))
        self.last_counter = self.counter
        return self.done


@ray.remote(num_cpus=1)
class Actor:
    def __init__(self, worker_id, epsilon, learner:Learner, buffer:GlobalBuffer):
        self.id = worker_id
        self.model = Network()
        self.model.eval()
        self.env = Environment(adaptive=True)
        self.epsilon = epsilon
        self.learner = learner
        self.global_buffer = buffer
        self.max_steps = config.max_steps
        self.counter = 0

    def run(self):
        """ Generate training batch sample """
        done = False
        buffer_list = []
        obs_pos, local_buffer = self.reset()

        while True:

            # sample action
            # Note: q_val is quantile values if it's distributional
            actions, q_val, hidden, comm_mask = self.model.step(torch.from_numpy(obs_pos[0].astype(np.float32)), torch.from_numpy(obs_pos[1].astype(np.float32)))

            if random.random() < self.epsilon:
                # Note: only one agent can do random action in order to make the whole environment more stable
                actions[0] = np.random.randint(0, 5)

            # take action in env
            next_obs_pos, r, done, _ = self.env.step(actions)

            # return data and update observation
            local_buffer.add(q_val[0], actions[0], r[0], next_obs_pos[0], hidden[0], comm_mask)

            if done == False and self.env.steps < self.max_steps:

                obs_pos = next_obs_pos
            else:
                # finish and send buffer
                if done:
                    buffer = local_buffer.finish()
                else:

                    _, q_val, _, comm_mask = self.model.step(torch.from_numpy(obs_pos[0].astype(np.float32)), torch.from_numpy(obs_pos[1].astype(np.float32)))

                    buffer = local_buffer.finish(q_val[0], comm_mask)

                buffer_list.append(buffer)

                done = False

                obs_pos, local_buffer = self.reset()

            self.counter += 1
            if self.counter == config.actor_update_steps:
                self.global_buffer.add.remote(buffer_list)
                buffer_list.clear()
                self.update_weights()
                self.counter = 0

    def update_weights(self):
        '''load weights from learner'''
        weights_id = ray.get(self.learner.get_weights.remote())
        weights = ray.get(weights_id)
        self.model.load_state_dict(weights)
    
    def reset(self):
        self.model.reset()
        level_id = ray.get(self.global_buffer.get_level.remote())
        obs_pos = self.env.reset(ray.get(level_id))
        local_buffer = LocalBuffer(self.id, self.env.num_agents, self.env.map_size[0], obs_pos[0])

        return obs_pos, local_buffer

