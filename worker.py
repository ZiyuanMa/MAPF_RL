import ray
import time
import random
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

import config
from model_dqn import Network
from environment import Environment
from buffer import LocalBuffer, GlobalBuffer
from search import find_path

@ray.remote(num_cpus=1)
class Actor:
    def __init__(self, worker_id, epsilon, learner):
        self.id = worker_id
        self.model = Network()
        self.model.eval()
        self.env = Environment()
        self.epsilon = epsilon
        self.learner = learner
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

            else:
                # sample action
                with torch.no_grad():

                    q_val = self.model.step(torch.FloatTensor(obs_pos[0]), torch.FloatTensor(obs_pos[1]))

                    if self.distributional:
                        q_val = (q_val.exp() * vrange).sum(2)

                    actions = q_val.argmax(1).cpu().tolist()

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
                    q_val = self.model.step(torch.FloatTensor(next_obs_pos[0]), torch.FloatTensor(next_obs_pos[1]))
                    if self.distributional:
                        q_val = (q_val.exp() * vrange).sum(2)
                    buffer.finish(q_val)
                self.learner.add_buffer.remote(buffer)

                done = False
                self.model.reset()

                imitation = True if random.random() < self.imitation_ratio else False
                if imitation:
                    imitation_actions = find_path(self.env)
                    while imitation_actions is None:
                        self.env.reset()
                        imitation_actions = find_path(self.env)

                    obs_pos = self.env.observe()
                    buffer = LocalBuffer(obs_pos, True)
                else:
                    # load weights from learner
                    weights = ray.get(self.learner.get_weights.remote())
                    self.model.load_state_dict(weights)

                    obs_pos = self.env.reset()
                    buffer = LocalBuffer(obs_pos, False)

            return self.id
        
    
@ray.remote(num_cpus=1, num_gpus=1)
class Learner:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Network()
        self.model.to(self.device)
        self.buffer = GlobalBuffer(config.global_buffer_size, self.device)
    
    def add_buffer(self, buffer:LocalBuffer):
        self.buffer.add(buffer)
    
    def get_weights(self):
        return self.model.state_dict()

    def run(self):
        tar_model = deepcopy(self.model)


        min_value = -5
        max_value = 5
        atom_num = 51
        delta_z = 10 / 50
        z_i = torch.linspace(-5, 5, 51).to(device)


        start_ts = time.time()
        for n_iter in range(1, training_timesteps + 1):

            buffer.beta += (1 - prioritized_replay_beta) / training_timesteps
                
            data = generator.__next__()
            buffer.add(data)

            # update self.model
            if n_iter > learning_starts and n_iter % train_freq == 0:
                b_obs, b_pos, b_action, b_reward, b_next_obs, b_next_pos, b_done, b_steps, b_bt_steps, b_next_bt_steps, *extra = buffer.sample(batch_size)

                with torch.no_grad():
                    b_next_dist = tar_model.bootstrap(b_next_obs, b_next_pos, b_next_bt_steps).exp()
                    b_next_action = (b_next_dist * z_i).sum(-1).argmax(1)
                    b_tzj = ((gamma**b_steps) * (1 - b_done) * z_i[None, :] + b_reward).clamp(min_value, max_value)
                    b_i = (b_tzj - min_value) / delta_z
                    b_l = b_i.floor()
                    b_u = b_i.ceil()
                    b_m = torch.zeros(batch_size*config.num_agents, atom_num).to(device)
                    temp = b_next_dist[torch.arange(batch_size*config.num_agents), b_next_action, :]
                    b_m.scatter_add_(1, b_l.long(), temp * (b_u - b_i))
                    b_m.scatter_add_(1, b_u.long(), temp * (b_i - b_l))

                b_q = self.model.bootstrap(b_obs, b_pos, b_bt_steps)[torch.arange(batch_size*config.num_agents), b_action.squeeze(1), :]

                kl_error = (-b_q*b_m).sum(dim=1).reshape(batch_size, config.num_agents).mean(dim=1)
                # kl_error = kl_div(b_q, b_m, reduction='none').sum(dim=1).reshape(batch_size, config.num_agents).mean(dim=1)
                # use kl error as priorities as proposed by Rainbow
                priorities = kl_error.detach().cpu().clamp(1e-6).numpy()
                loss = kl_error.mean()

                optimizer.zero_grad()

                loss.backward()
                # scaler.scale(loss).backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), 40)

                optimizer.step()
                # scaler.step(optimizer)
                # scaler.update()

                scheduler.step()


                buffer.update_priorities(extra[1], priorities)

            # update target net and log
            if n_iter % target_network_update_freq == 0:
                tar_model.load_state_dict(self.model.state_dict())

                print('{} Iter {} {}'.format('=' * 10, n_iter, '=' * 10))
                fps = int(target_network_update_freq / (time.time() - start_ts))
                start_ts = time.time()
                print('FPS {}'.format(fps))

                if n_iter > learning_starts:
                    print('vloss: {:.6f}'.format(loss.item()))
                
            # save model
            if save_interval and n_iter % save_interval == 0:
                torch.save(self.model.state_dict(), os.path.join(save_path, '{}.pth'.format(n_iter)))

        torch.save(self.model.state_dict(), os.path.join(save_path, 'model.pth'))