import numpy as np

import matplotlib.pyplot as plt
plt.ion()
from matplotlib import colors
import random
from typing import List, Union

import config


action_list = np.array([[0, 0],[-1, 0],[1, 0],[0, -1],[0, 1]], dtype=np.int)

color_map = np.array([[255, 255, 255],   # white
                    [190, 190, 190],   # gray
                    [0, 191, 255],   # blue
                    [255, 165, 0],   # orange
                    [0, 250, 154]])  # green


def map_partition(map):
    '''
    partition map into independent partitions so assign agent position and goal position of one agent in the same partition 
    '''

    empty_pos = np.argwhere(map==0).astype(np.int).tolist()

    empty_pos = [ tuple(pos) for pos in empty_pos ]

    if not empty_pos:
        raise RuntimeError('no empty position')

    partition_list = list()
    while empty_pos:

        start_pos = empty_pos.pop()

        open_list = list()
        open_list.append(start_pos)
        close_list = list()

        while open_list:
            x, y = open_list.pop(0)

            up = x-1, y
            if up[0] >= 0 and map[up]==0 and up in empty_pos:
                empty_pos.remove(up)
                open_list.append(up)
            
            down = x+1, y
            if down[0] < map.shape[0] and map[down]==0 and down in empty_pos:
                empty_pos.remove(down)
                open_list.append(down)
            
            left = x, y-1
            if left[1] >= 0 and map[left]==0 and left in empty_pos:
                empty_pos.remove(left)
                open_list.append(left)
            
            right = x, y+1
            if right[1] < map.shape[1] and map[right]==0 and right in empty_pos:
                empty_pos.remove(right)
                open_list.append(right)

            close_list.append((x, y))


        partition_list.append(close_list)

    return partition_list
    


class Environment:
    def __init__(self, adaptive=False, map_size:int=config.map_length, num_agents:int=config.num_agents,
                obs_radius:int=config.obs_radius, reward_fn:dict=config.reward_fn):
        '''
        self.map_length:
            x                   fixed map size (x, x)
            [x1, x2,...xn]      randomly choose one from x1 to xn as map side length every time reset environment

        self.map:
            0 = empty
            1 = obstacle

        self.num_agents:
            x                   fixed number of agents x
            (x, y)              randomly choose one from range x to y every time reset environment
            [x1, x2, ... xn]    randomly choose one from x1 to xn every time reset environment
        '''
        self.adaptive = adaptive
        if adaptive:
            self.level = config.env_level
            rand = random.randint(0, self.level)
            self.num_agents = rand//3+1
            self.map_size = (10+rand%3*30, 10+rand%3*30)
        else:
            self.num_agents = num_agents
            self.map_size = map_size


        # set as same as in PRIMAL
        self.obstacle_density = np.random.triangular(0, 0.33, 0.5)

        self.map = np.random.choice(2, self.map_size, p=[1-self.obstacle_density, self.obstacle_density]).astype(np.int)
        
        partition_list = map_partition(self.map)
        partition_list = [ partition for partition in partition_list if len(partition) >= 2 ]

        while len(partition_list) == 0:
            self.map = np.random.choice(2, self.map_size, p=[1-self.obstacle_density, self.obstacle_density]).astype(np.int)
            partition_list = map_partition(self.map)
            partition_list = [ partition for partition in partition_list if len(partition) >= 2 ]
        
        
        self.agents_pos = np.empty((self.num_agents, 2), dtype=np.int)
        self.goals_pos = np.empty((self.num_agents, 2), dtype=np.int)

        pos_num = sum([ len(partition) for partition in partition_list ])
        
        # loop to assign agent position and goal position for each agent
        for i in range(self.num_agents):

            pos_idx = random.randint(0, pos_num-1)
            partition_idx = 0
            for partition in partition_list:
                if pos_idx >= len(partition):
                    pos_idx -= len(partition)
                    partition_idx += 1
                else:
                    break 

            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.agents_pos[i] = np.asarray(pos, dtype=np.int)

            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.goals_pos[i] = np.asarray(pos, dtype=np.int)

            partition_list = [ partition for partition in partition_list if len(partition) >= 2 ]
            pos_num = sum([ len(partition) for partition in partition_list ])

        self.obs_radius = obs_radius

        self.reward_fn = reward_fn

        self.steps = 0

        # for save render images
        # self.fig = plt.figure()
        # self.imgs = []

        # self.history = [np.copy(self.agents_pos)]

    def reset(self, level=None):

        if self.adaptive:
            self.level = level
            rand = random.randint(0, self.level)
            self.num_agents = rand//3+1
            if self.num_agents > 16:
                self.map_size = (70, 70)
            elif self.num_agents > 4:
                self.map_size = (40+rand%2*30, 40+rand%2*30)
            else:
                self.map_size = (10+rand%3*30, 10+rand%3*30)

        self.obstacle_density = np.random.triangular(0, 0.33, 0.5)
        self.map = np.random.choice(2, self.map_size, p=[1-self.obstacle_density, self.obstacle_density]).astype(np.float32)
        
        partition_list = map_partition(self.map)
        partition_list = [ partition for partition in partition_list if len(partition) >= 2 ]

        while len(partition_list) == 0:
            self.map = np.random.choice(2, self.map_size, p=[1-self.obstacle_density, self.obstacle_density]).astype(np.float32)
            partition_list = map_partition(self.map)
            partition_list = [ partition for partition in partition_list if len(partition) >= 2 ]
        
        
        self.agents_pos = np.empty((self.num_agents, 2), dtype=np.int)
        self.goals_pos = np.empty((self.num_agents, 2), dtype=np.int)

        pos_num = sum([ len(partition) for partition in partition_list ])
        
        for i in range(self.num_agents):

            pos_idx = random.randint(0, pos_num-1)
            partition_idx = 0
            for partition in partition_list:
                if pos_idx >= len(partition):
                    pos_idx -= len(partition)
                    partition_idx += 1
                else:
                    break 

            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.agents_pos[i] = np.asarray(pos, dtype=np.int)

            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.goals_pos[i] = np.asarray(pos, dtype=np.int)

            partition_list = [ partition for partition in partition_list if len(partition) >= 2 ]
            pos_num = sum([ len(partition) for partition in partition_list ])

        self.steps = 0

        # self.fig = plt.figure()
        # self.imgs = []

        # self.history = [np.copy(self.agents_pos)]

        return self.observe()

    def load(self, map:np.ndarray, agents_pos:np.ndarray, goals_pos:np.ndarray):
        ''' load map, use for testing'''

        self.map = np.copy(map)
        self.agents_pos = np.copy(agents_pos)
        self.goals_pos = np.copy(goals_pos)

        self.num_agents = agents_pos.shape[0]

        # self.history = [np.copy(self.agents_pos)]
        
        self.steps = 0

        # self.fig = plt.figure()
        self.imgs = []

    def step(self, actions: List[int]):
        '''
        actions:
            list of indices
                0 stay
                1 up
                2 down
                3 left
                4 right
        '''

        assert len(actions) == self.num_agents, 'actions number' + str(actions)
        assert all([action_idx<5 and action_idx>=0 for action_idx in actions]), 'action index out of range'

        checking_list = [i for i in range(self.num_agents)]

        rewards = []
        next_pos = np.copy(self.agents_pos)

        # remove unmoving agent id
        for agent_id in checking_list.copy():
            if actions[agent_id] == 0:
                # unmoving

                if np.array_equal(self.agents_pos[agent_id], self.goals_pos[agent_id]):
                    rewards.append(self.reward_fn['stay_on_goal'])
                else:
                    rewards.append(self.reward_fn['stay_off_goal'])

                checking_list.remove(agent_id)
            else:
                # move
                next_pos[agent_id] += action_list[actions[agent_id]]
                rewards.append(self.reward_fn['move'])

        # assert len(rewards)==len(actions), '{}, {}'.format(len(rewards), len(actions))

        # for agent_id in checking_list:

        #     next_pos[agent_id] += action_list[actions[agent_id]]

        # first round check, these two conflicts have the heightest priority
        for agent_id in checking_list.copy():

            if np.any(next_pos[agent_id]<0) or np.any(next_pos[agent_id]>=self.map_size[0]):
                # agent out of map range
                rewards[agent_id] = self.reward_fn['collision']
                next_pos[agent_id] = self.agents_pos[agent_id]
                checking_list.remove(agent_id)

            elif self.map[tuple(next_pos[agent_id])] == 1:
                # collide obstacle
                rewards[agent_id] = self.reward_fn['collision']
                next_pos[agent_id] = self.agents_pos[agent_id]
                checking_list.remove(agent_id)

        # second round check, agent swapping conflict
        all_good = False
        while not all_good:

            all_good = True
            for agent_id in checking_list:

                target_agent_id = np.where(np.all(next_pos[agent_id]==self.agents_pos, axis=1))[0]

                if target_agent_id:

                    target_agent_id = target_agent_id.item()
                    if target_agent_id == agent_id:
                        print(agent_id)
                        print(target_agent_id)
                        print(actions)
                        print(checking_list)
                        raise RuntimeError('id check')

                    if np.array_equal(next_pos[target_agent_id], self.agents_pos[agent_id]):
                        assert target_agent_id in checking_list, 'not in check'

                        next_pos[agent_id] = self.agents_pos[agent_id]
                        rewards[agent_id] = self.reward_fn['collision']

                        next_pos[target_agent_id] = self.agents_pos[target_agent_id]
                        rewards[target_agent_id] = self.reward_fn['collision']

                        checking_list.remove(agent_id)
                        checking_list.remove(target_agent_id)
                        all_good = False
                        break


        # third round check, agent collision conflict
        all_good = False
        while not all_good:
            
            all_good = True
            for agent_id in checking_list:

                collide_agent_id = np.where(np.all(next_pos==next_pos[agent_id], axis=1))[0].tolist()
                if len(collide_agent_id) > 1:
                    # collide agent
                    
                    # if all agents in collide agent are in checking list
                    all_in_checking = True
                    for id in collide_agent_id.copy():
                        if id not in checking_list:
                            all_in_checking = False
                            collide_agent_id.remove(id)


                    if all_in_checking:

                        collide_agent_pos = next_pos[collide_agent_id].tolist()
                        for pos, id in zip(collide_agent_pos, collide_agent_id):
                            pos.append(id)
                        collide_agent_pos.sort(key=lambda x: x[0]*self.map_size[0]+x[1])

                        collide_agent_id.remove(collide_agent_pos[0][2])

                        # checking_list.remove(collide_agent_pos[0][2])

                    next_pos[collide_agent_id] = self.agents_pos[collide_agent_id]
                    for id in collide_agent_id:
                        rewards[id] = self.reward_fn['collision']

                    for id in collide_agent_id:
                        checking_list.remove(id)

                    all_good = False
                    break


        # self.history.append(np.copy(next_pos))
        self.agents_pos = np.copy(next_pos)

        self.steps += 1

        # check done
        if np.array_equal(self.agents_pos, self.goals_pos):
            done = True
            rewards = [ self.reward_fn['finish'] for _ in range(self.num_agents) ]
        else:
            done = False

        info = {'step': self.steps-1}

        # make sure no overlapping agents
        if np.unique(self.agents_pos, axis=0).shape[0] < self.num_agents:
            print(self.steps)
            print(self.map)
            print(self.agents_pos)
            raise RuntimeError('unique')

        return self.observe(), rewards, done, info


    def observe(self):
        '''
        return observation and position for each agent

        obs: tensor consists of two layers, (2, 2*self.obs_radius+1, 2*self.obs_radius+1)
            layer 1: obstacle, note 0 represents obstacle because we are using 0 padding in CNN 
            layer 2: other agents
        
        pos: vector of length 4, current agent position and goal position

        '''
        obs = np.zeros((self.num_agents, 2, 2*self.obs_radius+1, 2*self.obs_radius+1), dtype=np.bool)
        pos = np.zeros((self.num_agents, 4), dtype=np.uint8)

        pos[:, 0:2] = self.agents_pos
        pos[:, 2:4] = self.goals_pos

        # 0 represents obstacle to match 0 padding in CNN 
        obstacle_map = np.pad(self.map==0, self.obs_radius, 'constant', constant_values=0)

        agent_map = np.zeros((self.map_size), dtype=np.bool)
        agent_map[self.agents_pos[:,0], self.agents_pos[:,1]] = 1
        agent_map = np.pad(agent_map, self.obs_radius, 'constant', constant_values=0)

        # goal_map = np.zeros(self.map_size, dtype=np.float32)
        # goal_map[self.goals_pos[:,0], self.goals_pos[:,1]] = 1

        for i, agent_pos in enumerate(self.agents_pos):
            x, y = agent_pos

            obs[i,0] = obstacle_map[x:x+2*self.obs_radius+1, y:y+2*self.obs_radius+1]

            obs[i,1] = agent_map[x:x+2*self.obs_radius+1, y:y+2*self.obs_radius+1]

            obs[i,1,self.obs_radius,self.obs_radius] = 0

        return obs, pos
    
    def render(self):
        if not hasattr(self, 'fig'):
            self.fig = plt.figure()

        map = np.copy(self.map)
        for agent_id in range(self.num_agents):
            if np.array_equal(self.agents_pos[agent_id], self.goals_pos[agent_id]):
                map[tuple(self.agents_pos[agent_id])] = 4
            else:
                map[tuple(self.agents_pos[agent_id])] = 2
                map[tuple(self.goals_pos[agent_id])] = 3

        map = map.astype(np.uint8)
        # plt.xlabel('step: {}'.format(self.steps))

        # add text in plot
        self.imgs.append([])
        if hasattr(self, 'texts'):
            for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(zip(self.agents_pos, self.goals_pos)):
                self.texts[i].set_position((agent_y, agent_x))
                self.texts[i].set_text(i)
        else:
            self.texts = []
            for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(zip(self.agents_pos, self.goals_pos)):
                text = plt.text(agent_y, agent_x, i, color='black', ha='center', va='center')
                plt.text(goal_y, goal_x, i, color='black', ha='center', va='center')
                self.texts.append(text)


        plt.imshow(color_map[map], animated=True)


        plt.show()
        # plt.ion()
        plt.pause(0.5)

    def close(self, save=False):

        plt.close()
        del self.fig
