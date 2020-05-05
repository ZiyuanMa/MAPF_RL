import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors
import random
from typing import List


action_list = np.array([[0, 0],[-1, 0],[1, 0],[0, -1],[0, 1]], dtype=np.int)

color_map = np.array([[255, 255, 255],   # white
                    [190, 190, 190],   # gray
                    [0, 191, 255],   # blue
                    [255, 165, 0],   # orange
                    [0, 250, 154]])  # green


def map_partition(map):

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
            pos = open_list.pop(0)

            up = (pos[0]-1, pos[1])
            if up[0] >= 0 and map[up]==0 and up in empty_pos:
                empty_pos.remove(up)
                open_list.append(up)
            
            down = (pos[0]+1, pos[1])
            if down[0] < map.shape[0] and map[down]==0 and down in empty_pos:
                empty_pos.remove(down)
                open_list.append(down)
            
            left = (pos[0], pos[1]-1)
            if left[1] >= 0 and map[left]==0 and left in empty_pos:
                empty_pos.remove(left)
                open_list.append(left)
            
            right = (pos[0], pos[1]+1)
            if right[1] < map.shape[1] and map[right]==0 and right in empty_pos:
                empty_pos.remove(right)
                open_list.append(right)

            close_list.append(pos)


        partition_list.append(close_list)

    return partition_list
    


class Environment:
    def __init__(self, map_size=(20,20), num_agents=2, obs_radius=4):
        '''
        self.map:
            0 = empty
            1 = obstacle
        '''

        if num_agents == None:
            self.random_agents = True
            self.num_agents = random.randint(2, 4)
        else:
            self.random_agents = False
            self.num_agents = num_agents

        self.map_size = map_size
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

        self.steps = 0

        self.history = [np.copy(self.agents_pos)]

    def reset(self):
        if self.random_agents:
            self.num_agents = random.randint(2, 4)
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

        self.history = [np.copy(self.agents_pos)]

        return self.observe()

    def load(self, world: np.ndarray, agents_pos: np.ndarray, goals_pos: np.ndarray):

        self.map = np.copy(world)
        self.agents_pos = np.copy(agents_pos)
        self.goals_pos = np.copy(goals_pos)

        self.num_agents = agents_pos.shape[0]

        self.history = [np.copy(self.agents_pos)]
        
        self.steps = 0

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
        # assert all([action_idx<config.action_space and action_idx>=0 for action_idx in actions]), 'action index out of range'

        check_id = [i for i in range(self.num_agents)]

        rewards = []

        # remove no movement agent id
        for agent_id in check_id.copy():

            if actions[agent_id] == 0:
                # stay
                check_id.remove(agent_id)

                if np.array_equal(self.agents_pos[agent_id], self.goals_pos[agent_id]):
                    rewards.append(0)
                else:
                    rewards.append(-0.125)
            else:
                # move
                rewards.append(-0.075)


        next_pos = np.copy(self.agents_pos)

        for agent_id in check_id:

            next_pos[agent_id] += action_list[actions[agent_id]]


        for agent_id in check_id.copy():

            # move

            if np.any(next_pos[agent_id]<np.array([0,0])) or np.any(next_pos[agent_id]>=np.asarray(self.map_size)): 
                # agent out of bound
                rewards[agent_id] = -0.5
                next_pos[agent_id] = self.agents_pos[agent_id]
                check_id.remove(agent_id)

            elif self.map[tuple(next_pos[agent_id])] == 1:
                # collide obstacle
                rewards[agent_id] = -0.5
                next_pos[agent_id] = self.agents_pos[agent_id]
                check_id.remove(agent_id)

        # agent swap
        for agent_id in check_id:
            if np.any(np.all(next_pos[agent_id]==self.agents_pos, axis=1)):

                target_agent_id = np.where(np.all(next_pos[agent_id]==self.agents_pos, axis=1))[0].item()
                # assert len(target_agent_id) == 1, 'target > 1'

                if np.array_equal(next_pos[target_agent_id], self.agents_pos[agent_id]):
                    assert target_agent_id in check_id, 'not in check'

                    next_pos[agent_id] = self.agents_pos[agent_id]
                    rewards[agent_id] = -0.5

                    next_pos[target_agent_id] = self.agents_pos[target_agent_id]
                    rewards[target_agent_id] = -0.5

                    check_id.remove(agent_id)
                    check_id.remove(target_agent_id)

        flag = False
        while not flag:
            
            flag = True
            for agent_id in check_id.copy():
                
                if np.sum(np.all(next_pos==next_pos[agent_id], axis=1)) > 1:
                    # collide agent

                    collide_agent_id = np.where(np.all(next_pos==next_pos[agent_id], axis=1))[0].tolist()
                    all_in = True
                    for id in collide_agent_id:
                        if id not in check_id:
                            all_in =False
                            break

                    if not all_in:
                        # agent collide no movement agent
                        collide_agent_id = [ id for id in collide_agent_id if id in check_id]

                    else:

                        collide_agent_pos = next_pos[collide_agent_id].tolist()
                        for pos, id in zip(collide_agent_pos, collide_agent_id):
                            pos.append(id)
                        collide_agent_pos.sort(key=lambda x: x[0]*self.map_size[0]+x[1])

                        collide_agent_id.remove(collide_agent_pos[0][2])

                        # check_id.remove(collide_agent_pos[0][2])

                    next_pos[collide_agent_id] = self.agents_pos[collide_agent_id]
                    for id in collide_agent_id:
                        rewards[id] = -0.5

                    for id in collide_agent_id:
                        check_id.remove(id)

                    flag = False
                    break

                

        self.history.append(np.copy(next_pos))
        self.agents_pos = np.copy(next_pos)

        self.steps += 1

        # check done
        if np.array_equal(self.agents_pos, self.goals_pos):
            done = True
            rewards = [ config.finish_reward for _ in range(self.num_agents) ]
        else:
            done = False

        info = {'step': self.steps-1}

        if np.unique(self.agents_pos, axis=0).shape[0] < self.num_agents:
            # no overlapping agents
            print(self.steps)
            print(self.map)
            print(self.agents_pos)
            print(self.history[-1])
            raise RuntimeError('unique')

        return self.observe(), rewards, done, info


    def observe(self):
        obs = np.zeros((self.num_agents, 2, 2*self.obs_radius+1, 2*self.obs_radius+1), dtype=np.float32)
        pos = np.zeros((self.num_agents, 4), dtype=np.float32)

        obstacle_map = np.pad(self.map, 4, 'constant', constant_values=1)

        agent_map = np.zeros((self.map_size[0]+2*self.obs_radius, self.map_size[1]+2*self.obs_radius), dtype=np.float32)
        agent_map[self.agents_pos[:,0], self.agents_pos[:,1]] = 1

        # goal_map = np.zeros(self.map_size, dtype=np.float32)
        # goal_map[self.goals_pos[:,0], self.goals_pos[:,1]] = 1

        for i in range(self.num_agents):
            x, y = self.agents_pos[i]
            pos[i][0:2] = self.agents_pos[i]
            pos[i][2:4] = self.goals_pos[i]

            obs[i,0] = obstacle_map[x:x+2*self.obs_radius+1, y:y+2*self.obs_radius+1]==0

            obs[i,1] = agent_map[x:x+2*self.obs_radius+1, y:y+2*self.obs_radius+1]


        return obs, pos

    
    def render(self):
        map = np.copy(self.map)
        for agent_id in range(self.num_agents):
            if np.array_equal(self.agents_pos[agent_id], self.goals_pos[agent_id]):
                map[tuple(self.agents_pos[agent_id])] = 4
            else:
                map[tuple(self.agents_pos[agent_id])] = 2
                map[tuple(self.goals_pos[agent_id])] = 3

        map = map.astype(np.uint8)
        plt.imshow(color_map[map])
        plt.xlabel('step: {}'.format(self.steps))
        plt.ion()
        plt.show()
        plt.pause(0.5)

    def close(self):
        plt.close()
