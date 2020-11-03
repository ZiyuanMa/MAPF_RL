import numpy as np
import torch
from environment import Environment
from model import Network
from search import find_path
import pickle
import os
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import argparse
from typing import Union
import config
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
test_num = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def create_test(agent_range:Union[int,list,tuple], map_range:Union[int,list,tuple]):

    name = './test{}_{}.pkl'.format(agent_range, map_range)

    tests = {'maps': [], 'agents': [], 'goals': [], 'opt_steps': []}

    if type(agent_range) is int:
        num_agents = agent_range
    elif type(agent_range) is list:
        num_agents = random.choice(agent_range)
    else:
        num_agents = random.randint(agent_range[0], agent_range[1])

    if type(map_range) is int:
        map_length = map_range
    elif type(map_range) is list:
        map_length = random.choice(map_range)
    else:
        map_length = random.randint(map_range[0]//5, map_range[1]//5) * 5

    env = Environment(num_agents=num_agents, map_length=map_length)

    for _ in range(test_num):
        tests['maps'].append(np.copy(env.map))
        tests['agents'].append(np.copy(env.agents_pos))
        tests['goals'].append(np.copy(env.goals_pos))

        actions = find_path(env)
        while actions is None:
            env.reset()
            tests['maps'][-1] = np.copy(env.map)
            tests['agents'][-1] = np.copy(env.agents_pos)
            tests['goals'][-1] = np.copy(env.goals_pos)
            actions = find_path(env)

        tests['opt_steps'].append(len(actions))

        if type(agent_range) is int:
            num_agents = agent_range
        elif type(agent_range) is list:
            num_agents = random.choice(agent_range)
        else:
            num_agents = random.randint(agent_range[0], agent_range[1])

        if type(map_range) is int:
            map_length = map_range
        elif type(map_range) is list:
            map_length = random.choice(map_range)
        else:
            map_length = random.randint(map_range[0]//5, map_range[1]//5) * 5

        env.reset(num_agents=num_agents, map_length=map_length)

    tests['opt_mean_steps'] = sum(tests['opt_steps']) / len(tests['opt_steps'])

    with open(name, 'wb') as f:
        pickle.dump(tests, f)


def test_model(test_case='test10_10_0.3.pkl'):

    network = Network()
    network.eval()
    network.to(device)


    with open(test_case, 'rb') as f:
        tests = pickle.load(f)

    model_names = [337500]

    for model_name in model_names:
        state_dict = torch.load('./models/{}.pth'.format(model_name), map_location=device)
        network.load_state_dict(state_dict)
        env = Environment()

        case = 4
        show = False
        show_steps = 30

        fail = 0
        steps = []

        for i in range(test_num):
            env.load(tests['maps'][i], tests['agents'][i], tests['goals'][i])
            
            done = False
            network.reset()

            while not done and env.steps < config.max_steps:
                if i == case and show and env.steps < show_steps:
                    env.render()

                obs_pos = env.observe()

                actions, q_vals, _, _ = network.step(torch.FloatTensor(obs_pos[0]).to(device), torch.FloatTensor(obs_pos[1]).to(device))

                if i == case and show and env.steps < show_steps:
                    print(q_vals)

                if i == case and show and env.steps < show_steps:
                    print(actions)

                _, _, done, _ = env.step(actions)
                # print(done)

            steps.append(env.steps)

            if not np.array_equal(env.agents_pos, env.goals_pos):
                fail += 1
                if show:
                    print(i)

            if i == case and show:
                env.close(True)
        
        f_rate = (test_num-fail)/test_num
        mean_steps = sum(steps)/test_num

        print('--------------{}---------------'.format(model_name))
        print('success rate: %.4f' %f_rate)
        print('average steps: %.2f' %mean_steps)



def make_animation():
    color_map = np.array([[255, 255, 255],   # white
                    [190, 190, 190],   # gray
                    [0, 191, 255],   # blue
                    [255, 165, 0],   # orange
                    [0, 250, 154]])  # green

    test_name = 'test4.pkl'
    with open(test_name, 'rb') as f:
        tests = pickle.load(f)
    test_case = 1
    
    model_name = config.save_interval * 40
    steps = 30
    network = Network()
    network.eval()
    network.to(device)
    state_dict = torch.load('./models/{}.pth'.format(model_name), map_location=device)
    network.load_state_dict(state_dict)

    env = Environment()
    env.load(tests['maps'][test_case], tests['agents'][test_case], tests['goals'][test_case])

    fig = plt.figure()
            
    done = False
    obs_pos = env.observe()

    imgs = []
    while not done and env.steps < steps:
        imgs.append([])
        map = np.copy(env.map)
        for agent_id in range(env.num_agents):
            if np.array_equal(env.agents_pos[agent_id], env.goals_pos[agent_id]):
                map[tuple(env.agents_pos[agent_id])] = 4
            else:
                map[tuple(env.agents_pos[agent_id])] = 2
                map[tuple(env.goals_pos[agent_id])] = 3
        map = map.astype(np.uint8)

        img = plt.imshow(color_map[map], animated=True)

        imgs[-1].append(img)

        for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(zip(env.agents_pos, env.goals_pos)):
            text = plt.text(agent_y, agent_x, i, color='black', ha='center', va='center')
            imgs[-1].append(text)
            text = plt.text(goal_y, goal_x, i, color='black', ha='center', va='center')
            imgs[-1].append(text)


        actions, _ = network.step(torch.from_numpy(obs_pos[0].astype(np.float32)).to(device), torch.from_numpy(obs_pos[1].astype(np.float32)).to(device))
        obs_pos, _, done, _ = env.step(actions)
        # print(done)

    ani = animation.ArtistAnimation(fig, imgs, interval=500, blit=True,
                                repeat_delay=1000)

    ani.save('dynamic_images.mp4')

    

if __name__ == '__main__':

    # create_test(8, 20)
    test_model()
    # make_animation()
    