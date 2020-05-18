import numpy as np
import torch
from environment import Environment
from model_dqn import Network
from search import find_path
import pickle
import os
import matplotlib as mpl
mpl.use('TkAgg') 
import matplotlib.pyplot as plt
import random
import argparse
from typing import Union
import config
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
test_num = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_test(num_agents:Union[int,list,tuple]):

    name = './test{}.pkl'.format(num_agents) if num_agents != None else './test.pkl'

    tests = {'maps': [], 'agents': [], 'goals': [], 'opt_steps': []}

    env = Environment(num_agents=num_agents)

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

        env.reset()

    with open(name, 'wb') as f:
        pickle.dump(tests, f)


def test_model(num_agents, test_case='test2.pkl'):


    network = Network()
    network.eval()
    network.to(device)
    
    vrange = torch.linspace(-5, 5, 51).to(device)


    with open(test_case, 'rb') as f:
        tests = pickle.load(f)

    model_name = config.save_interval
    while os.path.exists('./models/{}.pth'.format(model_name)):
        state_dict = torch.load('./models/{}.pth'.format(model_name))
        network.load_state_dict(state_dict)
        env = Environment()

        case = 1
        show = False
        show_steps = 30
        fail = 0
        optimal = 0

        for i in range(test_num):
            env.load(tests['maps'][i], tests['agents'][i], tests['goals'][i])
            
            done = False
            network.reset()

            while not done and env.steps < config.max_steps:
                if i == case and show and env.steps < show_steps:
                    env.render()

                obs_pos = env.observe()
                # obs = np.expand_dims(obs, axis=0)

                with torch.no_grad():

                    q_vals = network.step(torch.FloatTensor(obs_pos[0]).to(device), torch.FloatTensor(obs_pos[1]).to(device))
                    if network.distributional:
                        q_vals = (q_vals.exp() * vrange).sum(2)

                if i == case and show and env.steps < show_steps:
                    print(q_vals)

                action = torch.argmax(q_vals, 1).tolist()

                if i == case and show and env.steps < show_steps:
                    print(action)


                _, _, done, _ = env.step(action)
                # print(done)



            if not np.array_equal(env.agents_pos, env.goals_pos):
                fail += 1
                if show:
                    print(i)

            if env.steps == tests['opt_steps'][i]:
                optimal += 1

            if i == case and show:
                env.close()
        
        f_rate = (test_num-fail)/test_num
        o_rate = optimal/test_num

        print('--------------{}---------------'.format(model_name))
        print('finish: %.4f' %f_rate)
        print('optimal: %.4f' %o_rate)

        model_name += config.save_interval

        # if test_case != 'test.pkl':
        #     finish_rate.append(f_rate)
        #     optimal_rate.append(o_rate)

    # plt.xlabel('number of agents')
    # plt.ylabel('percentage')

    # plt.plot(x, finish_rate, label='finish_rate')
    # plt.plot(x, optimal_rate, label='optimal_rate')
    # plt.xticks(range(2,6))
    
    # plt.legend()
    # plt.show()

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test MAPF model')

    parser.add_argument('--mode', type=str, choices=['test', 'create'], default='test', help='create test set or run test set')
    parser.add_argument('--number', type=int, default=2, help='number of agents in environment')

    args = parser.parse_args()

    if args.mode == 'test':
        test_model(args.number)
    elif args.mode == 'create':
        create_test(args.number)
    