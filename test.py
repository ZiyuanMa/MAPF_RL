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
import config
import time
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
test_num = 200
device = torch.device('cpu')
device = torch.device('cuda:0')

# def create_test(num_agents:int, map_length:int, density=None):

#     name = './test{}_{}_{}.pkl'.format(num_agents, map_length, density)

#     tests = {'maps': [], 'agents': [], 'goals': []}


#     env = Environment(fix_density=density, num_agents=num_agents, map_length=map_length)

#     for _ in tqdm(range(test_num)):
#         tests['maps'].append(np.copy(env.map))
#         tests['agents'].append(np.copy(env.agents_pos))
#         tests['goals'].append(np.copy(env.goals_pos))

#         env.reset(num_agents=num_agents, map_length=map_length)

#     with open(name, 'wb') as f:
#         pickle.dump(tests, f)


# def test_model(test_case='test32_40_0.3.pkl'):

#     network = Network()
#     network.eval()
#     network.to(device)


#     with open(test_case, 'rb') as f:
#         tests = pickle.load(f)

#     model_name = 502500
#     while os.path.exists('./models/{}.pth'.format(model_name)):
#         state_dict = torch.load('./models/{}.pth'.format(model_name), map_location=device)
#         network.load_state_dict(state_dict)
#         env = Environment()

#         case = 30
#         show = False
#         show_steps = 100

#         fail = 0
#         steps = []

#         start = time.time()
#         for i in range(test_num):
#             env.load(tests['maps'][i], tests['agents'][i], tests['goals'][i])
            
#             done = False
#             network.reset()

#             while not done and env.steps < config.max_steps:
#                 if i == case and show and env.steps < show_steps:
#                     env.render()

#                 obs_pos = env.observe()

#                 actions, q_vals, _ = network.step(torch.FloatTensor(obs_pos[0]).to(device), torch.FloatTensor(obs_pos[1]).to(device))

#                 if i == case and show and env.steps < show_steps:
#                     print(q_vals)

#                 if i == case and show and env.steps < show_steps:
#                     print(actions)

#                 _, _, done, _ = env.step(actions)
#                 # print(done)
                    
#             steps.append(env.steps)

#             if not np.array_equal(env.agents_pos, env.goals_pos):
#                 fail += 1
#                 if show:
#                     print(i)



#             if i == case and show:
#                 env.close(True)

#         duration = time.time()-start
#         f_rate = (test_num-fail)/test_num
#         mean_steps = sum(steps)/test_num

#         print('--------------{}---------------'.format(model_name))
#         print('finish: %.4f' %f_rate)
#         print('mean steps: %.2f' %mean_steps)
#         print('time spend: %.2f' %duration)


#         model_name -= config.save_interval

def test_model(map_length, density):

    network = Network()
    network.eval()
    network.to(device)

    state_dict = torch.load('./model.pth', map_location=device)
    network.load_state_dict(state_dict)

    num_agents = 2

    while os.path.exists('./test{}_{}_{}.pkl'.format(num_agents, map_length, density)):

        with open('./test{}_{}_{}.pkl'.format(num_agents, map_length, density), 'rb') as f:
            tests = pickle.load(f)

        env = Environment()

        case = 28
        show = True
        show_steps = 100

        fail = 0
        steps = []

        start = time.time()
        for i in range(test_num):
            env.load(tests['maps'][i], tests['agents'][i], tests['goals'][i])
            
            done = False
            network.reset()

            while not done and env.steps < config.max_steps:
                if i == case and show and env.steps < show_steps:
                    env.render()

                obs_pos = env.observe()

                actions, q_vals, _ = network.step(torch.FloatTensor(obs_pos[0]).to(device), torch.FloatTensor(obs_pos[1]).to(device))

                if i == case and show and env.steps < show_steps:
                    print(q_vals)
                    print(actions)


                _, _, done, _ = env.step(actions)
                # print(done)

            steps.append(env.steps)

            if not np.array_equal(env.agents_pos, env.goals_pos):
                fail += 1
                # if show:
                print(i)

            if i == case and show:
                env.close(True)

        f_rate = (test_num-fail)/test_num
        mean_steps = sum(steps)/test_num
        duration = time.time()-start

        print('-----number of agents:{} map_length:{} density:{}-----'.format(num_agents, map_length, density))
        print('success rate: %.4f' %f_rate)
        print('average steps: %.2f' %mean_steps)
        print('time spend: %.2f' %duration)

        num_agents *= 2

def make_animation():
    color_map = np.array([[255, 255, 255],   # white
                    [190, 190, 190],   # gray
                    [0, 191, 255],   # blue
                    [255, 165, 0],   # orange
                    [0, 250, 154]])  # green

    test_name = 'test2_15_0.3.pkl'
    with open(test_name, 'rb') as f:
        tests = pickle.load(f)
    test_case = 16
    

    steps = 25
    network = Network()
    network.eval()
    network.to(device)
    state_dict = torch.load('./model.pth', map_location=device)
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


        actions, _, _ = network.step(torch.from_numpy(obs_pos[0].astype(np.float32)).to(device), torch.from_numpy(obs_pos[1].astype(np.float32)).to(device))
        obs_pos, _, done, _ = env.step(actions)
        # print(done)

    ani = animation.ArtistAnimation(fig, imgs, interval=600, blit=True,
                                repeat_delay=1000)

    ani.save('dynamic_images.mp4')

    

if __name__ == '__main__':

    # create_test(8, 20)
    # test_model(15, 0.3)
    make_animation()
    # create_test(1, 20)
