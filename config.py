############################################################
####################    environment     ####################
############################################################
map_length = 20
num_agents = 2
obs_radius = 4
reward_fn = dict(move=-0.075,
                stay_on_goal=0,
                stay_off_goal=-0.125,
                collision=-0.5,
                finish=3)


############################################################
####################         DQN        ####################
############################################################

# basic training setting
training_timesteps = 2400000
save_interval=50000
gamma=0.99
batch_size_dqn=32
train_freq=4
learning_starts=50000
target_network_update_freq=2500
buffer_size=65536
save_path='./models'
max_steps = 200
bt_steps = 32
dtype = 'float32'
load_model = './models/1850000.pth'

# gradient norm clipping
grad_norm_dqn=10

# epsilon greedy
explore_start_eps = 1.0
explore_final_eps = 0.05

# prioritized replay
prioritized_replay_alpha=0.6
prioritized_replay_beta=0.4

# use double q learning
double_q = False

# imitation learning
imitation_ratio = 0.4

# dqn network setting
cnn_channel = 64
obs_dim = 2
obs_latent_dim = 240
pos_dim = 4
pos_latent_dim = 16
