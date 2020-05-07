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
training_timesteps = 2000000
save_interval=50000
gamma=0.99
batch_size_dqn=32
train_freq=4
learning_starts=20000
target_network_update_freq=4000
buffer_size=50000
save_path='./models'
max_steps = 200

# gradient norm clipping
grad_norm_dqn=10

# epsilon greedy
explore_start_eps = 1.0
explore_final_eps = 0.1

# prioritized replay
prioritized_replay_alpha=0.6
prioritized_replay_beta=0.4

# imitation learning
imitation_ratio = 0.6

# dqn network setting
cnn_channel = 64
obs_dim = 2
obs_latent_dim = 240
pos_dim = 4
pos_latent_dim = 16
