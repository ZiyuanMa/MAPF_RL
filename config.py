############################################################
####################    environment     ####################
############################################################
map_length = 20
num_agents = 4
obs_radius = 4
reward_fn = dict(move=-0.075,
                move_to_goal=0,
                stay_on_goal=0,
                stay_off_goal=-0.125,
                collision=-0.5,
                finish=3)

obs_shape = (2,9,9)
pos_shape = (4,)


############################################################
####################         DQN        ####################
############################################################

# basic training setting
save_interval=2000
gamma=0.99
batch_size=32
learning_starts=10000
target_network_update_freq=2000
save_path='./models'
max_steps = 256
bt_steps = 48
load_model = None

local_buffer_size = max_steps
global_buffer_size = 2048*local_buffer_size

# gradient norm clipping
grad_norm_dqn=40

# distributional dqn
distributional = False

# prioritized replay
prioritized_replay_alpha=0.6
prioritized_replay_beta=0.4

# use double q learning
double_q = False

# imitation learning
imitation_ratio = 0.2

# dqn network setting
cnn_channel = 96
obs_dim = 2
obs_latent_dim = 496
pos_dim = 4
pos_latent_dim = 16

# communication
max_comm_agents = 3 # including agent itself, means one can at most communicate with (max_comm_agents-1) agents
