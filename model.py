import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import log_softmax
from torch.nn.utils.rnn import pack_padded_sequence
import config

class ResBlock(nn.Module):
    def __init__(self, channel, type='linear', bn=False):
        super().__init__()
        if type == 'cnn':
            if bn:
                self.block1 = nn.Sequential(
                    nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(channel)
                )
                self.block2 = nn.Sequential(
                    nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(channel)
                )
            else:
                self.block1 = nn.Conv2d(channel, channel, 3, 1, 1)
                self.block2 = nn.Conv2d(channel, channel, 3, 1, 1)

        elif type == 'linear':
            self.block1 = nn.Linear(channel, channel)
            self.block2 = nn.Linear(channel, channel)
        else:
            raise RuntimeError('type does not support')

    def forward(self, x):
        identity = x

        x = self.block1(x)
        x = F.relu(x)

        x = self.block2(x)

        x += identity

        x = F.relu(x)

        return x

class Network(nn.Module):
    def __init__(self, cnn_channel=config.cnn_channel,
                obs_dim=config.obs_dim, obs_latent_dim=config.obs_latent_dim,
                pos_dim=config.pos_dim, pos_latent_dim=config.pos_latent_dim,
                distributional=config.distributional):

        super().__init__()

        self.obs_dim = obs_dim
        self.pos_dim = pos_dim
        self.latent_dim = obs_latent_dim + pos_latent_dim
        self.distributional = distributional
        self.num_quant = 200

        self.obs_encoder = nn.Sequential(
            nn.Conv2d(obs_dim, cnn_channel, 3, 1, 1),
            nn.ReLU(True),

            ResBlock(cnn_channel, type='cnn'),

            nn.Conv2d(cnn_channel, cnn_channel, 3, 1, 1),
            nn.ReLU(True),

            ResBlock(cnn_channel, type='cnn'),

            nn.Conv2d(cnn_channel, 8, 1, 1),
            nn.ReLU(True),

            nn.Flatten(),

            nn.Linear(8*9*9, obs_latent_dim),
            nn.ReLU(True),
        )

        self.pos_encoder = nn.Sequential(
            nn.Linear(pos_dim, pos_latent_dim),
            nn.ReLU(True),
            nn.Linear(pos_latent_dim, pos_latent_dim),
            nn.ReLU(True),
        )

        self.concat_encoder = nn.Sequential(
            ResBlock(self.latent_dim), 
            # ResBlock(self.latent_dim),
        )

        self.recurrent = nn.GRU(self.latent_dim, self.latent_dim, batch_first=True)

        # dueling q structure
        if distributional:
            self.adv = nn.Linear(self.latent_dim, 5*self.num_quant)
            self.state = nn.Linear(self.latent_dim, 1*self.num_quant)
        else:
            self.adv = nn.Linear(self.latent_dim, 5)
            self.state = nn.Linear(self.latent_dim, 1)

        self.hidden = None

        for _, m in self.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def step(self, obs, pos):
        
        obs_latent = self.obs_encoder(obs)
        pos_latent = self.pos_encoder(pos)
        concat_latent = torch.cat((obs_latent, pos_latent), dim=1)
        latent = self.concat_encoder(concat_latent)
        latent = latent.unsqueeze(1)
        if self.hidden is None:
            _, self.hidden = self.recurrent(latent)
        else:
            _, self.hidden = self.recurrent(latent, self.hidden)
        self.hidden = torch.squeeze(self.hidden)

        adv_val = self.adv(self.hidden)
        state_val = self.state(self.hidden)

        if self.distributional:
            adv_val = adv_val.view(-1, 5, self.num_quant)
            state_val = state_val.unsqueeze(1)
            q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)

        else:
            q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)

        self.hidden = self.hidden.unsqueeze(0)

        return q_val

    def reset(self):
        self.hidden = None

    def bootstrap(self, obs, pos, steps):
        batch_size = obs.size(0)

        obs = obs.view(-1, self.obs_dim, 9, 9)
        pos = pos.view(-1, self.pos_dim)


        obs_latent = self.obs_encoder(obs)
        pos_latent = self.pos_encoder(pos)

        concat_latent = torch.cat((obs_latent, pos_latent), dim=1)
        latent = self.concat_encoder(concat_latent)

        # latent = latent.split(steps)
        # latent = pad_sequence(latent, batch_first=True)

        latent = latent.view(batch_size, config.bt_steps, self.latent_dim)

        latent = pack_padded_sequence(latent, steps, batch_first=True, enforce_sorted=False)

        self.recurrent.flatten_parameters()
        _, hidden = self.recurrent(latent)

        hidden = torch.squeeze(hidden)
        
        adv_val = self.adv(hidden)
        state_val = self.state(hidden)

        if self.distributional:
            adv_val = adv_val.view(-1, 5, self.N)
            state_val = state_val.unsqueeze(1)
            q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)

        else:
            q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)

        return q_val