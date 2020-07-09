import torch
import torch.nn as nn
import torch.nn.functional as F
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
                obs_dim=config.obs_shape[0], obs_latent_dim=config.obs_latent_dim,
                pos_dim=config.pos_shape[0], pos_latent_dim=config.pos_latent_dim,
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

            ResBlock(cnn_channel, type='cnn'),

            ResBlock(cnn_channel, type='cnn'),

            nn.Conv2d(cnn_channel, 4, 1, 1),
            nn.ReLU(True),

            nn.Flatten(),

            nn.Linear(4*9*9, obs_latent_dim),
            nn.ReLU(True),
        )

        self.pos_encoder = nn.Sequential(
            nn.Linear(pos_dim, pos_latent_dim),
            nn.ReLU(True),
            ResBlock(pos_latent_dim),
        )

        self.act_encoder = nn.Sequential(
            nn.Linear(5, config.act_latent_dim),
            nn.ReLU(True),
            ResBlock(config.act_latent_dim),
        )

        self.concat_encoder = nn.Sequential(
            ResBlock(config.latent_dim),
            ResBlock(config.latent_dim),
        )

        self.recurrent = nn.GRU(config.obs_latent_dim+config.act_latent_dim, config.obs_latent_dim+config.act_latent_dim, batch_first=True)

        # dueling q structure
        if distributional:
            self.adv = nn.Linear(config.latent_dim, 5*self.num_quant)
            self.state = nn.Linear(config.latent_dim, 1*self.num_quant)
        else:
            self.adv = nn.Linear(config.latent_dim, 5)
            self.state = nn.Linear(config.latent_dim, 1)

        self.hidden = None

        for _, m in self.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def step(self, obs, pos, act):
        # print(obs.shape)
        obs_latent = self.obs_encoder(obs)
        pos_latent = self.pos_encoder(pos)
        act_latent = self.act_encoder(act)

        latent = torch.cat((obs_latent, act_latent), dim=1)
        latent = latent.unsqueeze(1)
        self.recurrent.flatten_parameters()
        if self.hidden is None:
            _, self.hidden = self.recurrent(latent)
        else:
            _, self.hidden = self.recurrent(latent, self.hidden)
        hidden = torch.squeeze(self.hidden, dim=0)

        hidden = torch.cat((hidden, pos_latent), dim=1)

        hidden = self.concat_encoder(hidden)
        adv_val = self.adv(hidden)
        state_val = self.state(hidden)

        if self.distributional:
            adv_val = adv_val.view(-1, 5, self.num_quant)
            state_val = state_val.unsqueeze(1)
            # batch_size x 5 x 200
            q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)

            actions = q_val.mean(2).argmax(1).tolist()

        else:
            q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)
            # print(q_val.shape)
            actions = torch.argmax(q_val, 1).tolist()

        return actions, q_val.numpy(), self.hidden[0].numpy()

    def reset(self):
        self.hidden = None

    def bootstrap(self, obs, pos, act, steps, hidden):
        batch_size = obs.size(0)
        step = obs.size(1)
        hidden = hidden.unsqueeze(0)

        obs = obs.contiguous().view(-1, self.obs_dim, 9, 9)
        pos = pos.contiguous().view(-1, self.pos_dim)
        act = act.contiguous().view(-1, 5)

        obs_latent = self.obs_encoder(obs)
        pos_latent = self.pos_encoder(pos)
        act_latent = self.act_encoder(act)

        obs_act_latent = torch.cat((obs_latent, act_latent), dim=1)

        # latent = latent.split(steps)
        # latent = pad_sequence(latent, batch_first=True)

        obs_act_latent = obs_act_latent.view(batch_size, step, config.obs_latent_dim+config.act_latent_dim)

        obs_act_latent = pack_padded_sequence(obs_act_latent, steps, batch_first=True, enforce_sorted=False)

        self.recurrent.flatten_parameters()
        _, hidden = self.recurrent(obs_act_latent, hidden)

        hidden = torch.squeeze(hidden)

        hidden = torch.cat((hidden, pos_latent), dim=1)

        hidden = self.concat_encoder(hidden)
        
        adv_val = self.adv(hidden)
        state_val = self.state(hidden)

        if self.distributional:
            adv_val = adv_val.view(-1, 5, self.num_quant)
            state_val = state_val.unsqueeze(1)
            q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)

        else:
            q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)

        return q_val