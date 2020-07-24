import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import config

class ResBlock(nn.Module):
    def __init__(self, channel, a=3, b=1, c=1, type='linear', bn=False):
        super().__init__()
        if type == 'cnn':
            if bn:
                self.block1 = nn.Sequential(
                    nn.Conv2d(channel, channel, a, b, c, bias=False),
                    nn.BatchNorm2d(channel)
                )
                self.block2 = nn.Sequential(
                    nn.Conv2d(channel, channel, a, b, c, bias=False),
                    nn.BatchNorm2d(channel)
                )
            else:
                self.block1 = nn.Conv2d(channel, channel, a, b, c)
                self.block2 = nn.Conv2d(channel, channel, a, b, c)

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
                distributional=config.distributional):

        super().__init__()

        self.obs_dim = config.obs_shape[0]
        self.latent_dim = config.latent_dim
        self.distributional = distributional
        self.num_quant = 200
        self.output_shape = (64, 5, 5)

        self.obs_encoder = nn.Sequential(
            nn.Conv2d(self.obs_dim, 128, 3, 1),
            nn.ReLU(True),

            ResBlock(128, type='cnn'),

            ResBlock(128, type='cnn'),

            ResBlock(128, type='cnn'),

            nn.Conv2d(128, 16, 1, 1),
            nn.ReLU(True),

            nn.Flatten(),

        )

        self.recurrent = nn.GRU(16*7*7, self.latent_dim, batch_first=True)

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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def step(self, obs):
        # print(obs.shape)
        latent = self.obs_encoder(obs)
        latent = latent.unsqueeze(1)
        self.recurrent.flatten_parameters()
        if self.hidden is None:
            _, self.hidden = self.recurrent(latent)
        else:
            _, self.hidden = self.recurrent(latent, self.hidden)
        self.hidden = torch.squeeze(self.hidden, dim=0)

        adv_val = self.adv(self.hidden)
        state_val = self.state(self.hidden)

        if self.distributional:
            adv_val = adv_val.view(-1, 5, self.num_quant)
            state_val = state_val.unsqueeze(1)
            # batch_size x 5 x 200
            q_val = (state_val + adv_val - adv_val.mean(1, keepdim=True)).mean(2)

            actions = q_val.argmax(1).tolist()

        else:
            q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)
            # print(q_val.shape)
            actions = torch.argmax(q_val, 1).tolist()

        self.hidden = self.hidden.unsqueeze(0)

        return actions, q_val.numpy(), self.hidden[0].numpy()

    def reset(self):
        self.hidden = None

    def bootstrap(self, obs, steps, hidden):
        batch_size = obs.size(0)
        step = obs.size(1)
        hidden = hidden.unsqueeze(0)

        obs = obs.contiguous().view(-1, self.obs_dim, 9, 9)

        latent = self.obs_encoder(obs)

        latent = latent.view(batch_size, step, 16*7*7)

        latent = pack_padded_sequence(latent, steps, batch_first=True, enforce_sorted=False)

        self.recurrent.flatten_parameters()
        _, hidden = self.recurrent(latent, hidden)

        hidden = torch.squeeze(hidden)
        
        adv_val = self.adv(hidden)
        state_val = self.state(hidden)

        if self.distributional:
            adv_val = adv_val.view(-1, 5, self.num_quant)
            state_val = state_val.unsqueeze(1)
            q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)

        else:
            q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)

        return q_val