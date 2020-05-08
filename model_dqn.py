import torch
import torch.nn as nn
import torch.nn.functional as F

import config

# class ResBlock(nn.Module):
#     def __init__(self, channel):
#         super().__init__()

#         self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(channel)

#         self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(channel)

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = F.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         out += identity

#         out = F.relu(out)

#         return out

class ResBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.linear1 = nn.Linear(channel, channel)

        self.linear2 = nn.Linear(channel, channel)

    def forward(self, x):
        identity = x

        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)

        x += identity

        x = F.relu(x)

        return x

class Network(nn.Module):
    def __init__(self, cnn_channel=config.cnn_channel,
                obs_dim=config.obs_dim, obs_latent_dim=config.obs_latent_dim,
                pos_dim=config.pos_dim, pos_latent_dim=config.pos_latent_dim):

        super().__init__()

        self.obs_dim = obs_dim
        self.pos_dim = pos_dim

        self.obs_encoder = nn.Sequential(
            nn.Conv2d(obs_dim, cnn_channel, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(cnn_channel, cnn_channel, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(cnn_channel, cnn_channel, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(cnn_channel, 4, 1, 1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(4*9*9, obs_latent_dim),
            nn.ReLU(True),
        )

        self.pos_encoder = nn.Sequential(
            nn.Linear(pos_dim, pos_latent_dim),
            nn.ReLU(True),
            nn.Linear(pos_latent_dim, pos_latent_dim),
            nn.ReLU(True),
        )

        self.concat_encoder = nn.Sequential(
            ResBlock(obs_latent_dim+pos_latent_dim), 
            # ResBlock(obs_latent_dim+pos_latent_dim),
        )

        self.recurrent = nn.GRU(obs_latent_dim+pos_latent_dim, obs_latent_dim+pos_latent_dim, batch_first=True)

        # dueling q structure
        self.adv = nn.Linear(obs_latent_dim+pos_latent_dim, 5)
        self.state = nn.Linear(obs_latent_dim+pos_latent_dim, 1)


        for _, m in self.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, obs, pos):

        if obs.dim() == 5:
            batch_size = obs.size(0)
            obs = obs.view(-1, self.obs_dim, 9, 9)
            pos = pos.view(-1, self.pos_dim)
        else:
            batch_size = 1

        obs_latent = self.obs_encoder(obs)
        pos_latent = self.pos_encoder(pos)
        concat_latent = torch.cat((obs_latent, pos_latent), dim=1)
        latent = self.concat_encoder(concat_latent)

        adv_val = self.adv(latent)
        state_val = self.state(latent)

        q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)

        if batch_size != 1:
            q_val = q_val.view(batch_size, -1, 5)

        return q_val

    def bootstrap(self, obs, pos):
        if obs.dim() == 5:
            batch_size = obs.size(0)
            obs = obs.view(-1, self.obs_dim, 9, 9)
            pos = pos.view(-1, self.pos_dim)
        else:
            batch_size = 1

        