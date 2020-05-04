import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


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
    def __init__(self):
        super().__init__()
        cnn_channel = 64
        latent_dim = 256
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(2, cnn_channel, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(cnn_channel, cnn_channel, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(cnn_channel, cnn_channel, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(cnn_channel, 4, 1, 1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(4*9*9, 240),
            nn.ReLU(True),
        )

        self.pos_encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(True),
            nn.Linear(16, 16),
            nn.ReLU(True),
        )

        self.concat_encoder = nn.Sequential(
            ResBlock(latent_dim), 
            ResBlock(latent_dim),
        )

        # self.recurrent = nn.GRUCell(latent_dim, latent_dim)

        self.pi = nn.Linear(latent_dim, 5)

        self.v = nn.Linear(latent_dim, 1)

    def forward(self, obs, pos):
        obs_latent = self.obs_encoder(obs)
        pos_latent = self.pos_encoder(pos)

        concat_latent = torch.cat((obs_latent, pos_latent), dim=1)

        latent = self.concat_encoder(concat_latent)

        logp = self.pi(latent)

        v = self.v(latent)

        return logp, v 