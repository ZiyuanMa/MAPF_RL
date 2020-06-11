import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import log_softmax
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
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

class CommBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=config.num_comm_heads, num_layers=config.num_comm_layers):
        super().__init__()

        self.self_attn = nn.ModuleList([nn.MultiheadAttention(embed_dim, num_heads) for _ in range(num_layers)])

        self.update_cell = nn.GRUCell(embed_dim, embed_dim)


    def forward(self, latent, comm_mask):

        # agent indices of agent that use communication
        comm_idx = (comm_mask.sum(dim=-1) > 1).nonzero(as_tuple=True)
        # no agent use communication, return
        if len(comm_idx[0]) == 0:
            return latent

        # print(comm_mask)
        attn_mask = comm_mask==False
        if attn_mask.dim() == 3:
            attn_mask = attn_mask.repeat_interleave(config.num_comm_heads, 0)
        

        for attn_layer in self.self_attn:
            # print(latent.shape)
            # res_latent = attn_layer(latent, latent, latent, attn_mask=attn_mask)[0]
            # print(latent.shape)

            info = attn_layer(latent, latent, latent, attn_mask=attn_mask)[0]
            # info = attn_layer(latent, latent, latent)[0]
            
            if len(comm_idx)==1:
                batch_idx = torch.zeros(len(comm_idx[0]), dtype=torch.long)
                latent[comm_idx[0], batch_idx] = self.update_cell(info[comm_idx[0], batch_idx], latent[comm_idx[0], batch_idx])
            else:
                # print(comm_idx)
                latent[comm_idx[1], comm_idx[0]] = self.update_cell(info[comm_idx[1], comm_idx[0]], latent[comm_idx[1], comm_idx[0]])
            # if torch.isnan(latent).any():
            #     print(attn_mask)
            #     print(identity_mask)
            #     print(latent)
            #     raise Exception

        return latent



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

            nn.Conv2d(cnn_channel, cnn_channel, 3, 1, 1),
            nn.ReLU(True),

            nn.Conv2d(cnn_channel, cnn_channel*2, 3, 1, 1),
            nn.ReLU(True),

            ResBlock(cnn_channel*2, type='cnn'),

            nn.Conv2d(cnn_channel*2, 8, 1, 1),
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

        self.concat_encoder = ResBlock(self.latent_dim)
        
        # nn.Sequential(
        #     ResBlock(self.latent_dim), 
        #     ResBlock(self.latent_dim),
        # )

        self.recurrent = nn.GRUCell(self.latent_dim, self.latent_dim)

        self.comm = CommBlock(self.latent_dim)

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

    @torch.no_grad()
    def step(self, obs, pos):
        
        obs_latent = self.obs_encoder(obs)
        pos_latent = self.pos_encoder(pos)
        concat_latent = torch.cat((obs_latent, pos_latent), dim=1)
        latent = self.concat_encoder(concat_latent)

        if self.hidden is None:
            self.hidden = self.recurrent(latent)
        else:
            self.hidden = self.recurrent(latent, self.hidden)

        # if torch.isnan(self.hidden).any():
        #     raise RuntimeError(self.hidden)
        # from num_agents x latent_dim become num_agents x 1 x latent_dim
        self.hidden = self.hidden.unsqueeze(1)

        # print(self.hidden)

        # masks for communication block
        agents_pos = pos[:, :2]
        pos_mat = (agents_pos.unsqueeze(1)-agents_pos.unsqueeze(0))
        dis_mat = (pos_mat[:,:,0]**2+pos_mat[:,:,1]**2).sqrt()

        # mask out agents that out of range of FOV
        # print(pos_mat)
        in_obs_mask = (pos_mat<=config.obs_radius).all(2)
        # mask out agents that too far away from agent
        _, ranking = dis_mat.topk(config.max_comm_agents, dim=1, largest=False)
        dis_mask = torch.zeros((config.num_agents, config.num_agents), dtype=torch.bool)
        dis_mask.scatter_(1, ranking, True)
        # print(in_obs_mask)

        comm_mask = torch.bitwise_and(in_obs_mask, dis_mask)
        # assert dis_mask[0, 0] == 0, dis_mat
        self.hidden = self.comm(self.hidden, comm_mask)
        # print(self.hidden)
        self.hidden = self.hidden.squeeze()

        # if torch.isnan(self.hidden).any():
        #     print(self.hidden)

        adv_val = self.adv(self.hidden)
        state_val = self.state(self.hidden)

        if self.distributional:
            adv_val = adv_val.view(-1, 5, self.num_quant)
            state_val = state_val.unsqueeze(1)
            # batch_size x 5 x 200
            q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)

            actions = q_val.mean(2).argmax(1).tolist()
        else:
            q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)
            actions = torch.argmax(q_val, 1).tolist()

        return actions, q_val

    def reset(self):
        self.hidden = None

    def bootstrap(self, obs, pos, steps, comm_mask):
        # comm_mask size: batch_size x bt_steps x num_agents x num_agents
        obs = obs.contiguous().view(-1, self.obs_dim, 9, 9)
        pos = pos.contiguous().view(-1, self.pos_dim)

        obs_latent = self.obs_encoder(obs)
        pos_latent = self.pos_encoder(pos)

        concat_latent = torch.cat((obs_latent, pos_latent), dim=1)
        latent = self.concat_encoder(concat_latent)

        # latent = latent.split(steps)
        # latent = pad_sequence(latent, batch_first=True)

        latent = latent.view(config.batch_size*config.num_agents, config.bt_steps, self.latent_dim).transpose(0, 1)

        hidden_buffer = []
        hidden = self.recurrent(latent[0])
        hidden = hidden.view(config.batch_size, config.num_agents, self.latent_dim).transpose(0, 1)
        hidden = self.comm(hidden, comm_mask[:, 0])
        hidden = hidden.transpose(0, 1).view(config.batch_size*config.num_agents, self.latent_dim)
        hidden_buffer.append(hidden[torch.arange(0, config.batch_size*config.num_agents, config.num_agents)])
        for i in range(1, config.bt_steps):
            # hidden size: batch_size*num_agents x self.latent_dim
            hidden = self.recurrent(latent[i], hidden)
            hidden = hidden.view(config.batch_size, config.num_agents, self.latent_dim).transpose(0, 1)
            hidden = self.comm(hidden, comm_mask[:, i])
            hidden = hidden.transpose(0, 1).view(config.batch_size*config.num_agents, self.latent_dim)
            # only hidden from agent 0
            hidden_buffer.append(hidden[torch.arange(0, config.batch_size*config.num_agents, config.num_agents)])

        # hidden buffer size: batch_size x bt_steps x self.latent_dim
        hidden_buffer = torch.stack(hidden_buffer).transpose(0, 1)

        # hidden size: batch_size x self.latent_dim
        hidden = hidden_buffer[torch.arange(config.batch_size), steps-1]

        adv_val = self.adv(hidden)
        state_val = self.state(hidden)

        if self.distributional:
            adv_val = adv_val.view(-1, 5, self.num_quant)
            state_val = state_val.unsqueeze(1)
            q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)

        else:
            q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)

        return q_val