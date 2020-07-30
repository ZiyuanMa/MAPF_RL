import torch
import torch.nn as nn
import torch.nn.functional as F
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


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W_Q = nn.Linear(input_dim, output_dim * num_heads)
        self.W_K = nn.Linear(input_dim, output_dim * num_heads)
        self.W_V = nn.Linear(input_dim, output_dim * num_heads)
        self.W_O = nn.Linear(output_dim * num_heads, output_dim, bias=False)

    def forward(self, input, attn_mask):
        # input: [batch_size x num_agents x input_dim]
        batch_size, num_agents, input_dim = input.size()
        assert input_dim == self.input_dim
        
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(input).view(batch_size, num_agents, self.num_heads, -1).transpose(1,2)  # q_s: [batch_size x num_heads x num_agents x output_dim]
        k_s = self.W_K(input).view(batch_size, num_agents, self.num_heads, -1).transpose(1,2)  # k_s: [batch_size x num_heads x num_agents x output_dim]
        v_s = self.W_V(input).view(batch_size, num_agents, self.num_heads, -1).transpose(1,2)  # v_s: [batch_size x num_heads x num_agents x output_dim]

        # print(attn_mask.dim())
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
        assert attn_mask.size(0) == batch_size, 'mask dim {} while batch size {}'.format(attn_mask.size(0), batch_size)

        attn_mask = attn_mask.unsqueeze(1).repeat_interleave(self.num_heads, 1) # attn_mask : [batch_size x num_heads x num_agents x num_agents]
        assert attn_mask.size() == (batch_size, self.num_heads, num_agents, num_agents)

        # context: [batch_size x num_heads x num_agents x output_dim]
        scores = torch.matmul(q_s, k_s.transpose(-1, -2)) / (self.output_dim**0.5) # scores : [batch_size x n_heads x num_agents x num_agents]

        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = F.softmax(scores, dim=-1)
        # print(attn.shape)
        # print(v_s.shape)
        context = torch.matmul(attn, v_s)
        # print(context.shape)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_agents, self.num_heads * self.output_dim) # context: [batch_size x len_q x n_heads * d_v]
        output = self.W_O(context)
        # print(output.shape)
        return output # output: [batch_size x num_agents x output_dim]

class CommBlock(nn.Module):
    def __init__(self, input_dim, output_dim=64, num_heads=config.num_comm_heads, num_layers=config.num_comm_layers):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.self_attn = MultiHeadAttention(input_dim, output_dim, num_heads)

        self.update_cell = nn.GRUCell(output_dim, input_dim)


    def forward(self, latent, comm_mask):
        num_agents = latent.size(1)

        # agent indices of agent that use communication
        update_mask = comm_mask.sum(dim=-1) > 1
        comm_idx = update_mask.nonzero(as_tuple=True)

        # no agent use communication, return
        if len(comm_idx[0]) == 0:
            return latent

        if len(comm_idx)>1:
            update_mask = update_mask.unsqueeze(2)

        # print(comm_mask)
        attn_mask = comm_mask==False

        for _ in range(2):

            info = self.self_attn(latent, attn_mask=attn_mask)
            # latent = attn_layer(latent, attn_mask=attn_mask)
            # print(info.shape)
            if len(comm_idx)==1:

                batch_idx = torch.zeros(len(comm_idx[0]), dtype=torch.long)
                # print(info[batch_idx, comm_idx[0]].shape)
                # print(latent[batch_idx, comm_idx[0]].shape)
                latent[batch_idx, comm_idx[0]] = self.update_cell(info[batch_idx, comm_idx[0]], latent[batch_idx, comm_idx[0]])
            else:
                update_info = self.update_cell(info.view(-1, self.output_dim), latent.view(-1, self.input_dim)).view(config.batch_size, num_agents, self.input_dim)
                # update_mask = update_mask.unsqueeze(2)
                latent = torch.where(update_mask, update_info, latent)
                # latent[comm_idx] = self.update_cell(info[comm_idx], latent[comm_idx])
                # latent = self.update_cell(info, latent)
                # print(latent.shape)

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
            nn.Conv2d(config.obs_shape[0], 128, 3, 1, 1),
            nn.ReLU(True),

            ResBlock(128, type='cnn'),

            ResBlock(128, type='cnn'),

            ResBlock(128, type='cnn'),

            nn.Conv2d(128, 16, 1, 1),
            nn.ReLU(True),

            nn.Flatten(),

        )

        self.recurrent = nn.GRUCell(16*7*7, self.latent_dim)

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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def step(self, obs, pos):
        # print(obs.shape)
        num_agents = obs.size(0)
        obs_latent = self.obs_encoder(obs)
        pos_latent = self.pos_encoder(pos)
        concat_latent = torch.cat((obs_latent, pos_latent), dim=1)
        latent = self.concat_encoder(concat_latent)

        if self.hidden is None:
            self.hidden = self.recurrent(latent)
        else:
            self.hidden = self.recurrent(latent, self.hidden)

        # from num_agents x latent_dim become num_agents x 1 x latent_dim
        self.hidden = self.hidden.unsqueeze(0)

        # masks for communication block
        agents_pos = pos[:, :2]
        pos_mat = (agents_pos.unsqueeze(1)-agents_pos.unsqueeze(0))
        dis_mat = (pos_mat[:,:,0]**2+pos_mat[:,:,1]**2).sqrt()

        # mask out agents that out of range of FOV
        # print(pos_mat)
        in_obs_mask = (pos_mat<=config.obs_radius).all(2)
        # mask out agents that too far away from agent
        _, ranking = dis_mat.topk(min(config.max_comm_agents, num_agents), dim=1, largest=False)
        dis_mask = torch.zeros((num_agents, num_agents), dtype=torch.bool)
        dis_mask.scatter_(1, ranking, True)
        # print(in_obs_mask)

        comm_mask = torch.bitwise_and(in_obs_mask, dis_mask)
        # assert dis_mask[0, 0] == 0, dis_mat
        self.hidden = self.comm(self.hidden, comm_mask)
        self.hidden = self.hidden.squeeze(0)

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
            # print(q_val.shape)
            actions = torch.argmax(q_val, 1).tolist()

        return actions, q_val.numpy(), self.hidden.numpy(), comm_mask.numpy()

    def reset(self):
        self.hidden = None

    def bootstrap(self, obs, pos, steps, hidden, comm_mask):
        # comm_mask size: batch_size x bt_steps x num_agents x num_agents
        max_steps = obs.size(1)
        num_agents = comm_mask.size(2)

        obs = obs.transpose(1, 2)
        pos = pos.transpose(1, 2)

        obs = obs.contiguous().view(-1, self.obs_dim, 9, 9)
        pos = pos.contiguous().view(-1, self.pos_dim)

        obs_latent = self.obs_encoder(obs)
        pos_latent = self.pos_encoder(pos)
        concat_latent = torch.cat((obs_latent, pos_latent), dim=1)
        latent = self.concat_encoder(concat_latent)

        latent = latent.view(config.batch_size*num_agents, max_steps, self.latent_dim).transpose(0, 1)

        hidden_buffer = []
        for i in range(max_steps):
            # hidden size: batch_size*num_agents x self.latent_dim
            hidden = self.recurrent(latent[i], hidden)
            hidden = hidden.view(config.batch_size, num_agents, self.latent_dim)
            hidden = self.comm(hidden, comm_mask[:, i])
            # only hidden from agent 0
            hidden_buffer.append(hidden[:, 0])
            hidden = hidden.view(config.batch_size*num_agents, self.latent_dim)

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