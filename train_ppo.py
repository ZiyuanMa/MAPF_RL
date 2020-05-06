import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
import gym
import time
from typing import Dict, List, Optional

from vec_env import VecEnv
from model import Network
from logx import EpochLogger
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, size, env_num, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, 2, 9, 9), dtype=np.float32)
        self.pos_buf = np.zeros((size, 4), dtype=np.float32)
        self.act_buf = np.zeros((size, 5), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.env_num = env_num
        self.gamma, self.lam = gamma, lam
        self.ptr, self.max_size = 0, size//env_num
        self.path_start_idx = [0 for _ in range(env_num)]

    def store(self, obs:List[np.ndarray], pos:List[np.ndarray], act:List[int], rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        for i in range(self.env_num):
            self.obs_buf[self.ptr+i*self.max_size] = obs[i]
            self.pos_buf[self.ptr+i*self.max_size] = pos[i]
            self.act_buf[self.ptr+i*self.max_size] = act[i]
            self.rew_buf[self.ptr+i*self.max_size] = rew[i]
            self.val_buf[self.ptr+i*self.max_size] = val[i]
            self.logp_buf[self.ptr+i*self.max_size] = logp[i]
        self.ptr += 1

    def finish_path(self, env_id: int, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx[env_id]+env_id*self.max_size, self.ptr+env_id*self.max_size)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx[env_id] = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, [0 for _ in range(self.env_num)]
        # the next two lines implement the advantage normalization trick
        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        # self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = [ self.obs_buf, self.act_buf, self.ret_buf, self.logp_buf, self.val_buf, self.adv_buf ]
        return [ torch.as_tensor(i, dtype=torch.float32).to(device) for i in data ]



def ppo(env_name, env_num,
        steps_per_epoch, minibatch_size, epochs, max_ep_len,
        gamma=0.99, clip_ratio=0.2, seed=0, 
        lam=0.97, ent_coef=0.01, v_coef=1, grad_norm=0.5,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10, actor_critic=Network):
    """
    Proximal Policy Optimization (by clipping), 
    with early stopping based on approximate KL
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.
        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================
            The ``act`` method behaves the same as ``step`` but only returns ``a``.
            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================
            The ``v`` module's forward call should accept a batch of observations
            and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================
        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.
        seed (int): Seed for random number generators.
        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.
        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.
        gamma (float): Discount factor. (Always between 0 and 1.)
        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 
        pi_lr (float): Learning rate for policy optimizer.
        vf_lr (float): Learning rate for value function optimizer.
        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)
        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.
        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    # setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = VecEnv(env_num)

    # Create actor-critic module
    ac = actor_critic()
    ac.to(device)

    # Sync params across processes
    # sync_params(ac)

    # Count variables
    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = steps_per_epoch // env_num
    buf = PPOBuffer(steps_per_epoch, env_num)

    # Set up function for computing PPO policy loss
    def compute_pi_loss(latent, act, adv, old_logp):

        # Policy loss
        pi, logp = ac.pi(latent, act)
        ent_loss = pi.entropy().mean()

        # pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - old_logp)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        pi_loss = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        # approx_kl = (old_logp - logp).mean().item()
        # ent = pi.entropy().mean().item()
        # clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        # clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        # pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return pi_loss, ent_loss

    # Set up function for computing value loss
    def compute_v_loss(latent, old_val, ret):
        val = ac.v(latent)
        clip_val = old_val + (val - old_val).clamp(-clip_ratio, clip_ratio)
        return 0.5* (torch.max(
                        (val - ret).pow(2),
                        (clip_val - ret).pow(2)
                    )).mean()
        # return 0.5 * ((ac.v(latent) - ret).clamp(-clip_ratio, clip_ratio)**2).mean()

    # Set up optimizers for policy and value function
    optimizer = Adam(ac.parameters(), lr=2.5e-4, eps=1e-5)
    scheduler = LambdaLR(optimizer, lambda epoch: 1 - epoch / epochs)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        # pi_l_old, pi_info_old = compute_pi_loss(data)
        # pi_l_old = pi_l_old.item()
        # v_l_old = compute_v_loss(data).item()

        # # Train policy with multiple steps of gradient descent
        # for i in range(train_pi_iters):
        #     pi_optimizer.zero_grad()
        #     pi_loss, pi_info = compute_pi_loss(data)
        #     # kl = mpi_avg(pi_info['kl'])
        #     kl = pi_info['kl']
        #     if kl > 1.5 * target_kl:
        #         logger.log('Early stopping at step %d due to reaching max kl.'%i)
        #         break
        #     pi_loss.backward()
        #     # mpi_avg_grads(ac.pi)    # average grads across MPI processes
        #     pi_optimizer.step()

        # logger.store(StopIter=i)

        # # Value function learning
        # for i in range(train_v_iters):
        #     vf_optimizer.zero_grad()
        #     v_loss = compute_v_loss(data)
        #     v_loss.backward()
        #     # mpi_avg_grads(ac.v)    # average grads across MPI processes
        #     vf_optimizer.step()
        total_loss = 0
        loader = DataLoader(list(zip(*data)), minibatch_size)

        for _ in range(4):
            total_loss = 0
            for obs, act, ret, logp, val, adv in loader:
                
                # trick: advantage normalization
                adv = (adv-adv.mean())/(adv.std()+1e-8)

                latent = ac.encoder(obs)

                pi_loss, ent_loss = compute_pi_loss(latent, act, adv, logp)
                v_loss = compute_v_loss(latent, val, ret)

                loss = pi_loss + v_coef*v_loss - ent_coef*ent_loss
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                if grad_norm is not None:
                    nn.utils.clip_grad_norm_(ac.parameters(), grad_norm)
                optimizer.step()

                logger.store(PiLoss=pi_loss.item(), VLoss=v_loss.item(), EntLoss=ent_loss.item())

        # print(total_loss/4/len(loader))
        scheduler.step()
        logger.store(Loss=total_loss/4/len(loader))


        # Log changes from update
        # kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        # logger.store(LossPi=pi_l_old, LossV=v_l_old,
        #              KL=kl, Entropy=ent, ClipFrac=cf,
        #              DeltaLossPi=(pi_loss.item() - pi_l_old),
        #              DeltaLossV=(v_loss.item() - v_l_old))

        # logger.store(LossPi=pi_l_old, LossV=v_l_old,
        #              KL=kl, Entropy=ent, ClipFrac=cf,
        #              DeltaLossPi=(pi_loss.item() - pi_l_old),
        #              DeltaLossV=(v_loss.item() - v_l_old))


    # Prepare for interaction with environment
    start_time = time.time()

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        ep_num = 0
        ep_ret = 0
        (obs, pos), ep_len = env.reset(), 0

        for t in range(local_steps_per_epoch):
            
            a, v, logp = ac.step(torch.as_tensor(obs, dtype=torch.float32).to(device), torch.as_tensor(pos, dtype=torch.float32).to(device))

            next_obs, next_pos, r, d = env.step(a)

            sum_r = [ sum(rew) for rew in r ]
            ep_ret += sum(sum_r)/env_num
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp)
            logger.store(VVals=v.mean())
            
            # Update obs (critical!)
            obs, pos = next_obs, next_pos

            done = True in d
            epoch_ended = t==local_steps_per_epoch-1

            if done or epoch_ended:

                # if trajectory didn't reach terminal state, bootstrap value target
                if epoch_ended:
                    concat_o = np.stack(o, axis=0)
                    _, v, _ = ac.step(torch.as_tensor(concat_o, dtype=torch.float32).to(device))
                    for i, d_ in enumerate(d):
                        if not d_:
                            buf.finish_path(i, v[i])

                if done:
                    
                    for i, d_ in enumerate(d):
                        if d_:
                            buf.finish_path(i)
                            ep_num += 1

                    o = env.reset()

                # if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    # logger.store(EpRet=ep_ret, EpLen=ep_len)

                ep_len = 0


        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Perform PPO update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', ep_ret)
        logger.log_tabular('EpNum', ep_num)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('PiLoss', average_only=True)
        logger.log_tabular('VLoss', average_only=True)
        logger.log_tabular('EntLoss', average_only=True)
        # logger.log_tabular('LossV', average_only=True)
        # logger.log_tabular('DeltaLossPi', average_only=True)
        # logger.log_tabular('DeltaLossV', average_only=True)
        # logger.log_tabular('Entropy', average_only=True)
        # logger.log_tabular('KL', average_only=True)
        # logger.log_tabular('ClipFrac', average_only=True)
        # logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

    torch.save(ac.state_dict(), 'model.pth')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MsPacman-v0')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--batch', type=int, default=256, help='minibatch size')
    parser.add_argument('--steps', type=int, default=1024, help='number of steps per epoch')
    parser.add_argument('--max_ep', type=int, default=128, help='max length for one episode')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--horizon', type=int, default=16)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()


    from run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(env_name=args.env, env_num=8,
        max_ep_len=args.max_ep,
        seed=args.seed, steps_per_epoch=args.steps, minibatch_size=args.batch, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
