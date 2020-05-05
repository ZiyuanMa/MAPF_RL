from environment import Environment
import numpy as np
from typing import Dict, List, Optional


class VecEnv:
    def __init__(self, num:int=8):
        self.envs = [Environment() for _ in range(num)]
        self.obs = [ None for _ in range(num) ]
        self.pos = [ None for _ in range(num) ]
        self.done_env_ids = []

    def reset(self):
        
        if len(self.done_env_ids) == 0:
            # reset all environments
            for env_id, env in enumerate(self.envs):
                o, p = env.reset()
                
                self.obs[env_id] = o
                self.pos[env_id] = p

        else:
            # reset done environments
            for env_id in self.done_env_ids:
                o, p = self.envs[env_id].reset()
                self.obs[env_id] = o
                self.pos[env_id] = p

            self.done_env_ids.clear()

        return np.stack(self.obs), np.stack(self.pos)

    def step(self, actions:List[List[list]]):
        rewards = []
        dones = []

        assert len(self.done_env_ids) == 0, 'need reset environment'
        assert len(actions) == len(self.envs), '{} actions but {} environments'.format(len(actions), len(self.envs))

        for env_id, (env, action) in enumerate(zip(self.envs, actions)):
            (o, p), r, d, info = env.step(action)

            self.obs[env_id] = o
            self.pos[env_id] = p

            rewards.append(r)
            dones.append(d)

        return np.stack(self.obs), np.stack(self.pos), rewards, dones


        