from typing import Tuple, Union

import gym
import torch
from gym.wrappers import LazyFrames


class NumpifyAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, act):
        return act.squeeze().cpu().numpy()


def obs_to_tensor(obs: LazyFrames):
    return torch.FloatTensor(obs).squeeze(-1)


class TrajectoryBuffer:
    def __init__(self, length: int, obs_shape: Tuple[int, int, int], act_shape: Tuple[int], gamma: float, lam: float):
        self.obs = torch.zeros((length, *obs_shape))
        self.act = torch.zeros((length, *act_shape))
        self.act_log_prob = torch.zeros(length)
        self.val = torch.zeros(length)
        self.rew = torch.zeros(length)
        self.adv = torch.zeros(length)
        self.ret = torch.zeros(length)

        self.gamma = gamma
        self.lam = lam

        self.current_length = 0
        self.capacity = length

    def store(self, obs: LazyFrames, act, act_log_prob, val, rew):
        assert self.current_length < self.capacity, 'Buffer full, cannot store'
        self.obs[self.current_length] = obs_to_tensor(obs)
        self.act[self.current_length] = act
        self.act_log_prob[self.current_length] = act_log_prob
        self.val[self.current_length] = val
        self.rew[self.current_length] = rew
        self.current_length += 1

    def retrieve_and_clear(self):
        assert self.current_length == self.capacity, 'Buffer not full, cannot retrieve'
        self.current_length = 0
        return self.obs, self.act, self.act_log_prob, self.val, self.rew, self.adv, self.ret

    def calculate_norm_advantage_and_return(self, last_val: Union[torch.Tensor, float]):
        val_with_bootstrap = torch.cat([self.val, torch.tensor(last_val).unsqueeze(0)])
        last_gae_lam = 0
        for t in reversed(range(self.capacity)):
            delta = self.rew[t] + self.gamma * val_with_bootstrap[t + 1] - val_with_bootstrap[t]
            last_gae_lam = delta + self.gamma * self.lam * last_gae_lam
            self.adv[t] = last_gae_lam
        self.ret = self.adv + self.val
        # self.adv = (self.adv - self.adv.mean()) / (self.adv.std() + 1e-8)  # normalize
