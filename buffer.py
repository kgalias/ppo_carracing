from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch


@dataclass  # (kw_only=True) # available from Python 3.10
class TrajectoryBatch:
    obs: torch.Tensor
    act: torch.Tensor
    act_log_prob: torch.Tensor
    val: torch.Tensor
    rew: torch.Tensor
    adv: Optional[torch.Tensor] = None
    ret: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> TrajectoryBatch:
        return TrajectoryBatch(obs=self.obs.to(device), act=self.act.to(device),
                               act_log_prob=self.act_log_prob.to(device),
                               val=self.val.to(device), rew=self.rew.to(device), adv=self.adv.to(device),
                               ret=self.ret.to(device))


class TrajectoryBuffer:
    def __init__(self, length: int, obs_shape: Tuple[int, int, int], act_shape: Tuple[int], gamma: float, lam: float,
                 normalize_adv: bool) -> None:
        self.obs = torch.zeros((length, *obs_shape))
        self.act = torch.zeros((length, *act_shape))
        self.act_log_prob = torch.zeros(length)
        self.val = torch.zeros(length)
        self.rew = torch.zeros(length)
        self.adv = torch.zeros(length)
        self.ret = torch.zeros(length)

        self.gamma = gamma
        self.lam = lam
        self.normalize_adv = normalize_adv

        self.current_length = 0
        self.capacity = length

    def store(self, transition: TrajectoryBatch) -> None:
        assert self.current_length < self.capacity, 'Buffer full, cannot store'
        self.obs[self.current_length] = transition.obs
        self.act[self.current_length] = transition.act
        self.act_log_prob[self.current_length] = transition.act_log_prob
        self.val[self.current_length] = transition.val
        self.rew[self.current_length] = transition.rew
        self.current_length += 1

    def retrieve_and_clear(self) -> TrajectoryBatch:
        assert self.current_length == self.capacity, 'Buffer not full, cannot retrieve'
        self.current_length = 0
        return TrajectoryBatch(obs=self.obs, act=self.act, act_log_prob=self.act_log_prob,
                               val=self.val, rew=self.rew, adv=self.adv, ret=self.ret)

    def calculate_norm_advantage_and_return(self, last_val: Union[torch.Tensor, float]) -> None:
        val_with_bootstrap = torch.cat([self.val, torch.tensor(last_val).unsqueeze(0)])
        last_gae_lam = 0
        for t in reversed(range(self.capacity)):
            delta = self.rew[t] + self.gamma * val_with_bootstrap[t + 1] - val_with_bootstrap[t]
            last_gae_lam = delta + self.gamma * self.lam * last_gae_lam
            self.adv[t] = last_gae_lam
        self.ret = self.adv + self.val
        if self.normalize_adv:
            self.adv = (self.adv - self.adv.mean()) / (self.adv.std() + 1e-8)
