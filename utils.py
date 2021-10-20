import numpy as np
import gym
import torch
from gym.wrappers import LazyFrames


class NumpifyAction(gym.ActionWrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

    def action(self, act) -> np.ndarray:
        return act.squeeze().cpu().numpy()


def obs_to_tensor(obs: LazyFrames) -> torch.Tensor:
    return torch.FloatTensor(obs).squeeze(-1)
