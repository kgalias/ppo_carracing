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
