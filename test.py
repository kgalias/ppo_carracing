import pytest
import torch
import numpy as np
from gym.wrappers import LazyFrames

from actor_critic import ActorCritic
from utils import TrajectoryBuffer


class TestActorCritic:
    def test_gradients(self):
        obs = torch.arange(end=4 * 64 * 64, dtype=torch.float32).view(-1, 4, 64, 64)
        mu, sigma, v = torch.tensor([[1., 1., 1.]]), torch.tensor([[1., 1., 1.]]), torch.tensor([[1.]])
        ac = ActorCritic(4)
        mu_, sigma_, v_ = ac(obs)
        loss = (mu - mu_).mean() + (sigma - sigma_).mean() + (v - v_).mean()
        loss.backward()
        assert all([p.grad is not None for p in ac.parameters() if p.requires_grad])


class TestTrajectoryBuffer:
    @pytest.fixture
    def example_tb(self):
        return TrajectoryBuffer(10, (3, 3, 3), (2,), 0.1, 0.5)

    @pytest.fixture
    def example_args(self):
        obs = LazyFrames(np.arange(stop=3 * 3 * 3).reshape((1, 3, 3, 3)))
        act = torch.Tensor([1., 2.])
        act_log_prob = 0.2
        val = torch.Tensor([1.])
        rew = 0.1
        return obs, act, act_log_prob, val, rew

    def test_store(self, example_tb, example_args):
        example_tb.store(*example_args)
        assert example_tb.current_length == 1

    # def test_retrieve(self):
    #     assert False
    #
    # def test_calculate_norm_advantage_and_return(self):
    #     assert False
