import pytest
import torch

from actor_critic import ActorCritic
from buffer import TrajectoryBatch, TrajectoryBuffer


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
        return TrajectoryBuffer(10, (3, 3, 3), (2,), 0.1, 0.5, False)

    @pytest.fixture
    def example_args(self):
        obs = torch.arange(end=3 * 3 * 3).view(1, 3, 3, 3)
        act = torch.Tensor([1., 2.])
        act_log_prob = torch.Tensor([0.2])
        val = torch.Tensor([1.])
        rew = torch.Tensor([0.1])
        return TrajectoryBatch(obs=obs, act=act, act_log_prob=act_log_prob, val=val, rew=rew)

    def test_store(self, example_tb, example_args):
        example_tb.store(example_args)
        assert example_tb.current_length == 1

    def test_retrieve_when_not_full(self, example_tb, example_args):
        example_tb.store(example_args)
        with pytest.raises(Exception):
            _ = example_tb.retrieve_and_clear()
