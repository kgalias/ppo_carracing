from typing import Tuple

import gym
import mlflow
import torch
from gym.wrappers import LazyFrames
from torch.distributions import MultivariateNormal

from actor_critic import ActorCritic
from utils import obs_to_tensor


class PPOAgent(object):
    def __init__(self, env: gym.Env, actor_critic: ActorCritic, optimizer: torch.optim.Optimizer,
                 device: torch.device, n_ppo_epochs: int, epsilon: float, vf_coef: float, max_grad_norm: float):
        self.env = env
        self.actor_critic = actor_critic
        self.optimizer = optimizer
        self.device = device
        self.n_ppo_epochs = n_ppo_epochs
        self.epsilon = epsilon
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.actor_critic.to(self.device)

    @torch.no_grad()
    def act(self, obs: LazyFrames) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = obs_to_tensor(obs).unsqueeze(0).to(self.device)
        mu, sigma, v = self.actor_critic(obs)
        dist = MultivariateNormal(mu, torch.diag_embed(sigma))  # diagonal covariance matrix
        action = dist.sample()
        lower = torch.tensor(self.env.action_space.low).expand_as(action).to(self.device)
        upper = torch.tensor(self.env.action_space.high).expand_as(action).to(self.device)
        action = torch.max(torch.min(action, upper), lower)  # clip the actions to actual intervals
        action_log_prob = dist.log_prob(action)  # TODO: fix log_prob calculation for truncated normal
        return action, action_log_prob, v

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, sigma, v = self.actor_critic(obs)
        dist = MultivariateNormal(mu, torch.diag_embed(sigma))
        action_log_prob = dist.log_prob(action)
        return action_log_prob, v

    def learn(self, obs, act, old_act_log_prob, val, rew, adv, ret):
        for _ in range(self.n_ppo_epochs):
            obs, act, old_act_log_prob = obs.to(self.device), act.to(self.device), old_act_log_prob.to(self.device)
            val, rew, adv, ret = val.to(self.device), rew.to(self.device), adv.to(self.device), ret.to(self.device)
            new_act_log_prob, v = self.evaluate(obs, act)
            ratio = torch.exp(new_act_log_prob - old_act_log_prob)
            clip_adv = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv
            loss_pi = -torch.min(ratio * adv, clip_adv).mean()

            loss_v = ((v - ret) ** 2).mean()
            loss = loss_pi + self.vf_coef * loss_v
            mlflow.log_metric('loss', loss.item())
            mlflow.log_metric('policy loss', loss_pi.item())
            mlflow.log_metric('critic loss', loss_v.item())

            # track approximate KL and fraction clipped
            approx_kl = (old_act_log_prob - new_act_log_prob).mean().item()
            clipped = ratio.gt(1 + self.epsilon) | ratio.lt(1 - self.epsilon)
            clip_frac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
            mlflow.log_metric('approx_kl', approx_kl)
            mlflow.log_metric('clip_frac', clip_frac)

            self.optimizer.zero_grad()
            loss.backward()

            # track gradient norm
            parameters = [p for p in self.actor_critic.parameters() if p.grad is not None and p.requires_grad]
            total_grad_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach()).to(self.device) for p in parameters]), 2.0
            ).item()
            mlflow.log_metric('total_grad_norm', total_grad_norm)

            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
