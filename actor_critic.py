import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, in_channels: int):
        super(ActorCritic, self).__init__()
        self.cnn = nn.Sequential(  # (4, 64, 64)
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=4, stride=2),  # (32, 31, 31)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),  # (64, 15, 15)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),  # (128, 7, 7)
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2),  # (256, 3, 3)
            nn.ReLU(),
        )

        self.pi = nn.Sequential(
            nn.Linear(256 * 3 * 3, 128),
        )

        self.mu = nn.Linear(128, 3)

        self.sigma = nn.Sequential(
            nn.Linear(128, 3),
            nn.Softplus(),
        )

        self.v = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1),
        )

    def forward(self, obs: torch.Tensor):
        conv = self.cnn(obs)
        conv = conv.view(-1, 256 * 3 * 3)  # reshape for dense
        pi = self.pi(conv)
        v = self.v(conv)
        mu, sigma = self.mu(pi), self.sigma(pi)
        return mu, sigma, v
