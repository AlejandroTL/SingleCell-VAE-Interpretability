import torch
import torch.nn as nn
from .types import *

torch.manual_seed(0)


class VAE(nn.Module):
    def __init__(self, input_dim: int, mid_dim: int, features: int, drop: int) -> None:
        super().__init__()

        self.features = features

        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=mid_dim),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(in_features=mid_dim, out_features=self.features * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.features, out_features=mid_dim),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(in_features=mid_dim, out_features=input_dim),
            nn.Tanh()
        )

    def reparametrize(self, mu: Tensor, log_var: Tensor) -> Tensor:

        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            sample = mu + (eps * std)
        else:
            sample = mu
        return sample

    def forward(self, x: Tensor) -> Tensor:

        mu_logvar = self.encoder(x).view(-1, 2, self.features)
        mu = mu_logvar[:, 0, :]
        log_var = mu_logvar[:, 1, :]

        z = self.reparametrize(mu, log_var)
        reconstruction = self.decoder(z)

        return reconstruction, mu, log_var, z


class AE(nn.Module):
    def __init__(self, input_dim: int, mid_dim: int, features: int, drop: int) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=mid_dim),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(in_features=mid_dim, out_features=features * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=features, out_features=mid_dim),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(in_features=mid_dim, out_features=input_dim),
            nn.Tanh()
        )

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        reconstruction = self.decoder(z)

        return reconstruction, z
