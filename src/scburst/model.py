"""
BranchingSDE models with growth-dependent diffusion.

PDE:  d rho/dt = -div(v rho) + (sigma^2 + g delta^2)/2 * Laplacian(rho) + g rho
SDE:  dz = v(z) dt + sqrt(sigma^2 + g delta^2) dW
      d(lnw) = g dt

v(z) = -nabla phi(z)   (potential gradient)
g >= 0                  (softplus constraint)
"""

import torch
import torch.nn as nn

from .networks import (
    PotentialNetwork,
    NonNegativeGrowthNetwork,
    NonNegativeTimeDependentGrowthNetwork,
    initialize_weights,
)


class BranchingSDE(nn.Module):
    """
    Growth-dependent diffusion SDE with time-independent growth g(x).

    State: y = (z, lnw)  where z is position, lnw is log-weight.
    Compatible with torchsde.
    """

    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(
        self,
        in_out_dim,
        sigma=0.05,
        delta=0.1,
        # Potential network
        potential_hidden_dim=64,
        potential_n_hiddens=4,
        potential_activation="tanh",
        potential_arch="mlp",
        potential_layer_norm=False,
        potential_dropout=0.0,
        # Growth network
        growth_hidden_dim=64,
        growth_n_hiddens=3,
        growth_activation="tanh",
        growth_arch="mlp",
        growth_layer_norm=False,
        growth_dropout=0.0,
    ):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.sigma = sigma
        self.delta = delta

        self.potential_net = PotentialNetwork(
            in_dim=in_out_dim,
            hidden_dim=potential_hidden_dim,
            n_hiddens=potential_n_hiddens,
            activation=potential_activation,
            arch=potential_arch,
            use_layer_norm=potential_layer_norm,
            dropout=potential_dropout,
        )

        self.growth_net = NonNegativeGrowthNetwork(
            in_dim=in_out_dim,
            hidden_dim=growth_hidden_dim,
            n_hiddens=growth_n_hiddens,
            activation=growth_activation,
            arch=growth_arch,
            use_layer_norm=growth_layer_norm,
            dropout=growth_dropout,
        )

    def velocity(self, z):
        """v(z) = -nabla phi(z)"""
        return -self.potential_net.gradient(z)

    def growth(self, z):
        """g(z) >= 0"""
        return self.growth_net(z)

    def potential(self, z):
        """phi(z)"""
        return self.potential_net(z)

    def f(self, t, y):
        """Drift for torchsde. y = (z, lnw) concatenated."""
        z = y[:, : self.in_out_dim]

        with torch.set_grad_enabled(True):
            if not z.requires_grad:
                z = z.requires_grad_(True)
            v = self.velocity(z)
            g_val = self.growth(z)

        return torch.cat([v, g_val], dim=1)

    def g(self, t, y):
        """Diffusion for torchsde. sqrt(sigma^2 + g*delta^2) for z, 0 for lnw."""
        z = y[:, : self.in_out_dim]

        with torch.set_grad_enabled(True):
            if not z.requires_grad:
                z = z.clone().requires_grad_(True)
            growth_rate = self.growth(z)

        diffusion_coeff = torch.sqrt(self.sigma**2 + growth_rate * self.delta**2)

        diffusion = torch.zeros_like(y)
        diffusion[:, : self.in_out_dim] = diffusion_coeff.expand(-1, self.in_out_dim)
        return diffusion


class BranchingSDE_TimeDep(nn.Module):
    """
    Growth-dependent diffusion SDE with time-dependent growth g(t, x).

    Same interface as BranchingSDE but growth depends on time.
    """

    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(
        self,
        in_out_dim,
        sigma=0.05,
        delta=0.1,
        # Potential network
        potential_hidden_dim=64,
        potential_n_hiddens=4,
        potential_activation="tanh",
        potential_arch="mlp",
        potential_layer_norm=False,
        potential_dropout=0.0,
        # Growth network
        growth_hidden_dim=64,
        growth_n_hiddens=3,
        growth_activation="tanh",
        growth_arch="mlp",
        growth_layer_norm=False,
        growth_dropout=0.0,
    ):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.sigma = sigma
        self.delta = delta

        self.potential_net = PotentialNetwork(
            in_dim=in_out_dim,
            hidden_dim=potential_hidden_dim,
            n_hiddens=potential_n_hiddens,
            activation=potential_activation,
            arch=potential_arch,
            use_layer_norm=potential_layer_norm,
            dropout=potential_dropout,
        )

        self.growth_net = NonNegativeTimeDependentGrowthNetwork(
            in_dim=in_out_dim,
            hidden_dim=growth_hidden_dim,
            n_hiddens=growth_n_hiddens,
            activation=growth_activation,
            arch=growth_arch,
            use_layer_norm=growth_layer_norm,
            dropout=growth_dropout,
        )

    def velocity(self, z):
        """v(z) = -nabla phi(z)"""
        return -self.potential_net.gradient(z)

    def growth(self, t, z):
        """g(t, z) >= 0"""
        return self.growth_net(t, z)

    def potential(self, z):
        """phi(z)"""
        return self.potential_net(z)

    def f(self, t, y):
        """Drift for torchsde. y = (z, lnw) concatenated."""
        z = y[:, : self.in_out_dim]

        with torch.set_grad_enabled(True):
            if not z.requires_grad:
                z = z.requires_grad_(True)
            v = self.velocity(z)
            g_val = self.growth(t, z)

        return torch.cat([v, g_val], dim=1)

    def g(self, t, y):
        """Diffusion for torchsde. sqrt(sigma^2 + g(t,z)*delta^2) for z, 0 for lnw."""
        z = y[:, : self.in_out_dim]

        with torch.set_grad_enabled(True):
            if not z.requires_grad:
                z = z.clone().requires_grad_(True)
            growth_rate = self.growth(t, z)

        diffusion_coeff = torch.sqrt(self.sigma**2 + growth_rate * self.delta**2)

        diffusion = torch.zeros_like(y)
        diffusion[:, : self.in_out_dim] = diffusion_coeff.expand(-1, self.in_out_dim)
        return diffusion
