"""
BranchingSDE model with growth-dependent diffusion.

PDE:  d rho/dt = -div(v rho) + (sigma^2 + g delta^2)/2 * Laplacian(rho) + g rho
SDE:  dz = v(z) dt + sqrt(sigma^2 + g delta^2) dW
      d(lnw) = g dt

Supports configurable velocity type (potential / free-form) and
time dependence for both velocity and growth networks.
"""

import torch
import torch.nn as nn

from .networks import (
    PotentialNetwork,
    TimeDependentPotentialNetwork,
    VelocityNetwork,
    TimeDependentVelocityNetwork,
    NonNegativeGrowthNetwork,
    NonNegativeTimeDependentGrowthNetwork,
    initialize_weights,
)


class BranchingSDE(nn.Module):
    """
    Unified growth-dependent diffusion SDE.

    State: y = (z, lnw)  where z is position, lnw is log-weight.
    Compatible with torchsde.

    Args:
        velocity_type: "potential" (v = -nabla phi) or "free" (v = MLP output).
        velocity_time_dependent: If True, velocity depends on (t, x).
        growth_time_dependent: If True, growth depends on (t, x).
    """

    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(
        self,
        in_out_dim,
        sigma=0.05,
        delta=0.1,
        # Velocity config
        velocity_type="potential",
        velocity_time_dependent=False,
        velocity_hidden_dim=64,
        velocity_n_hiddens=4,
        velocity_activation="tanh",
        velocity_arch="mlp",
        velocity_layer_norm=False,
        velocity_dropout=0.0,
        # Growth config
        growth_time_dependent=False,
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

        # Flags
        self.velocity_type = velocity_type
        self.velocity_time_dependent = velocity_time_dependent
        self.growth_time_dependent = growth_time_dependent
        self.has_potential = (velocity_type == "potential")

        # --- Velocity network (4 variants) ---
        vel_kw = dict(
            in_dim=in_out_dim,
            hidden_dim=velocity_hidden_dim,
            n_hiddens=velocity_n_hiddens,
            activation=velocity_activation,
            arch=velocity_arch,
            use_layer_norm=velocity_layer_norm,
            dropout=velocity_dropout,
        )
        if velocity_type == "potential":
            if velocity_time_dependent:
                self.potential_net = TimeDependentPotentialNetwork(**vel_kw)
            else:
                self.potential_net = PotentialNetwork(**vel_kw)
        elif velocity_type == "free":
            if velocity_time_dependent:
                self.velocity_net = TimeDependentVelocityNetwork(**vel_kw)
            else:
                self.velocity_net = VelocityNetwork(**vel_kw)
        else:
            raise ValueError(
                f"Unknown velocity_type: {velocity_type!r}. "
                "Must be 'potential' or 'free'."
            )

        # --- Growth network (2 variants) ---
        grw_kw = dict(
            in_dim=in_out_dim,
            hidden_dim=growth_hidden_dim,
            n_hiddens=growth_n_hiddens,
            activation=growth_activation,
            arch=growth_arch,
            use_layer_norm=growth_layer_norm,
            dropout=growth_dropout,
        )
        if growth_time_dependent:
            self.growth_net = NonNegativeTimeDependentGrowthNetwork(**grw_kw)
        else:
            self.growth_net = NonNegativeGrowthNetwork(**grw_kw)

    # ------------------------------------------------------------------
    # Public API — always takes (t, z)
    # ------------------------------------------------------------------

    def velocity(self, t, z):
        """v(t, z) — t is ignored when velocity is time-independent."""
        if self.velocity_type == "potential":
            if self.velocity_time_dependent:
                return -self.potential_net.gradient(t, z)
            else:
                return -self.potential_net.gradient(z)
        else:
            if self.velocity_time_dependent:
                return self.velocity_net(t, z)
            else:
                return self.velocity_net(z)

    def growth(self, t, z):
        """g(t, z) — t is ignored when growth is time-independent."""
        if self.growth_time_dependent:
            return self.growth_net(t, z)
        else:
            return self.growth_net(z)

    def potential(self, t, z):
        """phi(t, z) — only available for velocity_type='potential'."""
        if not self.has_potential:
            raise AttributeError(
                "potential() is not available for velocity_type='free'"
            )
        if self.velocity_time_dependent:
            return self.potential_net(t, z)
        else:
            return self.potential_net(z)

    # ------------------------------------------------------------------
    # torchsde interface
    # ------------------------------------------------------------------

    def f(self, t, y):
        """Drift for torchsde. y = (z, lnw) concatenated."""
        z = y[:, :self.in_out_dim]

        with torch.set_grad_enabled(True):
            if not z.requires_grad:
                z = z.requires_grad_(True)
            v = self.velocity(t, z)
            g_val = self.growth(t, z)

        return torch.cat([v, g_val], dim=1)

    def g(self, t, y):
        """Diffusion for torchsde. sqrt(sigma^2 + g*delta^2) for z, 0 for lnw."""
        z = y[:, :self.in_out_dim]

        with torch.set_grad_enabled(True):
            if not z.requires_grad:
                z = z.clone().requires_grad_(True)
            growth_rate = self.growth(t, z)

        diffusion_coeff = torch.sqrt(self.sigma**2 + growth_rate * self.delta**2)

        diffusion = torch.zeros_like(y)
        diffusion[:, :self.in_out_dim] = diffusion_coeff.expand(-1, self.in_out_dim)
        return diffusion


def BranchingSDE_TimeDep(
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
    """Backward-compatible factory: creates BranchingSDE with growth_time_dependent=True."""
    return BranchingSDE(
        in_out_dim=in_out_dim,
        sigma=sigma,
        delta=delta,
        velocity_type="potential",
        velocity_time_dependent=False,
        velocity_hidden_dim=potential_hidden_dim,
        velocity_n_hiddens=potential_n_hiddens,
        velocity_activation=potential_activation,
        velocity_arch=potential_arch,
        velocity_layer_norm=potential_layer_norm,
        velocity_dropout=potential_dropout,
        growth_time_dependent=True,
        growth_hidden_dim=growth_hidden_dim,
        growth_n_hiddens=growth_n_hiddens,
        growth_activation=growth_activation,
        growth_arch=growth_arch,
        growth_layer_norm=growth_layer_norm,
        growth_dropout=growth_dropout,
    )
