"""
DecoupledSDE — growth and diffusion are independent networks.

Unlike BranchingSDE where diffusion = sqrt(sigma^2 + 2*b*delta^2) couples
birth rate b to diffusion, DecoupledSDE has:
  - growth_net:    g(t,x) unconstrained (positive/negative)
  - diffusion_net: D(t,x) = sigma_min + softplus(NN output), always positive

SDE:  dz = v(z) dt + D(t,z) dW
      d(lnw) = g(t,z) dt
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import (
    PotentialNetwork,
    TimeDependentPotentialNetwork,
    VelocityNetwork,
    TimeDependentVelocityNetwork,
    ActivityNetwork,
    TimeDependentActivityNetwork,
    initialize_weights,
)


class DecoupledSDE(nn.Module):
    """
    SDE with independent growth and diffusion networks.

    State: y = (z, lnw)  where z is position, lnw is log-weight.
    Compatible with torchsde.

    Args:
        velocity_type: "potential" (v = -nabla phi) or "free" (v = MLP output).
        velocity_time_dependent: If True, velocity depends on (t, x).
        growth_time_dependent: If True, growth depends on (t, x).
        diffusion_time_dependent: If True, diffusion depends on (t, x).
        sigma_init: Initial diffusion coefficient value. Default 0.05.
        sigma_min: Minimum diffusion floor. Default 1e-4.
    """

    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(
        self,
        in_out_dim,
        sigma_init=0.05,
        sigma_min=1e-4,
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
        growth_time_dependent=True,
        growth_hidden_dim=64,
        growth_n_hiddens=3,
        growth_activation="tanh",
        growth_arch="mlp",
        growth_layer_norm=False,
        growth_dropout=0.0,
        # Diffusion config
        diffusion_time_dependent=True,
        diffusion_hidden_dim=64,
        diffusion_n_hiddens=3,
        diffusion_activation="tanh",
        diffusion_arch="mlp",
        diffusion_layer_norm=False,
        diffusion_dropout=0.0,
    ):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.sigma_min = sigma_min
        self.sigma_init = sigma_init

        # Flags
        self.velocity_type = velocity_type
        self.velocity_time_dependent = velocity_time_dependent
        self.activity_time_dependent = False  # no activity network
        self.has_potential = (velocity_type == "potential")
        self.growth_time_dependent = growth_time_dependent
        self.diffusion_time_dependent = diffusion_time_dependent

        # --- Velocity network ---
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

        # --- Growth network: g(t,x) unconstrained ---
        growth_kw = dict(
            in_dim=in_out_dim,
            hidden_dim=growth_hidden_dim,
            n_hiddens=growth_n_hiddens,
            activation=growth_activation,
            arch=growth_arch,
            use_layer_norm=growth_layer_norm,
            dropout=growth_dropout,
        )
        if growth_time_dependent:
            self.growth_net = TimeDependentActivityNetwork(**growth_kw)
        else:
            self.growth_net = ActivityNetwork(**growth_kw)

        # --- Diffusion network: D(t,x) = sigma_min + softplus(NN) ---
        diff_kw = dict(
            in_dim=in_out_dim,
            hidden_dim=diffusion_hidden_dim,
            n_hiddens=diffusion_n_hiddens,
            activation=diffusion_activation,
            arch=diffusion_arch,
            use_layer_norm=diffusion_layer_norm,
            dropout=diffusion_dropout,
        )
        if diffusion_time_dependent:
            self.diffusion_net = TimeDependentActivityNetwork(**diff_kw)
        else:
            self.diffusion_net = ActivityNetwork(**diff_kw)

        # Initialize diffusion_net final bias so initial output ≈ sigma_init
        self._init_diffusion_bias()

    def _init_diffusion_bias(self):
        """Set final-layer bias so softplus(bias) ≈ sigma_init - sigma_min."""
        target = max(self.sigma_init - self.sigma_min, 1e-6)
        # softplus_inv(x) = log(exp(x) - 1)
        bias_val = math.log(math.exp(target) - 1) if target < 20 else target
        net = self.diffusion_net
        # Find the last Linear layer
        if hasattr(net, 'net'):
            modules = list(net.net.modules()) if hasattr(net.net, 'modules') else []
            for m in reversed(modules):
                if isinstance(m, nn.Linear) and m.out_features == 1:
                    nn.init.constant_(m.bias, bias_val)
                    break

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def velocity(self, t, z):
        """v(t, z)."""
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
        """g(t, z) — unconstrained, can be positive or negative."""
        if self.growth_time_dependent:
            return self.growth_net(t, z)
        else:
            return self.growth_net(z)

    def diffusion_coeff(self, t, z):
        """D(t, z) = sigma_min + softplus(NN(t, z)). Always positive."""
        if self.diffusion_time_dependent:
            raw = self.diffusion_net(t, z)
        else:
            raw = self.diffusion_net(z)
        return self.sigma_min + F.softplus(raw)

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
        """Diffusion for torchsde. D(t,z) for z, 0 for lnw."""
        z = y[:, :self.in_out_dim]

        with torch.set_grad_enabled(True):
            if not z.requires_grad:
                z = z.clone().requires_grad_(True)
            d_coeff = self.diffusion_coeff(t, z)

        diffusion = torch.zeros_like(y)
        diffusion[:, :self.in_out_dim] = d_coeff.expand(-1, self.in_out_dim)
        return diffusion
