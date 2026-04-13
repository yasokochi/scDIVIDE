"""
NeuralSDE model with growth-dependent diffusion.

PDE:  d rho/dt = -div(v rho) + (sigma^2 + 2*b*delta^2)/2 * Laplacian(rho) + g rho
SDE:  dz = v(z) dt + sqrt(sigma^2 + 2*b*delta^2) dW
      d(lnw) = g dt

(beta, alpha) parametrization:
  beta(x) = NN output (unconstrained)
  b(x) = r0 * softplus(beta(x))              (birth rate)
  g(x) = r0 * softplus(beta(x)) * (1 - alpha)  (net growth)

Supports configurable velocity type (potential / free-form) and
time dependence for both velocity and activity networks.
"""

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


class NeuralSDE(nn.Module):
    """
    Unified growth-dependent diffusion SDE with (beta, alpha) parametrization.

    State: y = (z, lnw)  where z is position, lnw is log-weight.
    Compatible with torchsde.

    Parametrization:
        beta(x) = NN output (unconstrained)
        b(x) = r0 * softplus(beta(x))              (birth rate)
        g(x) = r0 * softplus(beta(x)) * (1 - alpha)  (net growth)
        diffusion = sqrt(sigma^2 + 2 * b * delta^2)

    Args:
        velocity_type: "potential" (v = -nabla phi) or "free" (v = MLP output).
        velocity_time_dependent: If True, velocity depends on (t, x).
        activity_time_dependent: If True, activity beta depends on (t, x).
        alpha: Relative death fraction in [0, 1]. Default 0.0 (pure proliferation).
        r0: Baseline turnover scale. Default 1.0.
    """

    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(
        self,
        in_out_dim,
        sigma=0.05,
        delta=0.1,
        alpha=0.0,
        r0=1.0,
        # Velocity config
        velocity_type="potential",
        velocity_time_dependent=False,
        velocity_hidden_dim=64,
        velocity_n_hiddens=4,
        velocity_activation="tanh",
        velocity_layer_norm=False,
        velocity_dropout=0.0,
        # Activity config
        activity_time_dependent=False,
        activity_hidden_dim=64,
        activity_n_hiddens=3,
        activity_activation="tanh",
        activity_layer_norm=False,
        activity_dropout=0.0,
    ):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.sigma = sigma
        self.delta = delta
        self.alpha = alpha
        self.r0 = r0

        # Flags
        self.velocity_type = velocity_type
        self.velocity_time_dependent = velocity_time_dependent
        self.activity_time_dependent = activity_time_dependent
        self.has_potential = (velocity_type == "potential")

        # --- Velocity network (4 variants) ---
        vel_kw = dict(
            in_dim=in_out_dim,
            hidden_dim=velocity_hidden_dim,
            n_hiddens=velocity_n_hiddens,
            activation=velocity_activation,
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

        # --- Activity network (2 variants) ---
        act_kw = dict(
            in_dim=in_out_dim,
            hidden_dim=activity_hidden_dim,
            n_hiddens=activity_n_hiddens,
            activation=activity_activation,
            use_layer_norm=activity_layer_norm,
            dropout=activity_dropout,
        )
        if activity_time_dependent:
            self.activity_net = TimeDependentActivityNetwork(**act_kw)
        else:
            self.activity_net = ActivityNetwork(**act_kw)

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

    def activity(self, t, z):
        """beta(t, z) — raw NN output (unconstrained)."""
        if self.activity_time_dependent:
            return self.activity_net(t, z)
        else:
            return self.activity_net(z)

    def birth_rate(self, t, z):
        """b(t,z) = r0 * softplus(beta). Birth rate."""
        beta = self.activity(t, z)
        return self.r0 * F.softplus(beta)

    def death_rate(self, t, z):
        """d(t,z) = alpha * b(t,z). Death rate."""
        return self.birth_rate(t, z) * self.alpha

    def growth(self, t, z):
        """g(t, z) = r0 * softplus(beta(t, z)) * (1 - alpha)."""
        return self.birth_rate(t, z) * (1 - self.alpha)

    def diffusion_coeff(self, t, z):
        """sqrt(sigma^2 + 2*b*delta^2) — diffusion coefficient."""
        b = self.birth_rate(t, z)
        return torch.sqrt(self.sigma**2 + 2 * b * self.delta**2)

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
        """Diffusion for torchsde. sqrt(sigma^2 + 2*b*delta^2) for z, 0 for lnw."""
        z = y[:, :self.in_out_dim]

        with torch.set_grad_enabled(True):
            if not z.requires_grad:
                z = z.clone().requires_grad_(True)
            b = self.birth_rate(t, z)

        diffusion_coeff = torch.sqrt(self.sigma**2 + 2 * b * self.delta**2)

        diffusion = torch.zeros_like(y)
        diffusion[:, :self.in_out_dim] = diffusion_coeff.expand(-1, self.in_out_dim)
        return diffusion
