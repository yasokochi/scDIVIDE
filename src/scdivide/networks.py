"""
Tunable neural network architectures for NeuralSDE.

Supports MLP with configurable activation and weight initialization.
"""

import torch
import torch.nn as nn


ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "leakyrelu": nn.LeakyReLU,
    "softplus": nn.Softplus,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


def _get_activation(name: str) -> nn.Module:
    key = name.lower()
    if key not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}. Available: {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[key]()


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Plain MLP: Linear -> Act -> ... -> Linear."""

    def __init__(self, in_dim, out_dim, hidden_dim=64, n_hiddens=4,
                 activation="tanh"):
        super().__init__()
        act_fn = _get_activation(activation)

        layers = []
        dims = [in_dim] + [hidden_dim] * n_hiddens + [out_dim]

        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act_fn)

        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Network classes for the SDE model
# ---------------------------------------------------------------------------

class PotentialNetwork(nn.Module):
    """phi(x) -> scalar. gradient() returns nabla phi(x)."""

    def __init__(self, in_dim, hidden_dim=64, n_hiddens=4,
                 activation="tanh"):
        super().__init__()
        self.in_dim = in_dim
        self.net = MLP(in_dim, 1, hidden_dim, n_hiddens,
                       activation=activation)

    def forward(self, x):
        return self.net(x)

    def gradient(self, x):
        """nabla phi(x)"""
        if not x.requires_grad:
            x = x.requires_grad_(True)
        phi = self.forward(x)
        grad_phi = torch.autograd.grad(
            outputs=phi,
            inputs=x,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
            retain_graph=True,
        )[0]
        return grad_phi


class ActivityNetwork(nn.Module):
    """beta(x) -> scalar (unconstrained, no activation on output)."""

    def __init__(self, in_dim, hidden_dim=64, n_hiddens=3,
                 activation="tanh"):
        super().__init__()
        self.net = MLP(in_dim, 1, hidden_dim, n_hiddens,
                       activation=activation)

    def forward(self, x):
        return self.net(x)


class TimeDependentActivityNetwork(nn.Module):
    """beta(t, x) -> scalar (unconstrained, no activation on output)."""

    def __init__(self, in_dim, hidden_dim=64, n_hiddens=3,
                 activation="tanh"):
        super().__init__()
        self.in_dim = in_dim
        self.net = MLP(in_dim + 1, 1, hidden_dim, n_hiddens,
                       activation=activation)

    def forward(self, t, x):
        t_tensor = _broadcast_t(t, x)
        tx = torch.cat([t_tensor, x], dim=1)
        return self.net(tx)


def _broadcast_t(t, x):
    """Broadcast scalar/0-dim/1-dim t to (batch, 1) matching x."""
    batch_size = x.shape[0]
    if isinstance(t, (int, float)):
        return torch.full((batch_size, 1), t, device=x.device, dtype=x.dtype)
    if t.dim() == 0:
        return t.expand(batch_size, 1).reshape(batch_size, 1)
    return t.reshape(batch_size, 1)


class TimeDependentPotentialNetwork(nn.Module):
    """phi(t, x) -> scalar. gradient(t, x) returns nabla_x phi(t, x)."""

    def __init__(self, in_dim, hidden_dim=64, n_hiddens=4,
                 activation="tanh"):
        super().__init__()
        self.in_dim = in_dim
        self.net = MLP(in_dim + 1, 1, hidden_dim, n_hiddens,
                       activation=activation)

    def forward(self, t, x):
        t_tensor = _broadcast_t(t, x)
        tx = torch.cat([t_tensor, x], dim=1)
        return self.net(tx)

    def gradient(self, t, x):
        """nabla_x phi(t, x) — gradient w.r.t. x only."""
        if not x.requires_grad:
            x = x.requires_grad_(True)
        phi = self.forward(t, x)
        grad_phi = torch.autograd.grad(
            outputs=phi,
            inputs=x,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
            retain_graph=True,
        )[0]
        return grad_phi


class VelocityNetwork(nn.Module):
    """v(x) -> d-dim vector (free-form MLP)."""

    def __init__(self, in_dim, hidden_dim=64, n_hiddens=4,
                 activation="tanh"):
        super().__init__()
        self.net = MLP(in_dim, in_dim, hidden_dim, n_hiddens,
                       activation=activation)

    def forward(self, x):
        return self.net(x)


class TimeDependentVelocityNetwork(nn.Module):
    """v(t, x) -> d-dim vector (free-form MLP)."""

    def __init__(self, in_dim, hidden_dim=64, n_hiddens=4,
                 activation="tanh"):
        super().__init__()
        self.in_dim = in_dim
        self.net = MLP(in_dim + 1, in_dim, hidden_dim, n_hiddens,
                       activation=activation)

    def forward(self, t, x):
        t_tensor = _broadcast_t(t, x)
        tx = torch.cat([t_tensor, x], dim=1)
        return self.net(tx)


# ---------------------------------------------------------------------------
# Weight initialization
# ---------------------------------------------------------------------------

INIT_METHODS = {
    "xavier_uniform": nn.init.xavier_uniform_,
    "xavier_normal": nn.init.xavier_normal_,
    "kaiming_uniform": nn.init.kaiming_uniform_,
    "kaiming_normal": nn.init.kaiming_normal_,
    "orthogonal": nn.init.orthogonal_,
}


def initialize_weights(module, method="xavier_uniform"):
    """Apply weight initialization to a module."""
    init_fn = INIT_METHODS.get(method, nn.init.xavier_uniform_)

    def _init(m):
        if hasattr(m, "weight") and m.weight.dim() > 1:
            init_fn(m.weight.data)

    module.apply(_init)
