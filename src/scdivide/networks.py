"""
Tunable neural network architectures for NeuralSDE.

Supports MLP and ResNet (residual block) architectures with configurable
activation, layer normalization, dropout, and weight initialization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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

class ResBlock(nn.Module):
    """Residual block: identity + 2-layer MLP (VarRUOT style)."""

    def __init__(self, dim, activation="gelu", use_layer_norm=True):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.activation = _get_activation(activation)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        if self.use_layer_norm:
            out = self.norm1(out)
        out = self.activation(out)
        out = self.fc2(out)
        if self.use_layer_norm:
            out = self.norm2(out)
        out = self.activation(out)
        return identity + out


class MLP(nn.Module):
    """Plain MLP: Linear -> Act -> ... -> Linear."""

    def __init__(self, in_dim, out_dim, hidden_dim=64, n_hiddens=4,
                 activation="tanh", use_layer_norm=False, dropout=0.0):
        super().__init__()
        act_fn = _get_activation(activation)

        layers = []
        dims = [in_dim] + [hidden_dim] * n_hiddens + [out_dim]

        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(act_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResNet(nn.Module):
    """ResBlock-based network: Linear -> ResBlocks -> Linear."""

    def __init__(self, in_dim, out_dim, hidden_dim=256, n_blocks=2,
                 activation="gelu", use_layer_norm=True, dropout=0.0):
        super().__init__()
        self.input_layer = nn.Linear(in_dim, hidden_dim)
        self.activation = _get_activation(activation)
        self.res_blocks = nn.Sequential(
            *[ResBlock(hidden_dim, activation=activation, use_layer_norm=use_layer_norm)
              for _ in range(n_blocks)]
        )
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        h = self.activation(self.input_layer(x))
        if self.dropout is not None:
            h = self.dropout(h)
        h = self.res_blocks(h)
        return self.output_layer(h)


# ---------------------------------------------------------------------------
# Network classes for the SDE model
# ---------------------------------------------------------------------------

class PotentialNetwork(nn.Module):
    """phi(x) -> scalar. gradient() returns nabla phi(x)."""

    def __init__(self, in_dim, hidden_dim=64, n_hiddens=4,
                 activation="tanh", arch="mlp",
                 use_layer_norm=False, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        if arch == "resnet":
            self.net = ResNet(in_dim, 1, hidden_dim, n_hiddens,
                              activation=activation,
                              use_layer_norm=use_layer_norm,
                              dropout=dropout)
        else:
            self.net = MLP(in_dim, 1, hidden_dim, n_hiddens,
                           activation=activation,
                           use_layer_norm=use_layer_norm,
                           dropout=dropout)

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
                 activation="tanh", arch="mlp",
                 use_layer_norm=False, dropout=0.0):
        super().__init__()
        if arch == "resnet":
            self.net = ResNet(in_dim, 1, hidden_dim, n_hiddens,
                              activation=activation,
                              use_layer_norm=use_layer_norm,
                              dropout=dropout)
        else:
            self.net = MLP(in_dim, 1, hidden_dim, n_hiddens,
                           activation=activation,
                           use_layer_norm=use_layer_norm,
                           dropout=dropout)

    def forward(self, x):
        return self.net(x)


class TimeDependentActivityNetwork(nn.Module):
    """beta(t, x) -> scalar (unconstrained, no activation on output)."""

    def __init__(self, in_dim, hidden_dim=64, n_hiddens=3,
                 activation="tanh", arch="mlp",
                 use_layer_norm=False, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        if arch == "resnet":
            self.net = ResNet(in_dim + 1, 1, hidden_dim, n_hiddens,
                              activation=activation,
                              use_layer_norm=use_layer_norm,
                              dropout=dropout)
        else:
            self.net = MLP(in_dim + 1, 1, hidden_dim, n_hiddens,
                           activation=activation,
                           use_layer_norm=use_layer_norm,
                           dropout=dropout)

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
                 activation="tanh", arch="mlp",
                 use_layer_norm=False, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        if arch == "resnet":
            self.net = ResNet(in_dim + 1, 1, hidden_dim, n_hiddens,
                              activation=activation,
                              use_layer_norm=use_layer_norm,
                              dropout=dropout)
        else:
            self.net = MLP(in_dim + 1, 1, hidden_dim, n_hiddens,
                           activation=activation,
                           use_layer_norm=use_layer_norm,
                           dropout=dropout)

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
                 activation="tanh", arch="mlp",
                 use_layer_norm=False, dropout=0.0):
        super().__init__()
        if arch == "resnet":
            self.net = ResNet(in_dim, in_dim, hidden_dim, n_hiddens,
                              activation=activation,
                              use_layer_norm=use_layer_norm,
                              dropout=dropout)
        else:
            self.net = MLP(in_dim, in_dim, hidden_dim, n_hiddens,
                           activation=activation,
                           use_layer_norm=use_layer_norm,
                           dropout=dropout)

    def forward(self, x):
        return self.net(x)


class TimeDependentVelocityNetwork(nn.Module):
    """v(t, x) -> d-dim vector (free-form MLP)."""

    def __init__(self, in_dim, hidden_dim=64, n_hiddens=4,
                 activation="tanh", arch="mlp",
                 use_layer_norm=False, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        if arch == "resnet":
            self.net = ResNet(in_dim + 1, in_dim, hidden_dim, n_hiddens,
                              activation=activation,
                              use_layer_norm=use_layer_norm,
                              dropout=dropout)
        else:
            self.net = MLP(in_dim + 1, in_dim, hidden_dim, n_hiddens,
                           activation=activation,
                           use_layer_norm=use_layer_norm,
                           dropout=dropout)

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
