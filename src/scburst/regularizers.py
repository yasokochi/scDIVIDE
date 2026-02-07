"""
Regularization functions for BranchingSDE training.

Each function takes the model (func), relevant inputs, and keyword args,
and returns a scalar loss tensor.

All functions use the unified API: func.velocity(t, x), func.activity(t, x),
func.growth(t, x), func.birth_rate(t, x).
"""

import torch


def activity_smoothness(func, t, x, **kw):
    """||nabla_x beta||^2  averaged over samples.

    Penalises sharp spatial variation in the activity field.
    """
    x = x.detach().requires_grad_(True)
    beta = func.activity(t, x)

    grad_beta = torch.autograd.grad(
        outputs=beta,
        inputs=x,
        grad_outputs=torch.ones_like(beta),
        create_graph=True,
    )[0]
    return (grad_beta ** 2).sum(dim=1).mean()


def mass_conservation(func, t, x, lnw, **kw):
    """(sum w_i g_i / sum w_i)^2.

    Encourages E_w[g] = 0 (mass conservation on average).
    """
    w = torch.exp(torch.clamp(lnw, min=-20, max=20)).squeeze()
    g = func.growth(t, x).squeeze()

    weighted_mean_g = (w * g).sum() / (w.sum() + 1e-10)
    return weighted_mean_g ** 2


def velocity_ratio(func, t, x, target=1.0, **kw):
    """(||v|| / sqrt(sigma^2 + b*delta^2) - target)^2  averaged over samples.

    Encourages a fixed ratio between drift and diffusion magnitudes.
    Uses birth rate b (not net growth g) for diffusion coefficient.
    """
    x_req = x.detach().requires_grad_(True)
    v = func.velocity(t, x_req)
    b = func.birth_rate(t, x_req).squeeze()

    v_norm = torch.norm(v, dim=1)
    diff_coeff = torch.sqrt(func.sigma ** 2 + b * func.delta ** 2)
    ratio = v_norm / (diff_coeff + 1e-10)
    return ((ratio - target) ** 2).mean()


def potential_hessian(func, t, x, **kw):
    """||nabla^2 phi||_F^2  averaged over samples.

    Penalises high curvature of the potential (encourages smooth velocity).
    Only active for velocity_type='potential'; returns 0 for 'free'.
    """
    if not getattr(func, 'has_potential', True):
        return torch.tensor(0.0, device=x.device)

    x = x.detach().requires_grad_(True)
    phi = func.potential(t, x)

    grad_phi = torch.autograd.grad(
        outputs=phi,
        inputs=x,
        grad_outputs=torch.ones_like(phi),
        create_graph=True,
    )[0]

    hess_norm_sq = 0.0
    for i in range(x.shape[1]):
        grad2 = torch.autograd.grad(
            outputs=grad_phi[:, i].sum(),
            inputs=x,
            create_graph=True,
        )[0]
        hess_norm_sq = hess_norm_sq + (grad2 ** 2).sum(dim=1)

    return hess_norm_sq.mean()


def activity_temporal_smoothness(func, t1, t2, x, **kw):
    """||beta(t2,x) - beta(t1,x)||^2 / (t2-t1)^2  averaged over samples.

    Only meaningful when activity is time-dependent; returns 0 otherwise.
    """
    if not getattr(func, 'activity_time_dependent', False):
        return torch.tensor(0.0, device=x.device)

    beta1 = func.activity(t1, x)
    beta2 = func.activity(t2, x)
    dt = float(t2 - t1) if not isinstance(t2, float) else (t2 - t1)
    return ((beta2 - beta1) ** 2).mean() / (dt ** 2 + 1e-10)


def velocity_spatial_smoothness(func, t, x, **kw):
    """||nabla_x v||^2_F  averaged over samples.

    Penalises sharp spatial variation in the velocity field.
    Useful for free-form velocity (no potential curvature regularizer).
    """
    x = x.detach().requires_grad_(True)
    v = func.velocity(t, x)

    jac_norm_sq = 0.0
    for i in range(v.shape[1]):
        grad_vi = torch.autograd.grad(
            outputs=v[:, i].sum(),
            inputs=x,
            create_graph=True,
        )[0]
        jac_norm_sq = jac_norm_sq + (grad_vi ** 2).sum(dim=1)

    return jac_norm_sq.mean()


def velocity_temporal_smoothness(func, t1, t2, x, **kw):
    """||v(t2,x) - v(t1,x)||^2 / (t2-t1)^2  averaged over samples.

    Only meaningful when velocity is time-dependent; returns 0 otherwise.
    """
    if not getattr(func, 'velocity_time_dependent', False):
        return torch.tensor(0.0, device=x.device)

    x = x.detach().requires_grad_(True)
    v1 = func.velocity(t1, x)
    v2 = func.velocity(t2, x)
    dt = float(t2 - t1) if not isinstance(t2, float) else (t2 - t1)
    return ((v2 - v1) ** 2).sum(dim=1).mean() / (dt ** 2 + 1e-10)


REGISTRY = {
    "activity_smoothness": activity_smoothness,
    "mass_conservation": mass_conservation,
    "velocity_ratio": velocity_ratio,
    "potential_hessian": potential_hessian,
    "activity_temporal_smoothness": activity_temporal_smoothness,
    "velocity_spatial_smoothness": velocity_spatial_smoothness,
    "velocity_temporal_smoothness": velocity_temporal_smoothness,
}
