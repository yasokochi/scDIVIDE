"""
Sinkhorn divergence computation using POT (Python Optimal Transport).

Differentiable via POT's torch backend.
"""

import torch


def compute_cost_scale(y):
    """Median pairwise squared Euclidean distance.

    Args:
        y: (n, d) samples.

    Returns:
        Scalar tensor (median of upper-triangular pairwise sq-distances).
    """
    sq_dists = torch.cdist(y, y, p=2) ** 2
    n = y.shape[0]
    triu_indices = torch.triu_indices(n, n, offset=1, device=y.device)
    pairwise_dists = sq_dists[triu_indices[0], triu_indices[1]]
    return torch.median(pairwise_dists)


def sinkhorn_divergence(x, y, epsilon=0.05, n_iter=500,
                        a=None, b=None, cost_scale=None):
    """Differentiable debiased Sinkhorn divergence.

    S(x, y) = OT_eps(x,y) - 0.5*OT_eps(x,x) - 0.5*OT_eps(y,y)

    Args:
        x: (n, d) source samples.
        y: (m, d) target samples.
        epsilon: entropic regularization.
        n_iter: max Sinkhorn iterations.
        a: (n,) source weights (None -> uniform).
        b: (m,) target weights (None -> uniform).
        cost_scale: cost normalization (None -> auto from y).

    Returns:
        Scalar tensor (differentiable).
    """
    import ot

    n, m = x.shape[0], y.shape[0]

    if a is None:
        a = torch.ones(n, device=x.device, dtype=x.dtype) / n
    else:
        a = torch.clamp(a, min=1e-10)
        a = a / a.sum()

    if b is None:
        b = torch.ones(m, device=y.device, dtype=y.dtype) / m
    else:
        b = torch.clamp(b, min=1e-10)
        b = b / b.sum()

    if cost_scale is None:
        cost_scale = compute_cost_scale(y)
    cost_scale = cost_scale + 1e-8

    C_xy = torch.cdist(x, y, p=2) ** 2 / cost_scale
    C_xx = torch.cdist(x, x, p=2) ** 2 / cost_scale
    C_yy = torch.cdist(y, y, p=2) ** 2 / cost_scale

    ot_xy = ot.sinkhorn2(a, b, C_xy, epsilon, numItermax=n_iter, method="sinkhorn_log")
    ot_xx = ot.sinkhorn2(a, a, C_xx, epsilon, numItermax=n_iter, method="sinkhorn_log")
    ot_yy = ot.sinkhorn2(b, b, C_yy, epsilon, numItermax=n_iter, method="sinkhorn_log")

    return ot_xy - 0.5 * ot_xx - 0.5 * ot_yy
