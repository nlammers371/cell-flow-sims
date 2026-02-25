from __future__ import annotations

import numpy as np


def parallel_transport(p: np.ndarray, n: np.ndarray, n_new: np.ndarray, eps: float) -> np.ndarray:
    """Parallel transport tangent vectors from n to n_new."""
    a = np.cross(n, n_new)
    a_norm = np.linalg.norm(a, axis=1)
    dot = np.einsum("ij,ij->i", n, n_new)
    theta = np.arctan2(a_norm, dot)

    p_tr = p.copy()
    mask = a_norm > eps
    if np.any(mask):
        u = a[mask] / a_norm[mask][:, None]
        p_sel = p[mask]
        theta_sel = theta[mask]
        cos_t = np.cos(theta_sel)[:, None]
        sin_t = np.sin(theta_sel)[:, None]
        u_cross_p = np.cross(u, p_sel)
        u_dot_p = np.einsum("ij,ij->i", u, p_sel)[:, None]
        p_rot = p_sel * cos_t + u_cross_p * sin_t + u * u_dot_p * (1.0 - cos_t)
        p_tr[mask] = p_rot

    # Reproject to tangent at n_new and normalize
    p_tr = p_tr - (np.einsum("ij,ij->i", p_tr, n_new)[:, None]) * n_new
    norms = np.linalg.norm(p_tr, axis=1)
    norms = np.where(norms > eps, norms, 1.0)
    p_tr = p_tr / norms[:, None]
    return p_tr


def random_tangent_rotation(p: np.ndarray, n: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Rotate tangent vectors around local normal by angle delta."""
    cos_d = np.cos(delta)[:, None]
    sin_d = np.sin(delta)[:, None]
    n_cross_p = np.cross(n, p)
    p_rot = p * cos_d + n_cross_p * sin_d
    return p_rot


def cil_target_flee(
    contact_dir_sum: np.ndarray, n_new: np.ndarray, eps: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute CIL flee target vectors from contact direction sums."""
    norms = np.linalg.norm(contact_dir_sum, axis=1)
    has_target = norms > eps
    p_flee = np.zeros_like(contact_dir_sum)
    if np.any(has_target):
        p = -contact_dir_sum[has_target] / norms[has_target][:, None]
        p = p - (np.einsum("ij,ij->i", p, n_new[has_target])[:, None]) * n_new[has_target]
        p_norm = np.linalg.norm(p, axis=1)
        p_norm = np.where(p_norm > eps, p_norm, 1.0)
        p_flee[has_target] = p / p_norm[:, None]
    return has_target, p_flee
