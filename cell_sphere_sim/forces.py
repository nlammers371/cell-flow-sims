from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .state import BehaviorParams


@dataclass
class ContactMetrics:
    contact_count: np.ndarray
    contact_dir_sum: np.ndarray


def contact_force_magnitude(
    d: np.ndarray,
    R_i: np.ndarray,
    R_j: np.ndarray,
    w_i: np.ndarray,
    w_j: np.ndarray,
    k_rep: float,
    alpha_dmin: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the existing regularized contact force magnitude and distance.

    This geometry-independent helper is shared by the spherical and planar
    engines. Callers are responsible for selecting pairs with ``d < R_i+R_j``
    and applying the appropriate direction vector.
    """
    sigma = R_i + R_j
    d_min = alpha_dmin * sigma
    d_eff = np.maximum(d, d_min + eps)
    rep = k_rep * ((sigma - d_eff) / (d_eff - d_min)) ** 1.5
    r_bar = 0.5 * (R_i + R_j)
    adh = (w_i * w_j / r_bar) * (sigma - d_eff)
    return rep - adh, d_eff


def compute_contact_forces_and_metrics(
    x: np.ndarray,
    behavior: BehaviorParams,
    k_rep: float,
    alpha_dmin: float,
    eps: float,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    R_E: float,
) -> tuple[np.ndarray, ContactMetrics]:
    """Compute contact forces and CIL contact metrics."""
    N = x.shape[0]
    F_contact = np.zeros_like(x)
    contact_count = np.zeros((N,), dtype=int)
    contact_dir_sum = np.zeros_like(x)

    if i_idx.size == 0:
        metrics = ContactMetrics(contact_count=contact_count, contact_dir_sum=contact_dir_sum)
        return F_contact, metrics

    R = behavior.R
    w = behavior.w

    xi = x[i_idx]
    xj = x[j_idx]
    dvec = xi - xj
    d = np.linalg.norm(dvec, axis=1)
    sigma = R[i_idx] + R[j_idx]

    contact_mask = d < sigma
    if not np.any(contact_mask):
        metrics = ContactMetrics(contact_count=contact_count, contact_dir_sum=contact_dir_sum)
        return F_contact, metrics

    i = i_idx[contact_mask]
    j = j_idx[contact_mask]
    dvec = dvec[contact_mask]
    d = d[contact_mask]
    sigma = sigma[contact_mask]

    f_mag, d_eff = contact_force_magnitude(
        d,
        R[i],
        R[j],
        w[i],
        w[j],
        k_rep,
        alpha_dmin,
        eps,
    )
    n_hat = dvec / d_eff[:, None]
    f_vec = f_mag[:, None] * n_hat

    np.add.at(F_contact, i, f_vec)
    np.add.at(F_contact, j, -f_vec)

    np.add.at(contact_count, i, 1)
    np.add.at(contact_count, j, 1)

    # Tangent direction toward neighbor for CIL metrics
    n = x / R_E
    d_ij = xj[contact_mask] - xi[contact_mask]
    n_i = n[i]
    d_t_i = d_ij - (np.sum(d_ij * n_i, axis=1)[:, None]) * n_i
    norm_i = np.linalg.norm(d_t_i, axis=1)
    valid_i = norm_i > eps
    if np.any(valid_i):
        np.add.at(contact_dir_sum, i[valid_i], d_t_i[valid_i] / norm_i[valid_i, None])

    d_ji = -d_ij
    n_j = n[j]
    d_t_j = d_ji - (np.sum(d_ji * n_j, axis=1)[:, None]) * n_j
    norm_j = np.linalg.norm(d_t_j, axis=1)
    valid_j = norm_j > eps
    if np.any(valid_j):
        np.add.at(contact_dir_sum, j[valid_j], d_t_j[valid_j] / norm_j[valid_j, None])

    metrics = ContactMetrics(contact_count=contact_count, contact_dir_sum=contact_dir_sum)
    return F_contact, metrics
