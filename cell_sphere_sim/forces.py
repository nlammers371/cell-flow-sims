from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .state import BehaviorParams


@dataclass
class ContactMetrics:
    contact_count: np.ndarray
    contact_dir_sum: np.ndarray


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

    d_min = alpha_dmin * sigma
    d_eff = np.maximum(d, d_min + eps)
    n_hat = dvec / d_eff[:, None]

    rep = k_rep * ((sigma - d_eff) / (d_eff - d_min)) ** 1.5
    r_bar = 0.5 * (R[i] + R[j])
    adh = (w[i] * w[j] / r_bar) * (sigma - d_eff)
    f_mag = rep - adh
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
