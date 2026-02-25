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
    state_id: np.ndarray,
    behavior: BehaviorParams,
    k_rep: float,
    alpha_dmin: float,
    eps: float,
    candidate_pairs: list[tuple[int, int]],
    R_E: float,
) -> tuple[np.ndarray, ContactMetrics]:
    """Compute contact forces and CIL contact metrics."""
    n = x / R_E
    N = x.shape[0]
    F_contact = np.zeros_like(x)
    contact_count = np.zeros((N,), dtype=int)
    contact_dir_sum = np.zeros_like(x)

    R = behavior.R
    w = behavior.w

    for i, j in candidate_pairs:
        xi = x[i]
        xj = x[j]
        dvec = xi - xj
        d = float(np.linalg.norm(dvec))
        sigma = float(R[i] + R[j])
        if d >= sigma:
            continue
        d_min = float(alpha_dmin * sigma)
        d_eff = max(d, d_min + eps)
        n_hat = dvec / d_eff

        rep = k_rep * ((sigma - d_eff) / (d_eff - d_min)) ** 1.5
        r_bar = 0.5 * (R[i] + R[j])
        adh = (w[i] * w[j] / r_bar) * (sigma - d_eff)
        f_mag = rep - adh
        f_vec = f_mag * n_hat

        F_contact[i] += f_vec
        F_contact[j] -= f_vec

        contact_count[i] += 1
        contact_count[j] += 1

        # Tangent direction toward neighbor for CIL metrics
        n_i = n[i]
        n_j = n[j]
        d_ij = xj - xi
        d_ji = xi - xj

        d_t_i = d_ij - np.dot(d_ij, n_i) * n_i
        norm_i = float(np.linalg.norm(d_t_i))
        if norm_i > eps:
            contact_dir_sum[i] += d_t_i / norm_i

        d_t_j = d_ji - np.dot(d_ji, n_j) * n_j
        norm_j = float(np.linalg.norm(d_t_j))
        if norm_j > eps:
            contact_dir_sum[j] += d_t_j / norm_j

    metrics = ContactMetrics(contact_count=contact_count, contact_dir_sum=contact_dir_sum)
    return F_contact, metrics
