from __future__ import annotations

import numpy as np

from ..forces import ContactMetrics, contact_force_magnitude
from ..state import BehaviorParams
from .neighbors import minimum_image_displacement, validate_box_size


def compute_planar_contact_forces_and_metrics(
    x: np.ndarray,
    behavior: BehaviorParams,
    k_rep: float,
    alpha_dmin: float,
    eps: float,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    box_size: np.ndarray | tuple[float, float],
) -> tuple[np.ndarray, ContactMetrics]:
    """Compute periodic planar contact forces and CIL direction sums."""
    points = np.asarray(x, dtype=float)
    box = validate_box_size(box_size)
    n_cells = points.shape[0]
    force = np.zeros_like(points)
    contact_count = np.zeros((n_cells,), dtype=int)
    contact_dir_sum = np.zeros_like(points)
    metrics = ContactMetrics(contact_count, contact_dir_sum)

    if i_idx.size == 0:
        return force, metrics

    radii = behavior.R
    dvec = minimum_image_displacement(points[i_idx], points[j_idx], box)
    distance = np.linalg.norm(dvec, axis=1)
    sigma = radii[i_idx] + radii[j_idx]
    contact_mask = distance < sigma
    if not np.any(contact_mask):
        return force, metrics

    i = i_idx[contact_mask]
    j = j_idx[contact_mask]
    dvec = dvec[contact_mask]
    distance = distance[contact_mask]

    f_mag, d_eff = contact_force_magnitude(
        distance,
        radii[i],
        radii[j],
        behavior.w[i],
        behavior.w[j],
        k_rep,
        alpha_dmin,
        eps,
    )
    f_vec = f_mag[:, None] * (dvec / d_eff[:, None])
    np.add.at(force, i, f_vec)
    np.add.at(force, j, -f_vec)

    np.add.at(contact_count, i, 1)
    np.add.at(contact_count, j, 1)

    valid = distance > eps
    if np.any(valid):
        direction_j_to_i = dvec[valid] / distance[valid, None]
        np.add.at(contact_dir_sum, i[valid], -direction_j_to_i)
        np.add.at(contact_dir_sum, j[valid], direction_j_to_i)

    return force, metrics
