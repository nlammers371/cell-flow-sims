from __future__ import annotations

import numpy as np

from .state import BehaviorParams


def sample_divisions(lambda_div: np.ndarray, dt: float, rng: np.random.Generator) -> np.ndarray:
    """Sample division events as a Bernoulli process per cell."""
    p = 1.0 - np.exp(-lambda_div * dt)
    return rng.random(size=lambda_div.shape[0]) < p


def _random_tangent_unit(n: np.ndarray, rng: np.random.Generator, eps: float) -> np.ndarray:
    g = rng.normal(size=3)
    u = g - np.dot(g, n) * n
    norm = np.linalg.norm(u)
    if norm <= eps:
        u = np.array([1.0, 0.0, 0.0], dtype=float) - n[0] * n
        norm = np.linalg.norm(u)
    return u / norm


def apply_divisions(
    x: np.ndarray,
    p: np.ndarray,
    state_id: np.ndarray,
    state_vars: np.ndarray,
    paused_until: np.ndarray,
    t: float,
    behavior: BehaviorParams,
    R_E: float,
    split_scale: float,
    rng: np.random.Generator,
    dt: float = 1.0,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply stochastic divisions to cells and insert daughters.

    Daughters inherit state_id and state_vars exactly. The split distance is
    split_scale * (2 * R_i), i.e., a fraction of the equal-radius sigma.
    """
    divide_mask = sample_divisions(behavior.lambda_div, dt=dt, rng=rng)
    if not np.any(divide_mask):
        return x, p, state_id, state_vars, paused_until

    N = x.shape[0]
    new_x = [x]
    new_p = [p]
    new_state_id = [state_id]
    new_state_vars = [state_vars]
    new_paused = [paused_until]

    for i in np.where(divide_mask)[0]:
        n = x[i] / R_E
        axis = _random_tangent_unit(n, rng, eps)
        s = split_scale * (2.0 * behavior.R[i])
        xi = x[i] + s * axis
        xj = x[i] - s * axis
        xi = R_E * xi / np.linalg.norm(xi)
        xj = R_E * xj / np.linalg.norm(xj)

        n_i = xi / R_E
        n_j = xj / R_E
        p_i = p[i] - np.dot(p[i], n_i) * n_i
        p_j = p[i] - np.dot(p[i], n_j) * n_j
        p_i = p_i / max(np.linalg.norm(p_i), eps)
        p_j = p_j / max(np.linalg.norm(p_j), eps)

        tau = behavior.tau_div[i]
        paused = max(paused_until[i], t + tau)

        # Update mother slot in-place, append daughter
        x[i] = xi
        p[i] = p_i
        paused_until[i] = paused

        new_x.append(xj[None, :])
        new_p.append(p_j[None, :])
        new_state_id.append(state_id[i][None])
        new_state_vars.append(state_vars[i][None, :])
        new_paused.append(np.array([paused], dtype=paused_until.dtype))

    x_out = np.vstack(new_x)
    p_out = np.vstack(new_p)
    state_id_out = np.concatenate(new_state_id)
    state_vars_out = np.vstack(new_state_vars)
    paused_out = np.concatenate(new_paused)
    return x_out, p_out, state_id_out, state_vars_out, paused_out
