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
    track_id: np.ndarray,
    parent_id: np.ndarray,
    next_track_id: int,
    t: float,
    behavior: BehaviorParams,
    R_E: float,
    split_scale: float,
    rng: np.random.Generator,
    dt: float = 1.0,
    eps: float = 1e-12,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    np.ndarray,
]:
    """Apply stochastic divisions to cells and insert daughters.

    Daughters inherit state_id and state_vars exactly. The split distance is
    split_scale * (2 * R_i), i.e., a fraction of the equal-radius sigma.
    """
    divide_mask = sample_divisions(behavior.lambda_div, dt=dt, rng=rng)
    div_idx = np.where(divide_mask)[0]
    if div_idx.size == 0:
        return x, p, state_id, state_vars, paused_until, track_id, parent_id, next_track_id, div_idx

    N = x.shape[0]
    n_div = int(div_idx.size)
    out_N = N + n_div

    x_out = np.empty((out_N, 3), dtype=x.dtype)
    p_out = np.empty((out_N, 3), dtype=p.dtype)
    state_id_out = np.empty((out_N,), dtype=state_id.dtype)
    state_vars_out = np.empty((out_N, state_vars.shape[1]), dtype=state_vars.dtype)
    paused_out = np.empty((out_N,), dtype=paused_until.dtype)
    track_id_out = np.empty((out_N,), dtype=track_id.dtype)
    parent_id_out = np.empty((out_N,), dtype=parent_id.dtype)

    x_out[:N] = x
    p_out[:N] = p
    state_id_out[:N] = state_id
    state_vars_out[:N] = state_vars
    paused_out[:N] = paused_until
    track_id_out[:N] = track_id
    parent_id_out[:N] = parent_id

    for k, i in enumerate(div_idx):
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

        old_tid = track_id[i]
        # Both daughters receive new track_ids; the original track ends at division.
        tid_a = next_track_id + 2 * k
        tid_b = next_track_id + 2 * k + 1

        x_out[i] = xi
        p_out[i] = p_i
        paused_out[i] = paused
        track_id_out[i] = tid_a
        parent_id_out[i] = old_tid

        out_i = N + k
        x_out[out_i] = xj
        p_out[out_i] = p_j
        state_id_out[out_i] = state_id[i]
        state_vars_out[out_i] = state_vars[i]
        paused_out[out_i] = paused
        track_id_out[out_i] = tid_b
        parent_id_out[out_i] = old_tid

    next_track_id_out = next_track_id + 2 * n_div
    return (
        x_out,
        p_out,
        state_id_out,
        state_vars_out,
        paused_out,
        track_id_out,
        parent_id_out,
        next_track_id_out,
        div_idx,
    )
