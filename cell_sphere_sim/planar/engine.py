from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import warnings

import numpy as np

from ..forces import ContactMetrics
from ..neighbors import interaction_radius
from ..state import BehaviorParams, StateTable, lookup_behavior
from .forces import compute_planar_contact_forces_and_metrics
from .metrics import largest_cluster_fraction, nematic_order_2d, polarization_magnitude
from .neighbors import candidate_pairs_periodic, minimum_image_displacement, validate_box_size


@dataclass
class PlanarParams:
    box_size: tuple[float, float]
    gamma_s: float
    k_rep: float
    alpha_dmin: float
    eps: float
    dt: float | None
    record_interval: int = 1
    neighbor_radius_buffer: float = 0.1
    division_enabled: bool = False


PlanarCellUpdateFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray, ContactMetrics, float, np.random.Generator, StateTable],
    tuple[np.ndarray, np.ndarray, BehaviorParams],
]


def default_planar_cell_update(
    state_id: np.ndarray,
    state_vars: np.ndarray,
    fields: np.ndarray,
    contact_metrics: ContactMetrics,
    dt: float,
    rng: np.random.Generator,
    state_table: StateTable,
) -> tuple[np.ndarray, np.ndarray, BehaviorParams]:
    """Identity state update and state-table behavior lookup."""
    return state_id, state_vars, lookup_behavior(state_id, state_table)


def _default_dt(state_table: StateTable, gamma_s: float, eps: float, eta: float = 0.02) -> float:
    sigma_min = 2.0 * float(np.min(state_table.R))
    v_m_max = float(np.max(state_table.Fm)) / gamma_s
    return eta * sigma_min / (v_m_max + eps)


def _warn_dt(dt: float, state_table: StateTable, gamma_s: float, eps: float) -> None:
    sigma_min = 2.0 * float(np.min(state_table.R))
    motile_step = float(np.max(state_table.Fm)) * dt / gamma_s
    if motile_step > 0.1 * sigma_min:
        warnings.warn("dt may be too large for stability", RuntimeWarning)
    if motile_step < 0.002 * sigma_min:
        warnings.warn("dt may be too small for efficiency", RuntimeWarning)


def _validate_state_table(state_table: StateTable) -> None:
    arrays = [np.asarray(getattr(state_table, name)) for name in state_table.__dataclass_fields__]
    if not arrays or arrays[0].ndim != 1 or arrays[0].size == 0:
        raise ValueError("state table arrays must be non-empty and one-dimensional")
    if any(array.ndim != 1 or array.shape != arrays[0].shape for array in arrays):
        raise ValueError("all state table arrays must be one-dimensional and equally sized")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("state table arrays must contain only finite values")
    if np.any(np.asarray(state_table.R) <= 0.0):
        raise ValueError("state table radii must be positive")
    for name in ("Fm", "Dr", "fcil", "w", "lambda_div", "tau_div"):
        if np.any(np.asarray(getattr(state_table, name)) < 0.0):
            raise ValueError(f"state table {name} values must be non-negative")


class PlanarSimulationEngine:
    """Standalone overdamped cell simulation in a periodic rectangular plane."""

    def __init__(
        self,
        x: np.ndarray,
        p: np.ndarray,
        state_id: np.ndarray,
        state_vars: np.ndarray,
        state_table: StateTable,
        params: PlanarParams,
        cell_update: PlanarCellUpdateFn | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        _validate_state_table(state_table)
        self.box_size = validate_box_size(params.box_size)
        self.params = params
        self._validate_params()
        if params.division_enabled:
            raise NotImplementedError("cell division is not supported by PlanarSimulationEngine")

        points = np.asarray(x, dtype=float)
        polarity = np.asarray(p, dtype=float)
        states = np.asarray(state_id, dtype=np.int32)
        variables = np.asarray(state_vars, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("x must have shape (N, 2)")
        if polarity.shape != points.shape:
            raise ValueError("p must have the same (N, 2) shape as x")
        if states.shape != (points.shape[0],):
            raise ValueError("state_id must have shape (N,)")
        if variables.ndim != 2 or variables.shape[0] != points.shape[0]:
            raise ValueError("state_vars must have shape (N, M)")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(polarity)):
            raise ValueError("x and p must contain only finite values")
        if states.size and (np.any(states < 0) or np.any(states >= len(state_table.R))):
            raise ValueError("state_id contains an index outside the state table")
        if not np.all(np.isfinite(variables)):
            raise ValueError("state_vars must contain only finite values")

        p_norm = np.linalg.norm(polarity, axis=1)
        if np.any(p_norm <= params.eps):
            raise ValueError("every polarity vector must have non-zero length")

        self.x = np.mod(points, self.box_size)
        self.x_unwrapped = points.copy()
        self.x_unwrapped_initial = points.copy()
        self.p = polarity / p_norm[:, None]
        self.state_id = states.copy()
        self.state_vars = variables.copy()
        self.state_table = state_table
        self.cell_update = cell_update or default_planar_cell_update
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.track_id = np.arange(points.shape[0], dtype=np.int64)
        self.v = np.zeros_like(points)
        self.contact_metrics = ContactMetrics(
            contact_count=np.zeros(points.shape[0], dtype=int),
            contact_dir_sum=np.zeros_like(points),
        )
        if self.params.dt is None:
            self.params.dt = _default_dt(state_table, params.gamma_s, params.eps)
        _warn_dt(float(self.params.dt), state_table, params.gamma_s, params.eps)

    def _validate_params(self) -> None:
        params = self.params
        if not np.isfinite(params.gamma_s) or params.gamma_s <= 0.0:
            raise ValueError("gamma_s must be finite and positive")
        if not np.isfinite(params.k_rep) or params.k_rep < 0.0:
            raise ValueError("k_rep must be finite and non-negative")
        if not np.isfinite(params.alpha_dmin) or not 0.0 <= params.alpha_dmin < 1.0:
            raise ValueError("alpha_dmin must be in [0, 1)")
        if not np.isfinite(params.eps) or params.eps <= 0.0:
            raise ValueError("eps must be finite and positive")
        if params.dt is not None and (not np.isfinite(params.dt) or params.dt <= 0.0):
            raise ValueError("dt must be None or finite and positive")
        if params.record_interval <= 0:
            raise ValueError("record_interval must be positive")
        if not np.isfinite(params.neighbor_radius_buffer) or params.neighbor_radius_buffer < 0.0:
            raise ValueError("neighbor_radius_buffer must be finite and non-negative")

    def step(self, t: float) -> dict[str, float | int]:
        params = self.params
        dt = float(params.dt)
        fields = np.zeros((self.x.shape[0], 0), dtype=float)
        self.state_id, self.state_vars, behavior = self.cell_update(
            self.state_id,
            self.state_vars,
            fields,
            self.contact_metrics,
            dt,
            self.rng,
            self.state_table,
        )
        self.state_id = np.asarray(self.state_id, dtype=np.int32)
        self.state_vars = np.asarray(self.state_vars, dtype=float)

        r_query = interaction_radius(behavior.R, params.neighbor_radius_buffer)
        i_idx, j_idx = candidate_pairs_periodic(self.x, r_query, self.box_size)
        force, contact_metrics = compute_planar_contact_forces_and_metrics(
            self.x,
            behavior,
            params.k_rep,
            params.alpha_dmin,
            params.eps,
            i_idx,
            j_idx,
            self.box_size,
        )
        self.contact_metrics = contact_metrics

        pair_dvec = minimum_image_displacement(self.x[i_idx], self.x[j_idx], self.box_size)
        pair_distances = np.linalg.norm(pair_dvec, axis=1)
        contact_mask = pair_distances < (behavior.R[i_idx] + behavior.R[j_idx])
        contact_i = i_idx[contact_mask]
        contact_j = j_idx[contact_mask]
        contact_distances = pair_distances[contact_mask]

        self.v = (behavior.Fm[:, None] * self.p + force) / params.gamma_s
        displacement = dt * self.v
        self.x_unwrapped = self.x_unwrapped + displacement
        self.x = np.mod(self.x + displacement, self.box_size)

        target_norm = np.linalg.norm(contact_metrics.contact_dir_sum, axis=1)
        has_target = target_norm > params.eps
        p_flee = np.zeros_like(self.p)
        p_flee[has_target] = (
            -contact_metrics.contact_dir_sum[has_target] / target_norm[has_target, None]
        )
        p_det = self.p.copy()
        if np.any(has_target):
            relax = np.exp(-behavior.fcil * dt)
            p_det[has_target] = p_flee[has_target] + relax[has_target, None] * (
                self.p[has_target] - p_flee[has_target]
            )

        delta = self.rng.normal(scale=np.sqrt(2.0 * behavior.Dr * dt))
        cos_delta = np.cos(delta)
        sin_delta = np.sin(delta)
        p_new = np.column_stack(
            (
                cos_delta * p_det[:, 0] - sin_delta * p_det[:, 1],
                sin_delta * p_det[:, 0] + cos_delta * p_det[:, 1],
            )
        )
        p_norm = np.linalg.norm(p_new, axis=1)
        p_norm = np.where(p_norm > params.eps, p_norm, 1.0)
        self.p = p_new / p_norm[:, None]

        speed = np.linalg.norm(self.v, axis=1)
        squared_displacement = np.sum((self.x_unwrapped - self.x_unwrapped_initial) ** 2, axis=1)

        return {
            "n_cells": int(self.x.shape[0]),
            "mean_speed": float(np.mean(speed)) if speed.size else 0.0,
            "mean_contacts": (
                float(np.mean(contact_metrics.contact_count)) if contact_metrics.contact_count.size else 0.0
            ),
            "n_candidates": int(i_idx.size),
            "n_contact_pairs": int(contact_i.size),
            "min_d_contact": (
                float(np.min(contact_distances)) if contact_distances.size else float("nan")
            ),
            "polarization": polarization_magnitude(self.p),
            "nematic_order": nematic_order_2d(self.p),
            "largest_cluster_fraction": largest_cluster_fraction(
                self.x.shape[0], contact_i, contact_j
            ),
            "mean_squared_displacement": (
                float(np.mean(squared_displacement)) if squared_displacement.size else 0.0
            ),
        }

    def run(
        self,
        n_steps: int,
        t0: float = 0.0,
        store=None,
        callback=None,
        show_progress: bool = False,
    ) -> list[dict[str, float | int]]:
        """Run steps and return their diagnostic dictionaries."""
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        t = float(t0)
        step_iter = range(n_steps)
        if show_progress:
            from tqdm import tqdm

            step_iter = tqdm(step_iter, desc="Planar sim", leave=False)
        diagnostics: list[dict[str, float | int]] = []
        for step_index in step_iter:
            diag = self.step(t)
            diagnostics.append(diag)
            if store is not None and step_index % self.params.record_interval == 0:
                store.append(
                    t=t,
                    x=self.x,
                    p=self.p,
                    state_id=self.state_id,
                    state_vars=self.state_vars,
                    v=self.v,
                    track_id=self.track_id,
                )
            if callback is not None:
                callback(step_index, t, diag)
            t += float(self.params.dt)
        return diagnostics
