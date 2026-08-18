from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import warnings

import numpy as np

from ..forces import ContactMetrics
from ..neighbors import interaction_radius
from ..state import BehaviorParams, StateTable, lookup_behavior
from .constraints import project_seeded_overlaps_periodic
from .division import apply_planar_divisions
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
    division_separation_factor: float = 1.0
    division_projection_enabled: bool = True
    division_projection_tolerance: float = 1e-8
    division_projection_max_iterations: int = 500
    division_projection_failure_policy: str = "reject"
    division_projection_max_displacement_factor: float | None = 2.0


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
        track_id: np.ndarray | None = None,
        parent_id: np.ndarray | None = None,
    ) -> None:
        _validate_state_table(state_table)
        self.box_size = validate_box_size(params.box_size)
        self.params = params
        self._validate_params()
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
        if track_id is None:
            track_ids = np.arange(points.shape[0], dtype=np.int64)
        else:
            track_ids = np.asarray(track_id, dtype=np.int64)
            if track_ids.shape != (points.shape[0],):
                raise ValueError("track_id must have shape (N,)")
            if np.unique(track_ids).size != track_ids.size:
                raise ValueError("track_id values must be unique")
        if parent_id is None:
            parent_ids = np.full(points.shape[0], -1, dtype=np.int64)
        else:
            parent_ids = np.asarray(parent_id, dtype=np.int64)
            if parent_ids.shape != (points.shape[0],):
                raise ValueError("parent_id must have shape (N,)")

        self.track_id = track_ids.copy()
        self.parent_id = parent_ids.copy()
        self.next_track_id = int(np.max(self.track_id)) + 1 if self.track_id.size else 0
        self.paused_until = np.zeros(points.shape[0], dtype=float)
        self.last_division_projection_displacement = np.zeros_like(points)
        self.total_divisions = 0
        self.total_rejected_divisions = 0
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
        if (
            not np.isfinite(params.division_separation_factor)
            or params.division_separation_factor <= 0.0
        ):
            raise ValueError("division_separation_factor must be finite and positive")
        if (
            not np.isfinite(params.division_projection_tolerance)
            or params.division_projection_tolerance < 0.0
        ):
            raise ValueError("division_projection_tolerance must be finite and non-negative")
        if params.division_projection_max_iterations <= 0:
            raise ValueError("division_projection_max_iterations must be positive")
        if params.division_projection_failure_policy not in {"reject", "raise"}:
            raise ValueError(
                "division_projection_failure_policy must be 'reject' or 'raise'"
            )
        max_projection_factor = params.division_projection_max_displacement_factor
        if max_projection_factor is not None and (
            not np.isfinite(max_projection_factor) or max_projection_factor <= 0.0
        ):
            raise ValueError(
                "division_projection_max_displacement_factor must be None or "
                "finite and positive"
            )

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

        gate = (t >= self.paused_until).astype(float)
        self.v = (gate[:, None] * behavior.Fm[:, None] * self.p + force) / params.gamma_s
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
        largest_cluster = largest_cluster_fraction(self.x.shape[0], contact_i, contact_j)

        n_divisions = 0
        n_division_attempts = 0
        n_rejected_divisions = 0
        projection_iterations = 0
        projection_cells_moved = 0
        projection_initial_overlap = 0.0
        projection_residual_overlap = 0.0
        projection_max_displacement = 0.0
        projection_rms_displacement = 0.0
        self.last_division_projection_displacement = np.zeros_like(self.x)
        if params.division_enabled:
            previous_n = self.x.shape[0]
            initial_positions = self.x_unwrapped_initial
            (
                divided_x,
                divided_x_unwrapped,
                divided_p,
                divided_state_id,
                divided_state_vars,
                divided_paused_until,
                divided_track_id,
                divided_parent_id,
                divided_next_track_id,
                div_idx,
            ) = apply_planar_divisions(
                self.x,
                self.x_unwrapped,
                self.p,
                self.state_id,
                self.state_vars,
                self.paused_until,
                self.track_id,
                self.parent_id,
                self.next_track_id,
                t,
                behavior,
                self.box_size,
                params.division_separation_factor,
                self.rng,
                dt,
            )
            n_division_attempts = int(div_idx.size)
            if n_division_attempts:
                appended = np.arange(previous_n, divided_x.shape[0])
                projection_displacement = np.zeros_like(divided_x)
                projection = None
                projection_error = None
                if params.division_projection_enabled:
                    projection_seeds = np.concatenate((div_idx, appended))
                    try:
                        (
                            divided_x,
                            divided_x_unwrapped,
                            projection,
                        ) = project_seeded_overlaps_periodic(
                            divided_x,
                            divided_x_unwrapped,
                            self.state_table.R[divided_state_id],
                            projection_seeds,
                            self.box_size,
                            tolerance=params.division_projection_tolerance,
                            max_iterations=params.division_projection_max_iterations,
                            eps=params.eps,
                        )
                    except RuntimeError as exc:
                        projection_error = exc

                if projection is not None:
                    projection_iterations = projection.iterations
                    projection_cells_moved = projection.n_cells_moved
                    projection_initial_overlap = projection.initial_max_overlap
                    projection_residual_overlap = projection.final_max_overlap
                    projection_max_displacement = projection.max_displacement
                    projection_rms_displacement = projection.rms_displacement
                    projection_displacement = projection.displacement.copy()
                    max_factor = params.division_projection_max_displacement_factor
                    if max_factor is not None:
                        allowed_displacement = max_factor * float(
                            np.max(behavior.R[div_idx])
                        )
                        if projection.max_displacement > allowed_displacement:
                            projection_error = RuntimeError(
                                "division overlap projection required an implausibly "
                                f"large displacement ({projection.max_displacement:.6g} > "
                                f"{allowed_displacement:.6g})"
                            )

                if projection_error is not None:
                    if params.division_projection_failure_policy == "raise":
                        raise projection_error
                    n_rejected_divisions = n_division_attempts
                    self.total_rejected_divisions += n_rejected_divisions
                else:
                    # Commit all daughter-related arrays together only after the
                    # geometry check succeeds. A rejected division therefore
                    # cannot leave partial daughters or wrapped edge artifacts.
                    self.x = divided_x
                    self.x_unwrapped = divided_x_unwrapped
                    self.p = divided_p
                    self.state_id = divided_state_id
                    self.state_vars = divided_state_vars
                    self.paused_until = divided_paused_until
                    self.track_id = divided_track_id
                    self.parent_id = divided_parent_id
                    self.next_track_id = divided_next_track_id
                    self.last_division_projection_displacement = projection_displacement
                    n_divisions = n_division_attempts

                if n_divisions:
                    new_initial_positions = np.empty_like(self.x_unwrapped)
                    new_initial_positions[:previous_n] = initial_positions
                    new_initial_positions[div_idx] = self.x_unwrapped[div_idx]
                    new_initial_positions[appended] = self.x_unwrapped[appended]
                    self.x_unwrapped_initial = new_initial_positions

                    self.v[div_idx] = 0.0
                    self.v = np.vstack(
                        [self.v, np.zeros((n_divisions, 2), dtype=self.v.dtype)]
                    )
                    self.contact_metrics = ContactMetrics(
                        contact_count=np.zeros(self.x.shape[0], dtype=int),
                        contact_dir_sum=np.zeros_like(self.x),
                    )
                    self.total_divisions += n_divisions

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
            "largest_cluster_fraction": largest_cluster,
            "mean_squared_displacement": (
                float(np.mean(squared_displacement)) if squared_displacement.size else 0.0
            ),
            "n_divisions": n_divisions,
            "n_division_attempts": n_division_attempts,
            "n_rejected_divisions": n_rejected_divisions,
            "total_divisions": self.total_divisions,
            "total_rejected_divisions": self.total_rejected_divisions,
            "division_projection_iterations": projection_iterations,
            "division_projection_cells_moved": projection_cells_moved,
            "division_projection_initial_overlap": projection_initial_overlap,
            "division_projection_residual_overlap": projection_residual_overlap,
            "division_projection_max_displacement": projection_max_displacement,
            "division_projection_rms_displacement": projection_rms_displacement,
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
                    parent_id=self.parent_id,
                )
            if callback is not None:
                callback(step_index, t, diag)
            t += float(self.params.dt)
        return diagnostics
