from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import warnings
import numpy as np

from .fields.base import FieldModel, NullField
from .forces import compute_contact_forces_and_metrics, ContactMetrics
from .neighbors import candidate_pairs_ckdtree, interaction_radius
from .polarity import parallel_transport, random_tangent_rotation, cil_target_flee
from .division import apply_divisions
from .state import BehaviorParams, StateTable, lookup_behavior


@dataclass
class SimParams:
    R_E: float
    gamma_s: float
    k_rep: float
    alpha_dmin: float
    eps: float
    dt: float | None
    neighbor_radius_buffer: float = 0.1
    record_interval: int
    division_enabled: bool = True
    split_scale: float = 0.5
    relax_substeps: int = 5
    relax_dt_scale: float = 0.2


CellUpdateFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray, ContactMetrics, float, np.random.Generator, StateTable],
    tuple[np.ndarray, np.ndarray, BehaviorParams],
]

CellSourcesFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray, float, StateTable],
    np.ndarray,
]


def compute_default_dt(state_table: StateTable, gamma_s: float, eps: float, eta: float = 0.02) -> float:
    sigma_min = 2.0 * float(np.min(state_table.R))
    v_m_max = float(np.max(state_table.Fm)) / gamma_s
    return eta * sigma_min / (v_m_max + eps)


def warn_dt(dt: float, state_table: StateTable, gamma_s: float, eps: float) -> None:
    sigma_min = 2.0 * float(np.min(state_table.R))
    v_m_max = float(np.max(state_table.Fm)) / gamma_s
    dx = v_m_max * dt
    if dx > 0.1 * sigma_min:
        warnings.warn("dt may be too large for stability", RuntimeWarning)
    if dx < 0.002 * sigma_min:
        warnings.warn("dt may be too small for efficiency", RuntimeWarning)


def default_cell_update(
    state_id: np.ndarray,
    state_vars: np.ndarray,
    fields: np.ndarray,
    contact_metrics: ContactMetrics,
    dt: float,
    rng: np.random.Generator,
    state_table: StateTable,
) -> tuple[np.ndarray, np.ndarray, BehaviorParams]:
    """Identity update for state and pure table lookup for behavior."""
    behavior = lookup_behavior(state_id, state_table)
    return state_id, state_vars, behavior


def default_cell_sources(
    state_id: np.ndarray,
    state_vars: np.ndarray,
    fields: np.ndarray,
    dt: float,
    state_table: StateTable,
) -> np.ndarray:
    return np.zeros((state_id.shape[0], fields.shape[1]), dtype=fields.dtype)


class SimulationEngine:
    def __init__(
        self,
        x: np.ndarray,
        p: np.ndarray,
        state_id: np.ndarray,
        state_vars: np.ndarray,
        state_table: StateTable,
        params: SimParams,
        field: FieldModel | None = None,
        cell_update: CellUpdateFn | None = None,
        cell_sources: CellSourcesFn | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.x = np.asarray(x, dtype=float)
        self.p = np.asarray(p, dtype=float)
        self.state_id = np.asarray(state_id, dtype=np.int32)
        self.state_vars = np.asarray(state_vars, dtype=float)
        self.state_table = state_table
        self.params = params
        if self.params.dt is None:
            self.params.dt = compute_default_dt(self.state_table, self.params.gamma_s, self.params.eps)
        warn_dt(float(self.params.dt), self.state_table, self.params.gamma_s, self.params.eps)
        self.field = field if field is not None else NullField(C=2)
        self.cell_update = cell_update or default_cell_update
        self.cell_sources = cell_sources or default_cell_sources
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.paused_until = np.zeros((self.x.shape[0],), dtype=float)
        self.track_id = np.arange(self.x.shape[0], dtype=np.int64)
        self.parent_id = np.full((self.x.shape[0],), -1, dtype=np.int64)
        self.next_track_id = int(self.track_id.max()) + 1 if self.track_id.size else 0
        self.v = np.zeros_like(self.x)
        self.contact_metrics = ContactMetrics(
            contact_count=np.zeros((self.x.shape[0],), dtype=int),
            contact_dir_sum=np.zeros_like(self.x),
        )

    def step(self, t: float) -> dict:
        params = self.params
        dt = float(params.dt)
        n = self.x / params.R_E
        fields = self.field.sample(self.x, t)

        self.state_id, self.state_vars, behavior = self.cell_update(
            self.state_id,
            self.state_vars,
            fields,
            self.contact_metrics,
            params.dt,
            self.rng,
            self.state_table,
        )

        x_contact = self.x
        r_query = interaction_radius(behavior.R, params.neighbor_radius_buffer)
        i_idx, j_idx = candidate_pairs_ckdtree(x_contact, r_query)
        F_contact, metrics = compute_contact_forces_and_metrics(
            x_contact,
            behavior,
            params.k_rep,
            params.alpha_dmin,
            params.eps,
            i_idx,
            j_idx,
            params.R_E,
        )
        self.contact_metrics = metrics

        gate = (t >= self.paused_until).astype(float)
        v = (gate[:, None] * behavior.Fm[:, None] * self.p + F_contact) / params.gamma_s

        x_old = self.x
        x_tmp = self.x + dt * v
        norms = np.linalg.norm(x_tmp, axis=1)
        norms = np.where(norms > params.eps, norms, 1.0)
        x_new = params.R_E * x_tmp / norms[:, None]
        n_new = x_new / params.R_E

        self.v = (x_new - x_old) / dt

        p_tr = parallel_transport(self.p, n, n_new, params.eps)
        has_target, p_flee = cil_target_flee(metrics.contact_dir_sum, n_new, params.eps)

        p_det = p_tr.copy()
        if np.any(has_target):
            relax = np.exp(-behavior.fcil * params.dt)
            p_det[has_target] = p_flee[has_target] + relax[has_target, None] * (
                p_tr[has_target] - p_flee[has_target]
            )

        delta = self.rng.normal(scale=np.sqrt(2.0 * behavior.Dr * dt))
        p_new = random_tangent_rotation(p_det, n_new, delta)
        p_new = p_new - (np.einsum("ij,ij->i", p_new, n_new)[:, None]) * n_new
        p_norm = np.linalg.norm(p_new, axis=1)
        p_norm = np.where(p_norm > params.eps, p_norm, 1.0)
        p_new = p_new / p_norm[:, None]

        self.x = x_new
        self.p = p_new

        prev_n = self.x.shape[0]
        if params.division_enabled:
            (
                self.x,
                self.p,
                self.state_id,
                self.state_vars,
                self.paused_until,
                self.track_id,
                self.parent_id,
                self.next_track_id,
                div_idx,
            ) = apply_divisions(
                self.x,
                self.p,
                self.state_id,
                self.state_vars,
                self.paused_until,
                self.track_id,
                self.parent_id,
                self.next_track_id,
                t,
                behavior,
                params.R_E,
                params.split_scale,
                self.rng,
                dt=dt,
                eps=params.eps,
            )
            if div_idx.size:
                self.v[div_idx] = 0.0
                new_n = self.x.shape[0]
                if new_n > prev_n:
                    self.v = np.vstack([self.v, np.zeros((new_n - prev_n, 3), dtype=self.v.dtype)])

        if self.x.shape[0] != prev_n and params.relax_substeps > 0:
            self._relax_contacts()
            self.contact_metrics = ContactMetrics(
                contact_count=np.zeros((self.x.shape[0],), dtype=int),
                contact_dir_sum=np.zeros_like(self.x),
            )

        fields = self.field.sample(self.x, t)
        sources = self.cell_sources(self.state_id, self.state_vars, fields, dt, self.state_table)
        self.field.accumulate_sources(self.x, sources)
        self.field.step(dt)
        self.field.reset_sources()

        speed = np.linalg.norm(v, axis=1)
        n_candidates = int(i_idx.size)
        n_contact_pairs = int(np.sum(metrics.contact_count) // 2)
        min_d_contact = float("nan")
        if i_idx.size:
            d = np.linalg.norm(x_contact[i_idx] - x_contact[j_idx], axis=1)
            sigma = behavior.R[i_idx] + behavior.R[j_idx]
            contact_mask = d < sigma
            if np.any(contact_mask):
                min_d_contact = float(np.min(d[contact_mask]))
        return {
            "n_cells": int(self.x.shape[0]),
            "mean_speed": float(np.mean(speed)) if speed.size else 0.0,
            "mean_contacts": float(np.mean(metrics.contact_count)) if metrics.contact_count.size else 0.0,
            "n_candidates": n_candidates,
            "n_contact_pairs": n_contact_pairs,
            "min_d_contact": min_d_contact,
        }

    def _relax_contacts(self) -> None:
        params = self.params
        for _ in range(params.relax_substeps):
            behavior = lookup_behavior(self.state_id, self.state_table)
            r_query = interaction_radius(behavior.R, params.neighbor_radius_buffer)
            i_idx, j_idx = candidate_pairs_ckdtree(self.x, r_query)
            F_contact, _ = compute_contact_forces_and_metrics(
                self.x,
                behavior,
                params.k_rep,
                params.alpha_dmin,
                params.eps,
                i_idx,
                j_idx,
                params.R_E,
            )

            x_tmp = self.x + float(params.dt) * params.relax_dt_scale * (F_contact / params.gamma_s)
            norms = np.linalg.norm(x_tmp, axis=1)
            norms = np.where(norms > params.eps, norms, 1.0)
            x_new = params.R_E * x_tmp / norms[:, None]
            n_old = self.x / params.R_E
            n_new = x_new / params.R_E
            self.p = parallel_transport(self.p, n_old, n_new, params.eps)
            self.x = x_new

    def run(
        self,
        n_steps: int,
        t0: float = 0.0,
        store=None,
        callback=None,
        show_progress: bool = False,
    ) -> None:
        t = float(t0)
        step_iter = range(n_steps)
        if show_progress:
            from tqdm import tqdm

            step_iter = tqdm(step_iter, desc="Sim", leave=False)
        for step in step_iter:
            diag = self.step(t)
            if store is not None and (step % self.params.record_interval == 0):
                store.append(
                    t=t,
                    x=self.x,
                    p=self.p,
                    state_id=self.state_id,
                    state_vars=self.state_vars,
                    v=self.v,
                    track_id=self.track_id,
                    extra={"parent_id": self.parent_id},
                )
            if callback is not None:
                callback(step, t, diag)
            t += float(self.params.dt)
