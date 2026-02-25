from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from .fields.base import FieldModel, NullField
from .forces import compute_contact_forces_and_metrics, ContactMetrics
from .neighbors import build_spatial_hash, candidate_pairs_from_hash
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
    dt: float
    neighbor_cell_size: float
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
        self.field = field if field is not None else NullField(C=2)
        self.cell_update = cell_update or default_cell_update
        self.cell_sources = cell_sources or default_cell_sources
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.paused_until = np.zeros((self.x.shape[0],), dtype=float)
        self.contact_metrics = ContactMetrics(
            contact_count=np.zeros((self.x.shape[0],), dtype=int),
            contact_dir_sum=np.zeros_like(self.x),
        )

    def step(self, t: float) -> dict:
        params = self.params
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

        hash_map = build_spatial_hash(self.x, params.neighbor_cell_size)
        candidates = candidate_pairs_from_hash(self.x, hash_map, params.neighbor_cell_size)
        F_contact, metrics = compute_contact_forces_and_metrics(
            self.x,
            self.state_id,
            behavior,
            params.k_rep,
            params.alpha_dmin,
            params.eps,
            candidates,
            params.R_E,
        )
        self.contact_metrics = metrics

        gate = (t >= self.paused_until).astype(float)
        v = (gate[:, None] * behavior.Fm[:, None] * self.p + F_contact) / params.gamma_s

        x_tmp = self.x + params.dt * v
        norms = np.linalg.norm(x_tmp, axis=1)
        norms = np.where(norms > params.eps, norms, 1.0)
        x_new = params.R_E * x_tmp / norms[:, None]
        n_new = x_new / params.R_E

        p_tr = parallel_transport(self.p, n, n_new, params.eps)
        has_target, p_flee = cil_target_flee(metrics.contact_dir_sum, n_new, params.eps)

        p_det = p_tr.copy()
        if np.any(has_target):
            relax = np.exp(-behavior.fcil * params.dt)
            p_det[has_target] = p_flee[has_target] + relax[has_target, None] * (
                p_tr[has_target] - p_flee[has_target]
            )

        delta = self.rng.normal(scale=np.sqrt(2.0 * behavior.Dr * params.dt))
        p_new = random_tangent_rotation(p_det, n_new, delta)
        p_new = p_new - (np.einsum("ij,ij->i", p_new, n_new)[:, None]) * n_new
        p_norm = np.linalg.norm(p_new, axis=1)
        p_norm = np.where(p_norm > params.eps, p_norm, 1.0)
        p_new = p_new / p_norm[:, None]

        self.x = x_new
        self.p = p_new

        prev_n = self.x.shape[0]
        if params.division_enabled:
            self.x, self.p, self.state_id, self.state_vars, self.paused_until = apply_divisions(
                self.x,
                self.p,
                self.state_id,
                self.state_vars,
                self.paused_until,
                t,
                behavior,
                params.R_E,
                params.split_scale,
                self.rng,
                dt=params.dt,
                eps=params.eps,
            )

        if self.x.shape[0] != prev_n and params.relax_substeps > 0:
            self._relax_contacts()
            self.contact_metrics = ContactMetrics(
                contact_count=np.zeros((self.x.shape[0],), dtype=int),
                contact_dir_sum=np.zeros_like(self.x),
            )

        fields = self.field.sample(self.x, t)
        sources = self.cell_sources(self.state_id, self.state_vars, fields, params.dt, self.state_table)
        self.field.accumulate_sources(self.x, sources)
        self.field.step(params.dt)
        self.field.reset_sources()

        speed = np.linalg.norm(v, axis=1)
        return {
            "n_cells": int(self.x.shape[0]),
            "mean_speed": float(np.mean(speed)) if speed.size else 0.0,
            "mean_contacts": float(np.mean(metrics.contact_count)) if metrics.contact_count.size else 0.0,
        }

    def _relax_contacts(self) -> None:
        params = self.params
        for _ in range(params.relax_substeps):
            behavior = lookup_behavior(self.state_id, self.state_table)
            hash_map = build_spatial_hash(self.x, params.neighbor_cell_size)
            candidates = candidate_pairs_from_hash(self.x, hash_map, params.neighbor_cell_size)
            F_contact, _ = compute_contact_forces_and_metrics(
                self.x,
                self.state_id,
                behavior,
                params.k_rep,
                params.alpha_dmin,
                params.eps,
                candidates,
                params.R_E,
            )

            x_tmp = self.x + params.dt * params.relax_dt_scale * (F_contact / params.gamma_s)
            norms = np.linalg.norm(x_tmp, axis=1)
            norms = np.where(norms > params.eps, norms, 1.0)
            x_new = params.R_E * x_tmp / norms[:, None]
            n_old = self.x / params.R_E
            n_new = x_new / params.R_E
            self.p = parallel_transport(self.p, n_old, n_new, params.eps)
            self.x = x_new

    def run(self, n_steps: int, t0: float = 0.0, store=None, callback=None) -> None:
        t = float(t0)
        for step in range(n_steps):
            diag = self.step(t)
            if store is not None and (step % self.params.record_interval == 0):
                store.append(t, self.x, self.p, self.state_id, self.state_vars)
            if callback is not None:
                callback(step, t, diag)
            t += self.params.dt
