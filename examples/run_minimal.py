from __future__ import annotations

import numpy as np

from cell_sphere_sim.engine import SimulationEngine, SimParams
from cell_sphere_sim.state import StateTable


def random_points_on_sphere(rng: np.random.Generator, n: int, R_E: float) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    return R_E * v


def random_tangent_polarity(rng: np.random.Generator, x: np.ndarray, R_E: float) -> np.ndarray:
    n = x / R_E
    g = rng.normal(size=x.shape)
    g = g - (np.einsum("ij,ij->i", g, n)[:, None]) * n
    g_norm = np.linalg.norm(g, axis=1)
    g_norm = np.where(g_norm > 1e-12, g_norm, 1.0)
    return g / g_norm[:, None]


def main() -> None:
    rng = np.random.default_rng(123)
    N = 1000
    R_E = 10.0

    x = random_points_on_sphere(rng, N, R_E)
    p = random_tangent_polarity(rng, x, R_E)
    state_id = np.zeros((N,), dtype=np.int32)
    state_vars = np.zeros((N, 0), dtype=float)

    state_table = StateTable(
        R=np.array([0.4]),
        Fm=np.array([1.0]),
        Dr=np.array([0.05]),
        fcil=np.array([2.0]),
        w=np.array([0.2]),
        lambda_div=np.array([0.0]),
        tau_div=np.array([1.0]),
    )

    params = SimParams(
        R_E=R_E,
        gamma_s=1.0,
        k_rep=2.0,
        alpha_dmin=0.2,
        eps=1e-8,
        dt=0.01,
        neighbor_cell_size=1.0,
        record_interval=1,
        division_enabled=False,
    )

    engine = SimulationEngine(x, p, state_id, state_vars, state_table, params, rng=rng)

    for step in range(10):
        diag = engine.step(t=step * params.dt)
        print(step, diag)


if __name__ == "__main__":
    main()
