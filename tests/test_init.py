import numpy as np
from scipy.spatial import cKDTree

from cell_sphere_sim.init import init_random_on_sphere
from cell_sphere_sim.state import StateTable


def test_init_enforces_min_distance():
    rng = np.random.default_rng(3)
    N = 40
    R_E = 10.0
    state_table = StateTable(
        R=np.array([0.5, 0.7]),
        Fm=np.array([1.0, 1.0]),
        Dr=np.array([0.0, 0.0]),
        fcil=np.array([0.0, 0.0]),
        w=np.array([0.0, 0.0]),
        lambda_div=np.array([0.0, 0.0]),
        tau_div=np.array([0.0, 0.0]),
    )
    state_id = rng.integers(0, 2, size=N, dtype=np.int32)

    x, _ = init_random_on_sphere(
        N=N,
        R_E=R_E,
        state_id=state_id,
        state_table=state_table,
        alpha_dmin=0.5,
        eps=1e-8,
        rng=rng,
        pos_mode="uniform",
        pos_strength=0.0,
        mixing=1.0,
    )

    tree = cKDTree(x)
    r_query = 0.5 * (2.0 * float(np.max(state_table.R)))
    pairs = tree.query_pairs(r=r_query, output_type="ndarray")
    for i, j in pairs:
        d = np.linalg.norm(x[i] - x[j])
        d_min = 0.5 * (state_table.R[state_id[i]] + state_table.R[state_id[j]])
        assert d >= d_min - 1e-8
