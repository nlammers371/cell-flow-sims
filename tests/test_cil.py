import numpy as np

from cell_sphere_sim.forces import compute_contact_forces_and_metrics
from cell_sphere_sim.polarity import cil_target_flee
from cell_sphere_sim.state import BehaviorParams


def test_cil_flee_direction():
    R_E = 10.0
    x = np.array([[0.0, 0.0, R_E], [0.0, 0.5, R_E]], dtype=float)
    behavior = BehaviorParams(
        R=np.array([1.0, 1.0]),
        Fm=np.array([0.0, 0.0]),
        Dr=np.array([0.0, 0.0]),
        fcil=np.array([1.0, 1.0]),
        w=np.array([0.0, 0.0]),
        lambda_div=np.array([0.0, 0.0]),
        tau_div=np.array([0.0, 0.0]),
    )

    i_idx = np.array([0], dtype=np.int32)
    j_idx = np.array([1], dtype=np.int32)
    _, metrics = compute_contact_forces_and_metrics(
        x,
        behavior,
        k_rep=1.0,
        alpha_dmin=0.2,
        eps=1e-8,
        i_idx=i_idx,
        j_idx=j_idx,
        R_E=R_E,
    )

    n_new = x / R_E
    has_target, p_flee = cil_target_flee(metrics.contact_dir_sum, n_new, eps=1e-8)
    assert has_target[0]

    dvec = x[1] - x[0]
    d_t = dvec - np.dot(dvec, n_new[0]) * n_new[0]
    d_t = d_t / np.linalg.norm(d_t)
    assert float(np.dot(p_flee[0], d_t)) < 0.0


def test_cil_relaxation_moves_toward_target():
    p = np.array([[1.0, 0.0, 0.0]])
    # Avoid exact antipodes so normalization does not collapse progress.
    p_flee = np.array([[-0.5, 0.8660254, 0.0]])
    fcil = 2.0
    dt = 0.1

    dots = []
    for _ in range(5):
        p = p_flee + np.exp(-fcil * dt) * (p - p_flee)
        p = p / np.linalg.norm(p, axis=1)[:, None]
        dots.append(float(np.dot(p[0], p_flee[0])))

    assert dots[-1] > dots[0]
