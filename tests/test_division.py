import numpy as np

from cell_sphere_sim.division import apply_divisions
from cell_sphere_sim.state import BehaviorParams


def test_division_inserts_daughter_and_inherits_state():
    rng = np.random.default_rng(0)
    R_E = 10.0
    x = np.array([[0.0, 0.0, R_E]], dtype=float)
    p = np.array([[1.0, 0.0, 0.0]], dtype=float)
    state_id = np.array([2], dtype=np.int32)
    state_vars = np.array([[3.0, 4.0]], dtype=float)
    paused_until = np.zeros((1,), dtype=float)

    behavior = BehaviorParams(
        R=np.array([0.5]),
        Fm=np.array([1.0]),
        Dr=np.array([0.0]),
        fcil=np.array([0.0]),
        w=np.array([0.0]),
        lambda_div=np.array([1000.0]),
        tau_div=np.array([2.0]),
    )

    x2, p2, state_id2, state_vars2, paused2 = apply_divisions(
        x,
        p,
        state_id,
        state_vars,
        paused_until,
        t=0.0,
        behavior=behavior,
        R_E=R_E,
        split_scale=0.5,
        rng=rng,
        dt=1.0,
    )

    assert x2.shape[0] == 2
    assert np.all(state_id2 == 2)
    assert np.all(state_vars2 == state_vars2[0])
    assert np.all(paused2 >= 0.0)

    gate = (0.0 >= paused2).astype(float)
    v = gate[:, None] * behavior.Fm[0] * p2
    assert np.allclose(v, 0.0)
