import numpy as np

from cell_sphere_sim.polarity import parallel_transport


def test_parallel_transport_small_move():
    n = np.array([[0.0, 0.0, 1.0]])
    n_new = np.array([[0.001, 0.0, np.sqrt(1.0 - 0.001**2)]])
    p = np.array([[1.0, 0.0, 0.0]])

    p_tr = parallel_transport(p, n, n_new, eps=1e-12)

    assert np.all(np.isfinite(p_tr))
    assert float(np.dot(p_tr[0], p[0])) > 0.9
