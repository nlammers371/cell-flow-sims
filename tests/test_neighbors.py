import numpy as np

from cell_sphere_sim.neighbors import candidate_pairs_ckdtree, interaction_radius


def test_neighbor_candidates_cover_contacts():
    rng = np.random.default_rng(1)
    N = 200
    R = 0.4
    x = rng.normal(size=(N, 3))

    r_query = interaction_radius(np.array([R]), buffer=0.1)
    i_idx, j_idx = candidate_pairs_ckdtree(x, r_query)
    cand_set = {tuple(sorted(pair)) for pair in zip(i_idx.tolist(), j_idx.tolist())}

    contacts = []
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(x[i] - x[j]) < 2.0 * R:
                contacts.append((i, j))

    for pair in contacts:
        assert pair in cand_set
