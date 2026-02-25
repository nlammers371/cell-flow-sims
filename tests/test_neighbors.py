import numpy as np

from cell_sphere_sim.neighbors import build_spatial_hash, candidate_pairs_from_hash


def test_neighbor_candidates_cover_contacts():
    rng = np.random.default_rng(1)
    N = 200
    R = 0.4
    x = rng.normal(size=(N, 3))

    cell_size = 2.0 * R
    hash_map = build_spatial_hash(x, cell_size)
    candidates = candidate_pairs_from_hash(x, hash_map, cell_size)
    cand_set = {tuple(sorted(pair)) for pair in candidates}

    contacts = []
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(x[i] - x[j]) < 2.0 * R:
                contacts.append((i, j))

    for pair in contacts:
        assert pair in cand_set
