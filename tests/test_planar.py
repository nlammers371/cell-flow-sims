import numpy as np

from cell_sphere_sim.forces import compute_contact_forces_and_metrics
from cell_sphere_sim.io import TrajectoryStore
from cell_sphere_sim.planar import (
    PlanarParams,
    PlanarSimulationEngine,
    candidate_pairs_periodic,
    compute_planar_contact_forces_and_metrics,
    init_random_periodic,
    minimum_image_displacement,
)
from cell_sphere_sim.state import BehaviorParams, StateTable, lookup_behavior


def _state_table(
    *,
    radii=(0.4,),
    motility=(1.0,),
    diffusion=(0.05,),
    fcil=(2.0,),
    adhesion=(0.2,),
) -> StateTable:
    n_states = len(radii)
    return StateTable(
        R=np.asarray(radii, dtype=float),
        Fm=np.asarray(motility, dtype=float),
        Dr=np.asarray(diffusion, dtype=float),
        fcil=np.asarray(fcil, dtype=float),
        w=np.asarray(adhesion, dtype=float),
        lambda_div=np.zeros(n_states),
        tau_div=np.ones(n_states),
    )


def _params(box=(10.0, 10.0), dt=0.01) -> PlanarParams:
    return PlanarParams(
        box_size=box,
        gamma_s=1.0,
        k_rep=2.0,
        alpha_dmin=0.2,
        eps=1e-8,
        dt=dt,
        division_enabled=False,
    )


def test_periodic_neighbors_detect_opposite_edges():
    x = np.array([[0.1, 5.0], [9.9, 5.0], [5.0, 5.0]])
    i_idx, j_idx = candidate_pairs_periodic(x, r=0.3, box_size=(10.0, 10.0))
    assert list(zip(i_idx, j_idx)) == [(0, 1)]


def test_minimum_image_displacement_has_consistent_direction():
    displacement = minimum_image_displacement(
        np.array([0.1, 9.8]),
        np.array([9.9, 0.2]),
        (10.0, 10.0),
    )
    assert np.allclose(displacement, [0.2, -0.4])


def test_two_cell_contact_force_is_equal_and_opposite():
    state_table = _state_table(motility=(0.0,), diffusion=(0.0,))
    behavior = lookup_behavior(np.zeros(2, dtype=np.int32), state_table)
    x = np.array([[0.2, 5.0], [9.7, 5.0]])
    force, metrics = compute_planar_contact_forces_and_metrics(
        x,
        behavior,
        k_rep=2.0,
        alpha_dmin=0.2,
        eps=1e-8,
        i_idx=np.array([0], dtype=np.int32),
        j_idx=np.array([1], dtype=np.int32),
        box_size=(10.0, 10.0),
    )
    assert np.linalg.norm(force[0]) > 0.0
    assert np.allclose(force[0], -force[1])
    assert np.array_equal(metrics.contact_count, [1, 1])


def test_planar_contact_force_is_zero_outside_contact_range():
    state_table = _state_table(motility=(0.0,), diffusion=(0.0,))
    behavior = lookup_behavior(np.zeros(2, dtype=np.int32), state_table)
    force, metrics = compute_planar_contact_forces_and_metrics(
        np.array([[1.0, 1.0], [2.0, 1.0]]),
        behavior,
        k_rep=2.0,
        alpha_dmin=0.2,
        eps=1e-8,
        i_idx=np.array([0], dtype=np.int32),
        j_idx=np.array([1], dtype=np.int32),
        box_size=(10.0, 10.0),
    )
    assert np.array_equal(force, np.zeros((2, 2)))
    assert np.array_equal(metrics.contact_count, [0, 0])


def test_planar_force_magnitude_matches_spherical_model():
    state_table = _state_table(
        radii=(0.4, 0.6),
        motility=(0.0, 0.0),
        diffusion=(0.0, 0.0),
        fcil=(0.0, 0.0),
        adhesion=(0.2, 0.3),
    )
    state_id = np.array([0, 1], dtype=np.int32)
    behavior = lookup_behavior(state_id, state_table)
    pairs = np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)
    planar_force, _ = compute_planar_contact_forces_and_metrics(
        np.array([[5.0, 5.0], [5.5, 5.0]]),
        behavior,
        2.0,
        0.2,
        1e-8,
        *pairs,
        (20.0, 20.0),
    )
    spherical_force, _ = compute_contact_forces_and_metrics(
        np.array([[0.0, 0.0, 10.0], [0.5, 0.0, 10.0]]),
        behavior,
        2.0,
        0.2,
        1e-8,
        *pairs,
        10.0,
    )
    assert np.allclose(np.linalg.norm(planar_force, axis=1), np.linalg.norm(spherical_force, axis=1))


def test_single_cell_ballistic_motion_and_msd():
    table = _state_table(radii=(1.0,), motility=(1.0,), diffusion=(0.0,), fcil=(0.0,))
    engine = PlanarSimulationEngine(
        np.array([[1.0, 2.0]]),
        np.array([[1.0, 0.0]]),
        np.array([0], dtype=np.int32),
        np.zeros((1, 0)),
        table,
        _params(dt=0.05),
        rng=np.random.default_rng(4),
    )
    diagnostics = engine.run(20)
    assert np.allclose(engine.x, [[2.0, 2.0]])
    assert np.allclose(engine.p, [[1.0, 0.0]])
    assert np.isclose(diagnostics[-1]["mean_squared_displacement"], 1.0)


def test_planar_run_records_in_dimension_agnostic_trajectory_store():
    table = _state_table(motility=(1.0,), diffusion=(0.0,), fcil=(0.0,))
    engine = PlanarSimulationEngine(
        np.array([[1.0, 2.0]]),
        np.array([[1.0, 0.0]]),
        np.array([0], dtype=np.int32),
        np.zeros((1, 0)),
        table,
        _params(),
    )
    initial_track_id = engine.track_id.copy()
    store = TrajectoryStore()
    engine.run(2, store=store)
    assert store.x[0].shape == (1, 2)
    assert np.array_equal(engine.track_id, initial_track_id)


def test_position_wraps_into_periodic_box():
    table = _state_table(radii=(1.0,), motility=(1.0,), diffusion=(0.0,), fcil=(0.0,))
    engine = PlanarSimulationEngine(
        np.array([[9.98, 2.0]]),
        np.array([[1.0, 0.0]]),
        np.array([0], dtype=np.int32),
        np.zeros((1, 0)),
        table,
        _params(dt=0.05),
        rng=np.random.default_rng(5),
    )
    engine.step(0.0)
    assert np.allclose(engine.x, [[0.03, 2.0]])
    assert np.allclose(engine.x_unwrapped, [[10.03, 2.0]])


def test_polarity_stays_finite_and_unit_length():
    rng = np.random.default_rng(8)
    table = _state_table(diffusion=(0.2,))
    states = np.zeros(40, dtype=np.int32)
    x, p = init_random_periodic(40, (15.0, 15.0), states, table, rng)
    engine = PlanarSimulationEngine(x, p, states, np.zeros((40, 0)), table, _params((15.0, 15.0)), rng=rng)
    engine.run(200)
    assert np.all(np.isfinite(engine.p))
    assert np.allclose(np.linalg.norm(engine.p, axis=1), 1.0, atol=1e-12)


def test_fixed_seed_reproducibility():
    table = _state_table()

    def build(seed: int) -> PlanarSimulationEngine:
        rng = np.random.default_rng(seed)
        states = np.zeros(50, dtype=np.int32)
        x, p = init_random_periodic(50, (12.0, 12.0), states, table, rng)
        return PlanarSimulationEngine(
            x, p, states, np.zeros((50, 0)), table, _params((12.0, 12.0)), rng=rng
        )

    first = build(99)
    second = build(99)
    assert np.allclose(first.x, second.x)
    assert np.allclose(first.p, second.p)
    first.run(25)
    second.run(25)
    assert np.allclose(first.x, second.x)
    assert np.allclose(first.p, second.p)


def test_initializer_respects_heterogeneous_periodic_clearance():
    rng = np.random.default_rng(14)
    table = _state_table(
        radii=(0.25, 0.45),
        motility=(1.0, 1.0),
        diffusion=(0.0, 0.0),
        fcil=(0.0, 0.0),
        adhesion=(0.0, 0.0),
    )
    states = rng.integers(0, 2, size=100, dtype=np.int32)
    x, p = init_random_periodic(
        100,
        (12.0, 10.0),
        states,
        table,
        rng,
        initial_min_separation_factor=0.9,
    )
    for i in range(100):
        dvec = minimum_image_displacement(x[i], x[i + 1 :], (12.0, 10.0))
        distances = np.linalg.norm(dvec, axis=1)
        required = 0.9 * (table.R[states[i]] + table.R[states[i + 1 :]])
        assert np.all(distances >= required - 1e-12)
    assert np.allclose(np.linalg.norm(p, axis=1), 1.0)


def test_moderate_density_simulation_stays_finite():
    rng = np.random.default_rng(23)
    table = _state_table(radii=(0.35,), diffusion=(0.1,))
    states = np.zeros(150, dtype=np.int32)
    x, p = init_random_periodic(150, (12.0, 12.0), states, table, rng)
    engine = PlanarSimulationEngine(
        x, p, states, np.zeros((150, 0)), table, _params((12.0, 12.0)), rng=rng
    )
    diagnostics = engine.run(50)
    assert len(diagnostics) == 50
    for array in (engine.x, engine.x_unwrapped, engine.p, engine.v):
        assert np.all(np.isfinite(array))
    assert 0.0 <= diagnostics[-1]["polarization"] <= 1.0
    assert 0.0 <= diagnostics[-1]["nematic_order"] <= 1.0
    assert 0.0 < diagnostics[-1]["largest_cluster_fraction"] <= 1.0


def test_planar_division_request_is_explicitly_rejected():
    table = _state_table()
    params = _params()
    params.division_enabled = True
    with np.testing.assert_raises_regex(NotImplementedError, "division"):
        PlanarSimulationEngine(
            np.array([[1.0, 1.0]]),
            np.array([[1.0, 0.0]]),
            np.array([0], dtype=np.int32),
            np.zeros((1, 0)),
            table,
            params,
        )
