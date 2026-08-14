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
    project_seeded_overlaps_periodic,
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
        motility=(1.0, 1.0),
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


def test_planar_division_is_state_specific_symmetric_and_lineage_aware():
    table = _state_table(
        radii=(0.4, 0.6),
        motility=(1.0, 1.0),
        diffusion=(0.0, 0.0),
        fcil=(0.0, 0.0),
        adhesion=(0.0, 0.0),
    )
    table.lambda_div[:] = [1000.0, 0.0]
    table.tau_div[:] = [0.7, 1.2]
    params = _params()
    params.division_enabled = True
    engine = PlanarSimulationEngine(
        np.array([[0.1, 5.0], [5.0, 5.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([0, 1], dtype=np.int32),
        np.array([[3.0], [4.0]]),
        table,
        params,
        rng=np.random.default_rng(41),
        track_id=np.array([10, 20], dtype=np.int64),
    )

    diagnostics = engine.step(0.0)

    assert diagnostics["n_divisions"] == 1
    assert diagnostics["total_divisions"] == 1
    assert diagnostics["n_cells"] == 3
    assert np.array_equal(engine.state_id, [0, 1, 0])
    assert np.array_equal(engine.state_vars[:, 0], [3.0, 4.0, 3.0])
    assert np.array_equal(engine.track_id, [21, 20, 22])
    assert np.array_equal(engine.parent_id, [10, -1, 10])
    assert np.allclose(engine.p[[0, 2]], [[1.0, 0.0], [1.0, 0.0]])
    assert np.allclose(engine.paused_until[[0, 2]], 0.01 + 0.7)

    daughter_displacement = minimum_image_displacement(engine.x[0], engine.x[2], (10.0, 10.0))
    assert np.isclose(np.linalg.norm(daughter_displacement), 2.0 * table.R[0])
    assert np.allclose(0.5 * (engine.x_unwrapped[0] + engine.x_unwrapped[2]), [0.11, 5.0])
    assert np.all((engine.x >= 0.0) & (engine.x < 10.0))


def test_seeded_projection_propagates_without_moving_unrelated_overlaps():
    x = np.array(
        [
            [1.0, 1.0],
            [1.5, 1.0],
            [2.2, 1.0],
            [7.0, 7.0],
            [7.5, 7.0],
        ]
    )
    projected, unwrapped, diagnostics = project_seeded_overlaps_periodic(
        x,
        x.copy(),
        np.full(5, 0.5),
        np.array([0]),
        (10.0, 10.0),
    )

    assert diagnostics.n_cells_moved == 3
    assert diagnostics.initial_max_overlap == 0.5
    assert diagnostics.final_max_overlap <= 1e-8
    assert diagnostics.max_displacement > 0.0
    assert np.allclose(np.sum(unwrapped[:3], axis=0), np.sum(x[:3], axis=0))
    assert np.allclose(projected[3:], x[3:])
    assert np.isclose(np.linalg.norm(projected[3] - projected[4]), 0.5)
    for i, j in ((0, 1), (1, 2)):
        distance = np.linalg.norm(
            minimum_image_displacement(projected[i], projected[j], (10.0, 10.0))
        )
        assert distance >= 1.0 - 1e-8


def test_seeded_projection_uses_periodic_minimum_image():
    x = np.array([[0.1, 5.0], [9.7, 5.0]])
    projected, unwrapped, diagnostics = project_seeded_overlaps_periodic(
        x,
        x.copy(),
        np.array([0.3, 0.3]),
        np.array([0]),
        (10.0, 10.0),
    )

    separation = np.linalg.norm(
        minimum_image_displacement(projected[0], projected[1], (10.0, 10.0))
    )
    assert separation >= 0.6 - 1e-8
    assert np.allclose(np.sum(unwrapped, axis=0), np.sum(x, axis=0))
    assert diagnostics.n_cells_moved == 2


def test_division_projection_shoves_neighbors_without_creating_velocity_spikes():
    radius = 0.4
    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    x = np.vstack(
        (
            [5.0, 5.0],
            np.column_stack(
                (
                    5.0 + 2.0 * radius * np.cos(angles),
                    5.0 + 2.0 * radius * np.sin(angles),
                )
            ),
        )
    )
    table = _state_table(
        radii=(radius, radius),
        motility=(1.0, 1.0),
        diffusion=(0.0, 0.0),
        fcil=(0.0, 0.0),
        adhesion=(0.0, 0.0),
    )
    table.lambda_div[:] = [1000.0, 0.0]
    params = _params()
    params.k_rep = 0.0
    params.division_enabled = True
    engine = PlanarSimulationEngine(
        x,
        np.tile([1.0, 0.0], (7, 1)),
        np.array([0, 1, 1, 1, 1, 1, 1], dtype=np.int32),
        np.zeros((7, 0)),
        table,
        params,
        rng=np.random.default_rng(41),
    )

    diagnostics = engine.step(0.0)

    assert diagnostics["n_divisions"] == 1
    assert diagnostics["division_projection_cells_moved"] > 2
    assert diagnostics["division_projection_max_displacement"] > 0.0
    assert diagnostics["division_projection_residual_overlap"] <= 1e-8
    assert np.allclose(engine.v[1:7], [1.0, 0.0])
    assert np.allclose(engine.v[[0, 7]], 0.0)
    assert np.any(
        np.linalg.norm(engine.last_division_projection_displacement[1:7], axis=1) > 0.0
    )
    assert np.allclose(np.sum(engine.last_division_projection_displacement, axis=0), 0.0)
    radii = table.R[engine.state_id]
    for i in range(engine.x.shape[0]):
        displacement = minimum_image_displacement(
            engine.x[i], engine.x[i + 1 :], engine.box_size
        )
        distances = np.linalg.norm(displacement, axis=1)
        assert np.all(distances >= radii[i] + radii[i + 1 :] - 1e-8)


def test_division_pause_gates_motility_for_full_tau():
    table = _state_table(
        radii=(0.4,),
        motility=(1.0,),
        diffusion=(0.0,),
        fcil=(0.0,),
        adhesion=(0.0,),
    )
    table.lambda_div[:] = 1000.0
    table.tau_div[:] = 0.5
    params = _params(dt=0.05)
    params.k_rep = 0.0
    params.division_enabled = True
    engine = PlanarSimulationEngine(
        np.array([[5.0, 5.0]]),
        np.array([[1.0, 0.0]]),
        np.array([0], dtype=np.int32),
        np.zeros((1, 0)),
        table,
        params,
        rng=np.random.default_rng(42),
    )

    engine.step(0.0)
    table.lambda_div[:] = 0.0
    birth_positions = engine.x_unwrapped.copy()
    engine.step(0.05)
    assert np.allclose(engine.x_unwrapped, birth_positions)

    engine.step(0.55)
    assert np.allclose(engine.x_unwrapped[:, 0], birth_positions[:, 0] + 0.05)


def test_division_run_store_records_dynamic_lineage_arrays():
    table = _state_table(motility=(1.0,), diffusion=(0.0,), fcil=(0.0,), adhesion=(0.0,))
    table.lambda_div[:] = 1000.0
    params = _params()
    params.division_enabled = True
    engine = PlanarSimulationEngine(
        np.array([[5.0, 5.0]]),
        np.array([[1.0, 0.0]]),
        np.array([0], dtype=np.int32),
        np.zeros((1, 0)),
        table,
        params,
        rng=np.random.default_rng(43),
    )
    store = TrajectoryStore()
    engine.run(2, store=store)

    assert [positions.shape[0] for positions in store.x] == [2, 4]
    assert [ids.shape[0] for ids in store.track_id] == [2, 4]
    assert [ids.shape[0] for ids in store.parent_id] == [2, 4]
    assert len(np.unique(store.track_id[-1])) == 4
