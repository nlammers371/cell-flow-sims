from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cell_sphere_sim.continuum import (
    ContinuumConfig,
    ContinuumEngine,
    ContinuumWorkbenchController,
    create_model,
    model_keys,
)
from cell_sphere_sim.continuum.comparison import ContinuumComparison
from cell_sphere_sim.continuum.diagnostics import periodic_clusters
from cell_sphere_sim.continuum.sweep import SweepRunner, SweepSpec


def config(model="model_b", preset=None, **changes):
    values = {
        "model": model,
        "preset": preset,
        "grid_size": 32,
        "domain_size": 32.0,
        "dt": 0.01,
        "seed": 7,
    }
    values.update(changes)
    return ContinuumConfig(**values)


def test_passive_mass_and_free_energy():
    engine = ContinuumEngine(config(preset="spinodal", dt=0.02))
    energies = [engine.diagnostics()["free_energy"]]
    for _ in range(200):
        engine.step()
        energies.append(engine.diagnostics()["free_energy"])
    assert abs(engine.diagnostics()["mass_error"]) < 1e-12
    assert np.max(np.diff(energies)) <= 1e-10


def test_active_zero_is_exact_passive_limit():
    passive = ContinuumEngine(config("model_b", "spinodal"))
    active = ContinuumEngine(config("active_model_b", "passive_limit"))
    assert np.array_equal(passive.state.fields["phi"], active.state.fields["phi"])
    passive.step(20)
    active.step(20)
    assert np.array_equal(passive.state.fields["phi"], active.state.fields["phi"])
    assert "free_energy" in active.diagnostics()


def test_active_energy_not_misreported():
    engine = ContinuumEngine(config("active_model_b", "active_demix"))
    assert "free_energy" not in engine.diagnostics()


def test_stable_and_spinodal_presets_move_variance_in_opposite_directions():
    stable = ContinuumEngine(config(preset="stable", dt=0.03))
    spinodal = ContinuumEngine(config(preset="spinodal", dt=0.02))
    stable_initial = stable.diagnostics()["variance"]
    spinodal_initial = spinodal.diagnostics()["variance"]
    stable.step(400)
    spinodal.step(500)
    assert stable.diagnostics()["variance"] < 0.02 * stable_initial
    assert spinodal.diagnostics()["variance"] > 2.0 * spinodal_initial


def test_subcritical_droplet_dissolves_and_supercritical_survives():
    small = ContinuumEngine(config(preset="nucleation", droplet_radius=1.5, dt=0.02))
    large = ContinuumEngine(config(preset="nucleation", droplet_radius=5.0, dt=0.02))
    small.step(1000)
    large.step(1000)
    center = small.grid.size // 2
    assert small.state.fields["phi"][center, center] < 0.0
    assert large.state.fields["phi"][center, center] > 0.5


def test_mips_mass_stability_and_instability_criterion():
    stable = ContinuumEngine(config("density_polarization", "constant_speed"))
    unstable = ContinuumEngine(config("density_polarization", "mips"))
    stable_variance = stable.diagnostics()["variance"]
    unstable_variance = unstable.diagnostics()["variance"]
    stable.step(500)
    unstable.step(5000)
    assert stable.diagnostics()["variance"] < stable_variance
    assert unstable.diagnostics()["mips_criterion"] < 0.0
    assert unstable.diagnostics()["variance"] > unstable_variance
    assert abs(unstable.diagnostics()["mass_error"]) < 1e-12


def test_keller_segel_diffusion_limit_matches_spectral_update():
    engine = ContinuumEngine(config("keller_segel", "diffusion", dt=0.02))
    initial = engine.state.fields["rho"].copy()
    expected = engine.grid.ifft(
        engine.grid.fft(initial)
        / (1.0 + engine.config.dt * engine.parameters["d_rho"] * engine.grid.k2)
    )
    engine.step()
    assert np.allclose(engine.state.fields["rho"], expected, rtol=0.0, atol=2e-15)
    assert abs(engine.diagnostics()["mass_error"]) < 1e-12


@pytest.mark.parametrize("model,preset", [
    ("model_b", "spinodal"),
    ("active_model_b", "active_demix"),
    ("density_polarization", "stable"),
    ("keller_segel", "weak"),
])
def test_seeded_runs_are_deterministic(model, preset):
    first = ContinuumEngine(config(model, preset, dynamic_noise=0.02))
    second = ContinuumEngine(config(model, preset, dynamic_noise=0.02))
    first.step(30)
    second.step(30)
    for key in first.state.fields:
        assert np.array_equal(first.state.fields[key], second.state.fields[key])


def test_render_grouping_does_not_change_trajectory():
    direct = ContinuumEngine(config(substeps_per_frame=7))
    grouped = ContinuumEngine(config(substeps_per_frame=7))
    direct.step(70)
    for _ in range(10):
        grouped.advance_frame()
    assert np.array_equal(direct.state.fields["phi"], grouped.state.fields["phi"])
    assert direct.state.time == grouped.state.time


@pytest.mark.parametrize("model,preset", [
    ("model_b", "multi_droplet"),
    ("active_model_b", "active_droplet"),
    ("density_polarization", "mips"),
    ("keller_segel", "aggregation"),
])
def test_representative_long_runs_remain_finite(model, preset):
    engine = ContinuumEngine(config(model, preset, dt=0.01))
    engine.step(1000)
    assert all(np.all(np.isfinite(value)) for value in engine.state.fields.values())
    assert abs(engine.diagnostics()["mass_error"]) < 1e-8


def test_controller_run_pause_step_reset_and_live_parameter():
    controller = ContinuumWorkbenchController(config())
    controller.tick()
    assert controller.engine.state.step == 0
    controller.step()
    assert controller.engine.state.step == controller.engine.config.substeps_per_frame
    controller.update_parameter("mobility", 2.0)
    assert controller.engine.parameters["mobility"] == 2.0
    controller.toggle_running()
    controller.tick()
    assert controller.engine.state.step == 2 * controller.engine.config.substeps_per_frame
    controller.pause()
    controller.reset()
    assert controller.engine.state.step == 0
    assert controller.engine.parameters["mobility"] == 2.0


def test_model_switch_preserves_each_models_parameters():
    controller = ContinuumWorkbenchController(config())
    controller.update_parameter("mobility", 2.5)
    controller.switch_model("density_polarization")
    controller.update_parameter("v0", 3.0)
    controller.switch_model("model_b")
    assert controller.engine.parameters["mobility"] == 2.5
    controller.switch_model("density_polarization")
    assert controller.engine.parameters["v0"] == 3.0


def test_controller_export_writes_reproducible_bundle(tmp_path):
    controller = ContinuumWorkbenchController(config())
    controller.step()
    paths = controller.export(tmp_path)
    assert set(paths) == {"metadata", "diagnostics", "arrays"}
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["model"] == "model_b"
    assert metadata["initial_condition"] == "uniform_noise"
    assert Path(paths["arrays"]).is_file()


def test_mips_sweep_can_vary_mean_density_and_exports_uncertainty(tmp_path):
    runner = SweepRunner(
        config("density_polarization", "mips", grid_size=16, domain_size=16.0),
        SweepSpec("mean", [0.75, 1.5], "v0", [1.0], [1, 2], 2),
    )
    rows = runner.run()
    assert {row["mean"] for row in rows} == {0.75, 1.5}
    for row in rows:
        assert row["mass"] / (16.0 * 16.0) == pytest.approx(row["mean"])
    paths = runner.export(tmp_path)
    assert paths["summary"].is_file()
    header = paths["summary"].read_text(encoding="utf-8").splitlines()[0]
    assert "variance_std" in header


def test_all_parameter_labels_are_short_and_controls_are_declared():
    for key in model_keys():
        model = create_model(key)
        assert model.parameter_specs
        assert model.presets
        assert all(1 <= len(spec.name.split()) <= 3 for spec in model.parameter_specs)


def test_periodic_cluster_merges_across_boundaries():
    field = np.zeros((8, 8))
    field[3, 0] = 1.0
    field[3, -1] = 1.0
    count, largest = periodic_clusters(field, 0.5)
    assert count == 1
    assert largest == 2 / 64


def test_comparison_uses_identical_initial_scalar():
    comparison = ContinuumComparison([
        config("model_b", "spinodal"),
        config("active_model_b", "active_demix"),
    ])
    first, second = comparison.engines
    assert np.array_equal(first.state.fields["phi"], second.state.fields["phi"])


def test_saved_regression_trajectories():
    path = Path(__file__).parent / "data" / "continuum_regression.json"
    references = json.loads(path.read_text(encoding="utf-8"))
    for model, reference in references.items():
        engine = ContinuumEngine(ContinuumConfig(
            model=model,
            preset=reference["preset"],
            grid_size=16,
            domain_size=16.0,
            dt=0.01,
            seed=17,
        ))
        for row in reference["trajectory"]:
            engine.step(row["step"] - engine.state.step)
            field = engine.state.fields[engine.model.primary_field]
            assert np.mean(field) == pytest.approx(row["mean"], abs=2e-15)
            assert np.var(field) == pytest.approx(row["variance"], rel=2e-13, abs=2e-15)
            assert np.sum(field * field) == pytest.approx(row["l2"], rel=2e-13, abs=2e-15)
