"""Passive conserved phase separation (Cahn--Hilliard / Model B)."""

from __future__ import annotations

import numpy as np

from ..base import ContinuumModel, ParameterSpec, PresetSpec
from ..initial_conditions import make_scalar_initial_condition
from ..numerics.grid import conserved_noise_modes


MODEL_B_PARAMETERS = (
    ParameterSpec(
        "mobility", "Mobility", 1.0, "Conserved relaxation rate M.",
        group="Dynamics", symbol="M", minimum=0.01, maximum=10.0, scale="log", units="L^2/(E T)",
    ),
    ParameterSpec(
        "a", "Bulk a", -1.0, "Quadratic free-energy coefficient; a < 0 permits demixing.",
        group="Free energy", symbol="a", minimum=-2.0, maximum=2.0,
    ),
    ParameterSpec(
        "b", "Bulk b", 1.0, "Quartic coefficient; must remain positive.",
        group="Free energy", symbol="b", minimum=0.05, maximum=5.0, scale="log",
    ),
    ParameterSpec(
        "kappa", "Interface cost", 1.0, "Gradient-energy coefficient controlling interface width and tension.",
        group="Free energy", symbol="κ", minimum=0.02, maximum=10.0, scale="log", units="E L^2",
    ),
)


MODEL_B_PRESETS = (
    PresetSpec(
        "stable", "Stable uniform", "A convex bulk free energy; fluctuations decay.",
        {"a": 1.0, "b": 1.0, "kappa": 1.0, "mobility": 1.0},
        "uniform_noise", {"mean": 0.0, "amplitude": 0.08},
    ),
    PresetSpec(
        "spinodal", "Spinodal", "A homogeneous state inside the spinodal region; fluctuations grow.",
        {"a": -1.0, "b": 1.0, "kappa": 1.0, "mobility": 1.0},
        "uniform_noise", {"mean": 0.0, "amplitude": 0.03},
    ),
    PresetSpec(
        "nucleation", "Nucleation", "A finite droplet in the metastable region.",
        {"a": -1.0, "b": 1.0, "kappa": 1.0, "mobility": 1.0},
        "droplet", {"mean": -0.80, "inside": 1.0, "radius": 7.0, "interface": 1.2},
    ),
    PresetSpec(
        "multi_droplet", "Many droplets", "Several droplets coarsen by diffusion and merger.",
        {"a": -1.0, "b": 1.0, "kappa": 1.0, "mobility": 1.0},
        "multiple_droplets",
        {"mean": -0.65, "inside": 1.0, "radius": 3.5, "count": 10, "interface": 1.0},
    ),
    PresetSpec(
        "interface", "Flat interface", "Two periodic domains separated by nearly flat interfaces.",
        {"a": -1.0, "b": 1.0, "kappa": 1.0, "mobility": 1.0},
        "single_interface", {"mean": 0.0, "amplitude": 1.0, "interface": 1.0},
    ),
)


def model_b_step(phi, grid, dt, parameters, rng, noise_strength, active_lambda=0.0):
    """Shared semi-implicit update used by both passive and active Model B."""

    mobility = float(parameters["mobility"])
    a = float(parameters["a"])
    b = float(parameters["b"])
    kappa = float(parameters["kappa"])
    phi_hat = grid.fft(phi)

    nonlinear_mu = b * phi * phi * phi
    if active_lambda != 0.0:
        grad_x, grad_y = grid.gradient(phi)
        nonlinear_mu = nonlinear_mu + active_lambda * (grad_x * grad_x + grad_y * grad_y)
    mu_explicit_hat = a * phi_hat + grid.filtered_fft(nonlinear_mu)
    numerator = phi_hat - dt * mobility * grid.k2 * mu_explicit_hat
    if noise_strength:
        numerator += np.sqrt(dt) * conserved_noise_modes(grid, rng, noise_strength)
    updated = numerator / (1.0 + dt * mobility * kappa * grid.k4)
    updated[0, 0] = phi_hat[0, 0]
    return grid.ifft(updated)


def chemical_potential(phi, grid, parameters, active_lambda=0.0):
    a = float(parameters["a"])
    b = float(parameters["b"])
    kappa = float(parameters["kappa"])
    mu = a * phi + b * phi ** 3 - kappa * grid.laplacian(phi)
    if active_lambda != 0.0:
        grad_x, grad_y = grid.gradient(phi)
        mu = mu + active_lambda * (grad_x * grad_x + grad_y * grad_y)
    return mu


def passive_free_energy(phi, grid, parameters):
    a = float(parameters["a"])
    b = float(parameters["b"])
    kappa = float(parameters["kappa"])
    grad_x, grad_y = grid.gradient(phi)
    density = 0.5 * a * phi * phi + 0.25 * b * phi ** 4
    density += 0.5 * kappa * (grad_x * grad_x + grad_y * grad_y)
    return float(np.sum(density) * grid.dx * grid.dx)


class ModelB(ContinuumModel):
    key = "model_b"
    name = "Passive Model B"
    description = "Conserved relaxational dynamics of a scalar phase field."
    equations = (
        "∂tφ = M∇²μ + ∇·Λ",
        "μ = aφ + bφ³ − κ∇²φ",
        "F = ∫[aφ²/2 + bφ⁴/4 + κ|∇φ|²/2] dA",
    )
    primary_field = "phi"
    secondary_field = "mu"
    conserved_fields = ("phi",)
    parameter_specs = MODEL_B_PARAMETERS
    presets = MODEL_B_PRESETS

    def validate(self, parameters):
        warnings = []
        if float(parameters["b"]) <= 0.0:
            warnings.append("Bulk b must be positive for a bounded free energy.")
        if float(parameters["mobility"]) <= 0.0 or float(parameters["kappa"]) <= 0.0:
            warnings.append("Mobility and interface cost must be positive.")
        return warnings

    def initialize(self, grid, rng, parameters, initial_condition, initial_values):
        values = dict(initial_values)
        mean = float(values.get("mean", 0.0))
        amplitude = float(values.get("amplitude", 0.03))
        radius = float(values.get("radius", 0.12 * grid.length))
        count = int(values.get("count", 8))
        inside = float(values.get("inside", 1.0))
        interface = float(values.get("interface", 1.0))
        phi = make_scalar_initial_condition(
            grid, rng, initial_condition, mean, amplitude, radius, count, inside, interface
        )
        return {"phi": phi}

    def step(self, state, grid, dt, parameters, rng, noise_strength):
        return {
            "phi": model_b_step(
                state.fields["phi"], grid, dt, parameters, rng, noise_strength, active_lambda=0.0
            )
        }

    def derived_fields(self, state, grid, parameters):
        return {"mu": chemical_potential(state.fields["phi"], grid, parameters)}

    def model_diagnostics(self, state, grid, parameters):
        a, b = float(parameters["a"]), float(parameters["b"])
        binodal = np.sqrt(-a / b) if a < 0.0 and b > 0.0 else float("nan")
        spinodal = np.sqrt(-a / (3.0 * b)) if a < 0.0 and b > 0.0 else float("nan")
        return {
            "free_energy": passive_free_energy(state.fields["phi"], grid, parameters),
            "binodal_magnitude": float(binodal),
            "spinodal_magnitude": float(spinodal),
        }
