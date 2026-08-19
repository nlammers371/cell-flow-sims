"""Mechanistic density--polarization continuum model of MIPS."""

from __future__ import annotations

import numpy as np

from ..base import ContinuumModel, ParameterSpec, PresetSpec
from ..initial_conditions import make_scalar_initial_condition


class DensityPolarizationModel(ContinuumModel):
    key = "density_polarization"
    name = "Density–polarization"
    description = "Density-dependent motility coupled to a relaxing polarization field."
    equations = (
        "∂tρ = −∇·[v(ρ)p] + Dρ∇²ρ",
        "∂tp = −Dr p − ½∇[v(ρ)ρ] + Dp∇²p + η",
        "v(ρ) = vmin + (v0−vmin)e^(−αρ)",
    )
    primary_field = "rho"
    secondary_field = "polarization"
    nonnegative_fields = ("rho",)
    conserved_fields = ("rho",)
    parameter_specs = (
        ParameterSpec("v0", "Free speed", 2.0, "Low-density motility speed.", group="Motility", symbol="v₀", minimum=0.05, maximum=10.0, scale="log", units="L/T"),
        ParameterSpec("v_min", "Speed floor", 0.05, "High-density residual motility speed.", group="Motility", symbol="vmin", minimum=0.001, maximum=2.0, scale="log", units="L/T"),
        ParameterSpec("alpha", "Slowdown", 1.0, "Strength of density-dependent motility suppression.", group="Motility", symbol="α", minimum=0.01, maximum=5.0, scale="log", units="1/ρ"),
        ParameterSpec("d_rho", "Density diffusion", 0.05, "Ordinary density diffusivity.", group="Transport", symbol="Dρ", minimum=0.001, maximum=2.0, scale="log", units="L²/T"),
        ParameterSpec("d_rotation", "Turn rate", 0.5, "Polarization relaxation rate.", group="Transport", symbol="Dr", minimum=0.01, maximum=5.0, scale="log", units="1/T"),
        ParameterSpec("d_polar", "Polar diffusion", 0.5, "Polarization diffusivity.", group="Transport", symbol="Dp", minimum=0.001, maximum=5.0, scale="log", units="L²/T"),
        ParameterSpec("polar_noise", "Polar noise", 0.0, "Nonconserved polarization noise amplitude.", group="Noise", symbol="η", minimum=0.0, maximum=1.0),
    )
    presets = (
        PresetSpec(
            "constant_speed", "Constant speed", "Density-independent speed removes the MIPS feedback.",
            {"v0": 1.0, "v_min": 1.0, "alpha": 1.0, "d_rho": 0.1, "d_rotation": 1.0, "d_polar": 0.5, "polar_noise": 0.0},
            "uniform_noise", {"mean": 1.0, "amplitude": 0.03},
        ),
        PresetSpec(
            "stable", "Weak slowing", "Weak slowdown; homogeneous density is linearly stable.",
            {"v0": 1.0, "v_min": 0.2, "alpha": 0.2, "d_rho": 0.1, "d_rotation": 1.0, "d_polar": 0.5, "polar_noise": 0.0},
            "uniform_noise", {"mean": 1.0, "amplitude": 0.03},
        ),
        PresetSpec(
            "mips", "MIPS", "Strong motility slowdown drives density aggregation.",
            {"v0": 2.0, "v_min": 0.03, "alpha": 1.0, "d_rho": 0.02, "d_rotation": 0.5, "d_polar": 0.25, "polar_noise": 0.0},
            "uniform_noise", {"mean": 2.0, "amplitude": 0.04},
        ),
        PresetSpec(
            "seeded", "Seeded cluster", "A dense seed tests whether aggregation grows or dissolves.",
            {"v0": 2.0, "v_min": 0.03, "alpha": 1.0, "d_rho": 0.02, "d_rotation": 0.5, "d_polar": 0.25, "polar_noise": 0.0},
            "droplet", {"mean": 1.2, "inside": 3.5, "radius": 6.0, "interface": 1.5},
        ),
        PresetSpec(
            "high_persistence", "High persistence", "Slow polarization decorrelation strengthens persistent transport.",
            {"v0": 2.0, "v_min": 0.03, "alpha": 1.0, "d_rho": 0.02, "d_rotation": 0.15, "d_polar": 0.25, "polar_noise": 0.0},
            "uniform_noise", {"mean": 2.0, "amplitude": 0.04},
        ),
        PresetSpec(
            "low_persistence", "Low persistence", "Rapid polarization decorrelation suppresses density organization.",
            {"v0": 2.0, "v_min": 0.03, "alpha": 1.0, "d_rho": 0.08, "d_rotation": 2.0, "d_polar": 0.5, "polar_noise": 0.0},
            "uniform_noise", {"mean": 2.0, "amplitude": 0.04},
        ),
        PresetSpec(
            "matched", "Matched field", "A common positive scalar field for cross-model comparison.",
            {"v0": 2.0, "v_min": 0.03, "alpha": 1.0, "d_rho": 0.03, "d_rotation": 0.5, "d_polar": 0.25, "polar_noise": 0.0},
            "uniform_noise", {"mean": 1.0, "amplitude": 0.03},
        ),
    )

    @staticmethod
    def speed(rho, parameters):
        v0 = float(parameters["v0"])
        v_min = float(parameters["v_min"])
        alpha = float(parameters["alpha"])
        return v_min + (v0 - v_min) * np.exp(-alpha * rho)

    def validate(self, parameters):
        warnings = []
        if float(parameters["v_min"]) > float(parameters["v0"]):
            warnings.append("Speed floor exceeds free speed, so motility rises rather than falls with density.")
        return warnings

    def initialize(self, grid, rng, parameters, initial_condition, initial_values):
        values = dict(initial_values)
        mean = float(values.get("mean", 1.0))
        rho = make_scalar_initial_condition(
            grid, rng, initial_condition, mean, float(values.get("amplitude", 0.03)),
            float(values.get("radius", 0.1 * grid.length)), int(values.get("count", 8)),
            float(values.get("inside", 2.5)), float(values.get("interface", 1.0)),
        )
        if np.min(rho) < 0.0:
            raise ValueError("The requested initial density contains negative values.")
        zeros = np.zeros_like(rho)
        return {"rho": rho, "px": zeros.copy(), "py": zeros.copy()}

    def step(self, state, grid, dt, parameters, rng, noise_strength):
        rho = state.fields["rho"]
        px = state.fields["px"]
        py = state.fields["py"]
        speed = self.speed(rho, parameters)
        rho_hat = grid.fft(rho)
        flux_div_hat = (
            1j * grid.kx * grid.filtered_fft(speed * px)
            + 1j * grid.ky * grid.filtered_fft(speed * py)
        )
        d_rho = float(parameters["d_rho"])
        updated_rho_hat = (rho_hat - dt * flux_div_hat) / (1.0 + dt * d_rho * grid.k2)
        updated_rho_hat[0, 0] = rho_hat[0, 0]

        vrho_hat = grid.filtered_fft(speed * rho)
        d_rotation = float(parameters["d_rotation"])
        d_polar = float(parameters["d_polar"])
        denominator = 1.0 + dt * (d_rotation + d_polar * grid.k2)
        px_hat = (grid.fft(px) - 0.5 * dt * 1j * grid.kx * vrho_hat) / denominator
        py_hat = (grid.fft(py) - 0.5 * dt * 1j * grid.ky * vrho_hat) / denominator
        polar_noise = float(parameters["polar_noise"]) + float(noise_strength)
        if polar_noise:
            px_hat += np.sqrt(dt) * polar_noise * grid.filtered_fft(rng.standard_normal(rho.shape)) / denominator
            py_hat += np.sqrt(dt) * polar_noise * grid.filtered_fft(rng.standard_normal(rho.shape)) / denominator
        return {"rho": grid.ifft(updated_rho_hat), "px": grid.ifft(px_hat), "py": grid.ifft(py_hat)}

    def derived_fields(self, state, grid, parameters):
        px, py = state.fields["px"], state.fields["py"]
        return {
            "polarization": np.sqrt(px * px + py * py),
            "speed": self.speed(state.fields["rho"], parameters),
        }

    def model_diagnostics(self, state, grid, parameters):
        rho_bar = float(np.mean(state.fields["rho"]))
        v0 = float(parameters["v0"])
        v_min = float(parameters["v_min"])
        alpha = float(parameters["alpha"])
        exponential = np.exp(-alpha * rho_bar)
        speed = v_min + (v0 - v_min) * exponential
        criterion = 1.0 - alpha * rho_bar * (v0 - v_min) * exponential / speed
        return {"mips_criterion": float(criterion), "mean_speed": float(speed)}
