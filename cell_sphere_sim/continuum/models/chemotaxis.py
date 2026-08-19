"""Keller--Segel-style autochemotaxis with optional crowding and growth."""

from __future__ import annotations

import numpy as np

from ..base import ContinuumModel, ParameterSpec, PresetSpec
from ..initial_conditions import make_scalar_initial_condition
from ..numerics.grid import conserved_noise_modes


class KellerSegelModel(ContinuumModel):
    key = "keller_segel"
    name = "Autochemotaxis"
    description = "Cells secrete, sense, and move up gradients of a diffusing signal."
    equations = (
        "∂tρ = Dρ∇²ρ − χ∇·(ρ∇c) + Dcrowd∇²(ρᵐ) + R(ρ)",
        "∂tc = Dc∇²c + αcρ − kcc",
    )
    primary_field = "rho"
    secondary_field = "c"
    nonnegative_fields = ("rho", "c")
    conserved_fields = ("rho",)
    parameter_specs = (
        ParameterSpec("d_rho", "Cell diffusion", 0.1, "Random cell-density diffusivity.", group="Cells", symbol="Dρ", minimum=0.001, maximum=5.0, scale="log", units="L²/T"),
        ParameterSpec("chemotaxis", "Chemotaxis", 0.5, "Up-gradient response coefficient.", group="Cells", symbol="χ", minimum=0.0, maximum=10.0, scale="linear", stability_warning="Strong attraction can concentrate density and force timestep reduction."),
        ParameterSpec("d_crowd", "Crowding", 0.05, "Nonlinear density-diffusion coefficient.", group="Cells", symbol="Dcrowd", minimum=0.0, maximum=5.0, stability_warning="Zero crowding permits classical Keller–Segel blow-up."),
        ParameterSpec("crowd_power", "Crowd power", 2.0, "Exponent m in the nonlinear crowding flux.", group="Cells", symbol="m", minimum=1.0, maximum=4.0),
        ParameterSpec("signal_diffusion", "Signal diffusion", 1.0, "Signal diffusivity.", group="Signal", symbol="Dc", minimum=0.01, maximum=20.0, scale="log", units="L²/T"),
        ParameterSpec("secretion", "Secretion", 1.0, "Signal production per unit density.", group="Signal", symbol="αc", minimum=0.0, maximum=5.0),
        ParameterSpec("signal_decay", "Signal decay", 0.2, "First-order signal decay rate.", group="Signal", symbol="kc", minimum=0.001, maximum=5.0, scale="log", units="1/T"),
        ParameterSpec("signal_mode", "Signal mode", "dynamic", "Evolve the signal or solve its quasistatic screened-Poisson equation.", group="Signal", choices=("dynamic", "quasistatic"), requires_reset=True),
        ParameterSpec("growth_rate", "Growth rate", 0.0, "Logistic population growth; zero by default.", group="Growth", symbol="r", minimum=0.0, maximum=2.0, units="1/T"),
        ParameterSpec("capacity", "Capacity", 4.0, "Carrying density for optional logistic growth.", group="Growth", symbol="K", minimum=0.1, maximum=20.0, scale="log"),
    )
    presets = (
        PresetSpec(
            "diffusion", "Diffusion only", "Chemotaxis and growth disabled; density diffuses exactly.",
            {"d_rho": 0.15, "chemotaxis": 0.0, "d_crowd": 0.0, "crowd_power": 2.0, "signal_diffusion": 1.0, "secretion": 1.0, "signal_decay": 0.2, "signal_mode": "dynamic", "growth_rate": 0.0, "capacity": 4.0},
            "uniform_noise", {"mean": 1.0, "amplitude": 0.1},
        ),
        PresetSpec(
            "weak", "Weak response", "Signal gradients form but density remains diffuse.",
            {"d_rho": 0.15, "chemotaxis": 0.15, "d_crowd": 0.08, "crowd_power": 2.0, "signal_diffusion": 1.5, "secretion": 0.8, "signal_decay": 0.4, "signal_mode": "dynamic", "growth_rate": 0.0, "capacity": 4.0},
            "uniform_noise", {"mean": 1.0, "amplitude": 0.04},
        ),
        PresetSpec(
            "aggregation", "Aggregation", "Chemotactic attraction overcomes diffusion, regularized by crowding.",
            {"d_rho": 0.08, "chemotaxis": 0.6, "d_crowd": 0.3, "crowd_power": 2.0, "signal_diffusion": 1.0, "secretion": 1.0, "signal_decay": 0.5, "signal_mode": "dynamic", "growth_rate": 0.0, "capacity": 4.0},
            "uniform_noise", {"mean": 1.0, "amplitude": 0.10},
        ),
        PresetSpec(
            "quasistatic", "Fast signal", "The signal is eliminated quasistatically at every step.",
            {"d_rho": 0.08, "chemotaxis": 0.6, "d_crowd": 0.3, "crowd_power": 2.0, "signal_diffusion": 1.0, "secretion": 1.0, "signal_decay": 0.5, "signal_mode": "quasistatic", "growth_rate": 0.0, "capacity": 4.0},
            "multiple_droplets", {"mean": 1.0, "inside": 1.8, "radius": 3.0, "count": 8, "interface": 1.0},
        ),
        PresetSpec(
            "multiple_peaks", "Merging peaks", "Multiple density peaks interact and merge through a dynamic signal.",
            {"d_rho": 0.08, "chemotaxis": 0.6, "d_crowd": 0.3, "crowd_power": 2.0, "signal_diffusion": 1.0, "secretion": 1.0, "signal_decay": 0.5, "signal_mode": "dynamic", "growth_rate": 0.0, "capacity": 4.0},
            "multiple_droplets", {"mean": 1.0, "inside": 1.8, "radius": 2.5, "count": 10, "interface": 1.0},
        ),
        PresetSpec(
            "long_range", "Long range", "A large signal length couples distant density peaks.",
            {"d_rho": 0.08, "chemotaxis": 0.3, "d_crowd": 0.3, "crowd_power": 2.0, "signal_diffusion": 4.0, "secretion": 1.0, "signal_decay": 0.2, "signal_mode": "quasistatic", "growth_rate": 0.0, "capacity": 4.0},
            "multiple_droplets", {"mean": 1.0, "inside": 1.6, "radius": 2.5, "count": 10, "interface": 1.0},
        ),
        PresetSpec(
            "short_range", "Short range", "Rapid decay and slow signal diffusion localize communication.",
            {"d_rho": 0.08, "chemotaxis": 0.6, "d_crowd": 0.3, "crowd_power": 2.0, "signal_diffusion": 0.25, "secretion": 1.0, "signal_decay": 1.0, "signal_mode": "quasistatic", "growth_rate": 0.0, "capacity": 4.0},
            "multiple_droplets", {"mean": 1.0, "inside": 1.6, "radius": 2.5, "count": 10, "interface": 1.0},
        ),
        PresetSpec(
            "classical", "Classical collapse", "Unregularized Keller–Segel attraction; intentionally blow-up prone.",
            {"d_rho": 0.08, "chemotaxis": 0.3, "d_crowd": 0.0, "crowd_power": 2.0, "signal_diffusion": 1.0, "secretion": 1.0, "signal_decay": 0.5, "signal_mode": "quasistatic", "growth_rate": 0.0, "capacity": 4.0},
            "uniform_noise", {"mean": 1.0, "amplitude": 0.02},
        ),
    )

    def validate(self, parameters):
        warnings = []
        if float(parameters["chemotaxis"]) > 0.0 and float(parameters["d_crowd"]) == 0.0:
            warnings.append("Chemotaxis without crowding can approach Keller–Segel collapse and require very small dt.")
        if float(parameters["signal_decay"]) <= 0.0 and parameters["signal_mode"] == "quasistatic":
            warnings.append("Quasistatic signal requires positive decay to define its zero mode.")
        return warnings

    def _quasistatic_signal(self, rho, grid, parameters):
        denominator = float(parameters["signal_decay"]) + float(parameters["signal_diffusion"]) * grid.k2
        if denominator[0, 0] == 0.0:
            raise ValueError("Quasistatic signal needs positive signal decay.")
        return grid.ifft(float(parameters["secretion"]) * grid.fft(rho) / denominator)

    def initialize(self, grid, rng, parameters, initial_condition, initial_values):
        values = dict(initial_values)
        mean = float(values.get("mean", 1.0))
        rho = make_scalar_initial_condition(
            grid, rng, initial_condition, mean, float(values.get("amplitude", 0.03)),
            float(values.get("radius", 0.1 * grid.length)), int(values.get("count", 8)),
            float(values.get("inside", 2.0)), float(values.get("interface", 1.0)),
        )
        if np.min(rho) < 0.0:
            raise ValueError("The requested initial density contains negative values.")
        if parameters["signal_mode"] == "quasistatic":
            signal = self._quasistatic_signal(rho, grid, parameters)
        else:
            equilibrium_mean = float(parameters["secretion"]) * mean / float(parameters["signal_decay"])
            signal = np.full_like(rho, equilibrium_mean)
        return {"rho": rho, "c": signal}

    def step(self, state, grid, dt, parameters, rng, noise_strength):
        rho, signal = state.fields["rho"], state.fields["c"]
        if parameters["signal_mode"] == "quasistatic":
            signal = self._quasistatic_signal(rho, grid, parameters)
        grad_cx, grad_cy = grid.gradient(signal)
        chemotaxis_hat = -float(parameters["chemotaxis"]) * (
            1j * grid.kx * grid.filtered_fft(rho * grad_cx)
            + 1j * grid.ky * grid.filtered_fft(rho * grad_cy)
        )
        crowd_power = float(parameters["crowd_power"])
        d_crowd = float(parameters["d_crowd"])
        # Stabilized semi-implicit split of ∇²(ρ^m).  Taking the stabilizer as
        # the largest local derivative keeps the stiff crowding regularizer on
        # the implicit side while retaining the requested nonlinear flux.
        crowd_stabilizer = 0.0
        if d_crowd > 0.0:
            crowd_stabilizer = crowd_power * max(float(np.max(rho)), 0.0) ** (crowd_power - 1.0)
        crowd_residual = rho ** crowd_power - crowd_stabilizer * rho
        crowd_hat = -d_crowd * grid.k2 * grid.filtered_fft(crowd_residual)
        growth_rate = float(parameters["growth_rate"])
        growth = growth_rate * rho * (1.0 - rho / float(parameters["capacity"]))
        rho_hat = grid.fft(rho)
        updated_rho_hat = (
            rho_hat + dt * (chemotaxis_hat + crowd_hat + grid.filtered_fft(growth))
        ) / (1.0 + dt * (float(parameters["d_rho"]) + d_crowd * crowd_stabilizer) * grid.k2)
        if noise_strength:
            updated_rho_hat += (
                np.sqrt(dt) * conserved_noise_modes(grid, rng, noise_strength)
                / (1.0 + dt * (float(parameters["d_rho"]) + d_crowd * crowd_stabilizer) * grid.k2)
            )
        if growth_rate == 0.0:
            updated_rho_hat[0, 0] = rho_hat[0, 0]
        updated_rho = grid.ifft(updated_rho_hat)

        if parameters["signal_mode"] == "quasistatic":
            updated_signal = self._quasistatic_signal(updated_rho, grid, parameters)
        else:
            signal_hat = grid.fft(signal)
            denominator = 1.0 + dt * (
                float(parameters["signal_decay"]) + float(parameters["signal_diffusion"]) * grid.k2
            )
            updated_signal = grid.ifft(
                (signal_hat + dt * float(parameters["secretion"]) * updated_rho_hat) / denominator
            )
        return {"rho": updated_rho, "c": updated_signal}

    def derived_fields(self, state, grid, parameters):
        grad_x, grad_y = grid.gradient(state.fields["c"])
        return {"signal_gradient": np.sqrt(grad_x * grad_x + grad_y * grad_y)}

    def model_diagnostics(self, state, grid, parameters):
        decay = float(parameters["signal_decay"])
        signal_length = np.inf if decay == 0.0 else np.sqrt(float(parameters["signal_diffusion"]) / decay)
        return {"signal_length": float(signal_length)}
