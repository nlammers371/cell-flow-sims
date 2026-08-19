"""Render-independent continuum simulation engine."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .base import ContinuumState
from .config import ContinuumConfig
from .diagnostics import scalar_diagnostics
from .numerics import SpectralGrid
from .registry import create_model


class StepRejected(RuntimeError):
    """Raised when a positive-density update remains invalid at the minimum dt."""


class ContinuumEngine:
    """Own a model, numerical workspace, RNG, state, and diagnostic history."""

    def __init__(self, config: Optional[ContinuumConfig] = None, initial_scalar=None):
        self.config = config if config is not None else ContinuumConfig()
        self.model = create_model(self.config.model)
        self.grid = SpectralGrid(self.config.grid_size, self.config.domain_size)
        self.rng = np.random.default_rng(self.config.seed)
        self.parameters, self.preset, initial_condition, initial_values = self._resolved_setup()
        self.initial_condition = initial_condition
        self.initial_values = dict(initial_values)
        self.warnings = list(self.model.validate(self.parameters))
        fields = self.model.initialize(
            self.grid, self.rng, self.parameters, initial_condition, initial_values
        )
        if initial_scalar is not None:
            scalar = np.asarray(initial_scalar, dtype=np.float64)
            expected = (self.grid.size, self.grid.size)
            if scalar.shape != expected:
                raise ValueError(f"initial_scalar has shape {scalar.shape}; expected {expected}")
            if self.model.primary_field in self.model.nonnegative_fields and np.min(scalar) < 0.0:
                raise ValueError("initial_scalar contains negative density")
            fields[self.model.primary_field] = scalar.copy()
            # Recompute fields algebraically slaved to the primary field.
            if self.model.key == "keller_segel" and self.parameters["signal_mode"] == "quasistatic":
                fields["c"] = self.model._quasistatic_signal(scalar, self.grid, self.parameters)
        self.state = ContinuumState(fields=fields)
        self.current_dt = float(self.config.dt)
        self.initial_masses = {
            key: float(np.sum(fields[key]) * self.grid.dx * self.grid.dx)
            for key in self.model.conserved_fields
        }
        self.corrections = []
        self.history = []
        self.record_diagnostics()

    def _resolved_setup(self):
        preset_key = self.config.preset or self.model.presets[0].key
        preset = self.model.preset(preset_key)
        parameters = self.model.defaults
        parameters.update(dict(preset.parameters))
        parameters.update(dict(self.config.parameters))
        initial_values = dict(preset.initial_values)
        if self.config.mean is not None:
            initial_values["mean"] = self.config.mean
        if self.config.initial_amplitude is not None:
            initial_values["amplitude"] = self.config.initial_amplitude
        if self.config.droplet_radius is not None:
            initial_values["radius"] = self.config.droplet_radius
        if self.config.droplet_count is not None:
            initial_values["count"] = self.config.droplet_count
        initial_condition = self.config.initial_condition or preset.initial_condition
        return parameters, preset, initial_condition, initial_values

    def set_parameter(self, key: str, value: Any) -> None:
        specs = {spec.key: spec for spec in self.model.parameter_specs}
        if key not in specs:
            raise KeyError(f"{key!r} is not a parameter of {self.model.name}")
        if specs[key].requires_reset:
            raise ValueError(f"{specs[key].name} requires reset")
        candidate = dict(self.parameters)
        candidate[key] = value
        self.parameters = candidate
        self.warnings.extend(self.model.validate(candidate))

    def _negative_violation(self, fields: Mapping[str, np.ndarray]):
        for key in self.model.nonnegative_fields:
            minimum = float(np.min(fields[key]))
            if not np.isfinite(minimum):
                return key, minimum
            if minimum < -self.config.negative_tolerance:
                return key, minimum
        return None

    def _correct_roundoff(self, fields: Dict[str, np.ndarray]) -> None:
        for key in self.model.nonnegative_fields:
            field = fields[key]
            minimum = float(np.min(field))
            if minimum >= 0.0:
                continue
            old_sum = float(np.sum(field))
            np.maximum(field, 0.0, out=field)
            new_sum = float(np.sum(field))
            if old_sum > 0.0 and new_sum > 0.0:
                field *= old_sum / new_sum
            self.corrections.append(
                {"step": self.state.step + 1, "field": key, "minimum": minimum, "kind": "roundoff"}
            )

    def step(self, count: int = 1) -> ContinuumState:
        for _ in range(int(count)):
            self._step_once()
        return self.state

    def _step_once(self) -> None:
        dt = self.current_dt
        rng_state = copy.deepcopy(self.rng.bit_generator.state)
        violation = None
        energy_violation = None
        for attempt in range(self.config.max_step_retries + 1):
            self.rng.bit_generator.state = copy.deepcopy(rng_state)
            with np.errstate(over="ignore", invalid="ignore"):
                fields = self.model.step(
                    self.state, self.grid, dt, self.parameters, self.rng,
                    float(self.config.dynamic_noise),
                )
            violation = self._negative_violation(fields)
            finite = all(np.all(np.isfinite(field)) for field in fields.values())
            energy_violation = None
            if finite and not self.config.dynamic_noise:
                old_model_values = self.model.model_diagnostics(self.state, self.grid, self.parameters)
                if "free_energy" in old_model_values:
                    trial_state = ContinuumState(fields=fields)
                    new_energy = self.model.model_diagnostics(
                        trial_state, self.grid, self.parameters
                    )["free_energy"]
                    old_energy = old_model_values["free_energy"]
                    tolerance = 1.0e-12 * max(1.0, abs(old_energy))
                    if new_energy > old_energy + tolerance:
                        energy_violation = (old_energy, new_energy)
            if violation is None and finite and energy_violation is None:
                self._correct_roundoff(fields)
                self.state = ContinuumState(
                    fields=fields,
                    time=self.state.time + dt,
                    step=self.state.step + 1,
                    metadata=dict(self.state.metadata),
                )
                self.current_dt = dt
                return
            if not self.config.adaptive_dt or dt * 0.5 < self.config.minimum_dt:
                break
            dt *= 0.5
            reason = "free-energy increase" if energy_violation is not None else "invalid density"
            self.warnings.append(
                f"Step {self.state.step + 1}: rejected {reason} and reduced dt to {dt:.3g}."
            )
        if energy_violation is not None:
            detail = f"free energy increased from {energy_violation[0]:.6g} to {energy_violation[1]:.6g}"
        else:
            detail = "non-finite values" if violation is None else f"{violation[0]} minimum {violation[1]:.3g}"
        raise StepRejected(f"Could not produce a valid update ({detail}) at dt={dt:.3g}")

    def advance_frame(self, substeps: Optional[int] = None) -> ContinuumState:
        self.step(self.config.substeps_per_frame if substeps is None else substeps)
        self.record_diagnostics()
        return self.state

    def diagnostics(self) -> Dict[str, float]:
        field = self.state.fields[self.model.primary_field]
        if self.config.cluster_threshold is None:
            threshold = float(np.mean(field))
        else:
            threshold = float(self.config.cluster_threshold)
        initial_mass = self.initial_masses.get(
            self.model.primary_field,
            float(np.sum(field) * self.grid.dx * self.grid.dx),
        )
        result = scalar_diagnostics(field, self.grid, initial_mass, threshold)
        result.update(self.model.model_diagnostics(self.state, self.grid, self.parameters))
        result.update({"time": float(self.state.time), "step": int(self.state.step), "dt": self.current_dt})
        return result

    def record_diagnostics(self) -> Dict[str, float]:
        values = self.diagnostics()
        self.history.append(values)
        return values

    def reset(self, **changes) -> "ContinuumEngine":
        config_values = self.config.to_dict()
        config_values.update(changes)
        return ContinuumEngine(ContinuumConfig(**config_values))

    def clone_config(self, **changes) -> ContinuumConfig:
        return replace(self.config, **changes)

    def derived_fields(self):
        return self.model.derived_fields(self.state, self.grid, self.parameters)
