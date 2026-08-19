"""Base types shared by all continuum models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ParameterSpec:
    """Declarative description of one user-facing model parameter."""

    key: str
    name: str
    default: Any
    description: str
    group: str = "Model"
    symbol: str = ""
    units: str = "dimensionless"
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    scale: str = "linear"
    choices: Tuple[Any, ...] = ()
    requires_reset: bool = False
    stability_warning: str = ""


@dataclass(frozen=True)
class PresetSpec:
    """Named, scientifically motivated collection of parameter overrides."""

    key: str
    name: str
    description: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    initial_condition: str = "uniform_noise"
    initial_values: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ContinuumState:
    """Mutable simulation state, with one real array per continuum field."""

    fields: Dict[str, np.ndarray]
    time: float = 0.0
    step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "ContinuumState":
        return ContinuumState(
            fields={key: value.copy() for key, value in self.fields.items()},
            time=float(self.time),
            step=int(self.step),
            metadata=dict(self.metadata),
        )


class ContinuumModel(ABC):
    """Interface implemented by a continuum PDE model."""

    key: str
    name: str
    description: str
    equations: Sequence[str]
    equation_version: str = "1"
    primary_field: str
    secondary_field: Optional[str] = None
    nonnegative_fields: Tuple[str, ...] = ()
    conserved_fields: Tuple[str, ...] = ()
    parameter_specs: Tuple[ParameterSpec, ...] = ()
    presets: Tuple[PresetSpec, ...] = ()

    @property
    def defaults(self) -> Dict[str, Any]:
        return {spec.key: spec.default for spec in self.parameter_specs}

    def preset(self, key: str) -> PresetSpec:
        for preset in self.presets:
            if preset.key == key:
                return preset
        available = ", ".join(p.key for p in self.presets)
        raise KeyError(f"Unknown {self.key} preset {key!r}; choose from {available}")

    def validate(self, parameters: Mapping[str, Any]) -> Sequence[str]:
        return ()

    @abstractmethod
    def initialize(self, grid, rng, parameters, initial_condition, initial_values):
        """Return a dictionary of initialized real fields."""

    @abstractmethod
    def step(self, state, grid, dt, parameters, rng, noise_strength):
        """Return the fields after one integration step."""

    def derived_fields(self, state, grid, parameters) -> Dict[str, np.ndarray]:
        return {}

    def model_diagnostics(self, state, grid, parameters) -> Dict[str, float]:
        return {}
