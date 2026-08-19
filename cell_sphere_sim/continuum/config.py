"""Configuration for continuum simulations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional


@dataclass
class ContinuumConfig:
    """Numerical and initialization controls shared across continuum models."""

    model: str = "model_b"
    preset: Optional[str] = None
    grid_size: int = 128
    domain_size: float = 64.0
    dt: float = 0.02
    substeps_per_frame: int = 5
    seed: int = 1
    initial_condition: Optional[str] = None
    initial_amplitude: Optional[float] = None
    dynamic_noise: float = 0.0
    mean: Optional[float] = None
    droplet_radius: Optional[float] = None
    droplet_count: Optional[int] = None
    cluster_threshold: Optional[float] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    adaptive_dt: bool = True
    minimum_dt: float = 1.0e-7
    max_step_retries: int = 12
    negative_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.grid_size < 8:
            raise ValueError("grid_size must be at least 8")
        if self.domain_size <= 0.0:
            raise ValueError("domain_size must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.substeps_per_frame < 1:
            raise ValueError("substeps_per_frame must be at least 1")
        if self.dynamic_noise < 0.0:
            raise ValueError("dynamic_noise cannot be negative")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ContinuumConfig":
        return cls(**dict(values))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
