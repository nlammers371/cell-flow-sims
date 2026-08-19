"""Side-by-side continuum comparisons from a shared scalar initial condition."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import ContinuumConfig
from .engine import ContinuumEngine


class ContinuumComparison:
    def __init__(self, configs: Sequence[ContinuumConfig]):
        if len(configs) != 2:
            raise ValueError("ContinuumComparison currently supports exactly two models")
        first = ContinuumEngine(configs[0])
        scalar = first.state.fields[first.model.primary_field].copy()
        second_probe = ContinuumEngine(configs[1])
        if second_probe.model.primary_field in second_probe.model.nonnegative_fields and np.min(scalar) < 0.0:
            raise ValueError("The first model's scalar field is negative and cannot initialize a density model")
        second = ContinuumEngine(configs[1], initial_scalar=scalar)
        self.engines = (first, second)

    def step(self, count: int = 1):
        for engine in self.engines:
            engine.step(count)
        return tuple(engine.state for engine in self.engines)

    def advance_frame(self):
        for engine in self.engines:
            engine.advance_frame()
        return tuple(engine.state for engine in self.engines)
