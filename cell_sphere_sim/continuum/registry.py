"""Central model registry used by engines, GUIs, and batch tools."""

from __future__ import annotations

from typing import Dict, Type

from .base import ContinuumModel
from .models import ActiveModelB, DensityPolarizationModel, KellerSegelModel, ModelB


MODEL_REGISTRY: Dict[str, Type[ContinuumModel]] = {
    ModelB.key: ModelB,
    ActiveModelB.key: ActiveModelB,
    DensityPolarizationModel.key: DensityPolarizationModel,
    KellerSegelModel.key: KellerSegelModel,
}


def model_keys():
    return tuple(MODEL_REGISTRY)


def create_model(key: str) -> ContinuumModel:
    try:
        return MODEL_REGISTRY[key]()
    except KeyError as exc:
        choices = ", ".join(MODEL_REGISTRY)
        raise KeyError(f"Unknown continuum model {key!r}; choose from {choices}") from exc
