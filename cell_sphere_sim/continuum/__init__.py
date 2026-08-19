"""Continuum phase-separation and aggregation workbench.

The public API deliberately stays small: construct a :class:`ContinuumConfig`,
pass it to :class:`ContinuumEngine`, and advance the engine independently of
rendering.  Model-specific controls and presets are exposed through the
registry so GUIs and batch scripts do not duplicate scientific defaults.
"""

from .base import ContinuumState, ParameterSpec, PresetSpec
from .config import ContinuumConfig
from .controller import ContinuumWorkbenchController
from .engine import ContinuumEngine
from .registry import MODEL_REGISTRY, create_model, model_keys

__all__ = [
    "ContinuumConfig",
    "ContinuumEngine",
    "ContinuumState",
    "ContinuumWorkbenchController",
    "MODEL_REGISTRY",
    "ParameterSpec",
    "PresetSpec",
    "create_model",
    "model_keys",
]
