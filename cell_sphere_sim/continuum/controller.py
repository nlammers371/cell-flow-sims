"""UI-neutral controls for interactive continuum workbenches."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional

from .config import ContinuumConfig
from .engine import ContinuumEngine
from .export import export_run


class ContinuumWorkbenchController:
    """Run/pause/step/reset semantics with per-model configuration memory."""

    def __init__(self, config: Optional[ContinuumConfig] = None):
        config = config or ContinuumConfig()
        self.engine = ContinuumEngine(config)
        self.running = False
        self._configs: Dict[str, ContinuumConfig] = {config.model: config}

    def toggle_running(self) -> bool:
        self.running = not self.running
        return self.running

    def pause(self) -> None:
        self.running = False

    def tick(self):
        if self.running:
            return self.engine.advance_frame()
        return self.engine.state

    def step(self):
        return self.engine.advance_frame()

    def update_parameter(self, key, value) -> None:
        self.engine.set_parameter(key, value)
        parameters = dict(self.engine.parameters)
        self._configs[self.engine.model.key] = replace(self.engine.config, parameters=parameters)

    def reset(self):
        config = replace(self.engine.config, parameters=dict(self.engine.parameters))
        self._configs[self.engine.model.key] = config
        self.engine = ContinuumEngine(config)
        return self.engine.state

    def select_preset(self, preset: str):
        config = replace(
            self.engine.config,
            preset=preset,
            parameters={},
            initial_condition=None,
            mean=None,
            initial_amplitude=None,
            droplet_radius=None,
            droplet_count=None,
        )
        self._configs[self.engine.model.key] = config
        self.engine = ContinuumEngine(config)
        return self.engine.state

    def switch_model(self, model: str):
        current = replace(self.engine.config, parameters=dict(self.engine.parameters))
        self._configs[self.engine.model.key] = current
        config = self._configs.get(model)
        if config is None:
            config = replace(current, model=model, preset=None, parameters={})
            self._configs[model] = config
        self.engine = ContinuumEngine(config)
        return self.engine.state

    def export(self, directory, figure=None):
        return export_run(self.engine, directory, figure)

    def reconfigure(self, **changes):
        values = self.engine.config.to_dict()
        values["parameters"] = dict(self.engine.parameters)
        values.update(changes)
        config = ContinuumConfig.from_mapping(values)
        self._configs[self.engine.model.key] = config
        self.engine = ContinuumEngine(config)
        return self.engine.state
