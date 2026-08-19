"""Small reproducible two-parameter sweep runner."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

from .config import ContinuumConfig
from .engine import ContinuumEngine
from .export import git_revision, json_safe


@dataclass(frozen=True)
class SweepSpec:
    parameter_x: str
    values_x: Sequence[float]
    parameter_y: str
    values_y: Sequence[float]
    seeds: Sequence[int]
    steps: int


class SweepRunner:
    def __init__(self, config: ContinuumConfig, spec: SweepSpec):
        self.config = config
        self.spec = spec
        self.cancelled = False
        self.results = []

    def cancel(self) -> None:
        self.cancelled = True

    def run(self, progress: Optional[Callable[[int, int, Dict], None]] = None):
        total = len(self.spec.values_x) * len(self.spec.values_y) * len(self.spec.seeds)
        completed = 0
        for value_x in self.spec.values_x:
            for value_y in self.spec.values_y:
                for seed in self.spec.seeds:
                    if self.cancelled:
                        return self.results
                    config_values = self.config.to_dict()
                    parameters = dict(self.config.parameters)
                    for key, value in (
                        (self.spec.parameter_x, value_x),
                        (self.spec.parameter_y, value_y),
                    ):
                        if key in config_values and key != "parameters":
                            config_values[key] = value
                        else:
                            parameters[key] = value
                    config_values.update({"parameters": parameters, "seed": int(seed)})
                    config = ContinuumConfig.from_mapping(config_values)
                    engine = ContinuumEngine(config)
                    engine.step(self.spec.steps)
                    row = {
                        self.spec.parameter_x: value_x,
                        self.spec.parameter_y: value_y,
                        "seed": int(seed),
                        **engine.diagnostics(),
                    }
                    self.results.append(row)
                    completed += 1
                    if progress is not None:
                        progress(completed, total, row)
        return self.results

    def export(self, directory) -> Dict[str, Path]:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        csv_path = output / "sweep.csv"
        keys = sorted({key for row in self.results for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.results)
        summary_rows = []
        metrics = ("variance", "largest_cluster", "length_scale")
        if self.results:
            import numpy as np

            for value_x in self.spec.values_x:
                for value_y in self.spec.values_y:
                    matches = [
                        row for row in self.results
                        if row[self.spec.parameter_x] == value_x
                        and row[self.spec.parameter_y] == value_y
                    ]
                    row = {
                        self.spec.parameter_x: value_x,
                        self.spec.parameter_y: value_y,
                        "replicates": len(matches),
                    }
                    for metric in metrics:
                        values = np.asarray([match[metric] for match in matches], dtype=float)
                        row[f"{metric}_mean"] = float(np.nanmean(values))
                        row[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
                    summary_rows.append(row)
        summary_path = output / "sweep_summary.csv"
        if summary_rows:
            with summary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
                writer.writeheader()
                writer.writerows(summary_rows)
        probe = ContinuumEngine(self.config)
        metadata = {
            "label": "operational map (finite-time, finite-size)",
            "model": probe.model.key,
            "model_name": probe.model.name,
            "equation_version": probe.model.equation_version,
            "equations": list(probe.model.equations),
            "base_parameters": dict(probe.parameters),
            "initial_condition": probe.initial_condition,
            "initial_values": dict(probe.initial_values),
            "config": self.config.to_dict(),
            "sweep": {
                "parameter_x": self.spec.parameter_x,
                "values_x": list(self.spec.values_x),
                "parameter_y": self.spec.parameter_y,
                "values_y": list(self.spec.values_y),
                "seeds": list(self.spec.seeds),
                "steps": self.spec.steps,
            },
            "cancelled": self.cancelled,
            "git_revision": git_revision(Path(__file__).resolve().parents[2]),
        }
        json_path = output / "sweep_metadata.json"
        json_path.write_text(json.dumps(json_safe(metadata), indent=2), encoding="utf-8")
        paths = {"csv": csv_path, "metadata": json_path}
        if summary_rows:
            paths["summary"] = summary_path
        if self.results:
            try:
                import numpy as np
                import matplotlib.pyplot as plt

                metric = "largest_cluster"
                image_values = np.full((len(self.spec.values_y), len(self.spec.values_x)), np.nan)
                for iy, value_y in enumerate(self.spec.values_y):
                    for ix, value_x in enumerate(self.spec.values_x):
                        matches = [row for row in summary_rows
                                   if row[self.spec.parameter_x] == value_x
                                   and row[self.spec.parameter_y] == value_y]
                        if matches:
                            image_values[iy, ix] = matches[0][f"{metric}_mean"]
                figure, axis = plt.subplots(figsize=(6, 5))
                image = axis.imshow(image_values, origin="lower", aspect="auto", cmap="viridis")
                axis.set_xticks(range(len(self.spec.values_x)), [f"{v:g}" for v in self.spec.values_x])
                axis.set_yticks(range(len(self.spec.values_y)), [f"{v:g}" for v in self.spec.values_y])
                axis.set_xlabel(self.spec.parameter_x)
                axis.set_ylabel(self.spec.parameter_y)
                axis.set_title("Operational map (finite-time, finite-size)")
                figure.colorbar(image, ax=axis, label="largest cluster fraction")
                image_path = output / "operational_map.png"
                figure.savefig(image_path, dpi=180, bbox_inches="tight")
                plt.close(figure)
                paths["image"] = image_path
            except ImportError:
                pass
        return paths
