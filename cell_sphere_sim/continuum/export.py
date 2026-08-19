"""Reproducible exports for simulations, plots, and animations."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


def json_safe(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def git_revision(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(root), check=True,
            capture_output=True, text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def metadata(engine) -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    return {
        "format_version": 1,
        "model": engine.model.key,
        "model_name": engine.model.name,
        "equation_version": engine.model.equation_version,
        "equations": list(engine.model.equations),
        "parameters": dict(engine.parameters),
        "config": engine.config.to_dict(),
        "effective_dt": engine.current_dt,
        "time": engine.state.time,
        "step": engine.state.step,
        "preset": engine.preset.key,
        "initial_condition": engine.initial_condition,
        "initial_values": dict(engine.initial_values),
        "git_revision": git_revision(root),
        "corrections": list(engine.corrections),
        "warnings": list(engine.warnings),
    }


def write_diagnostics_csv(history: Sequence[Mapping[str, Any]], path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in history for key in row})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    return output


def export_run(engine, directory, figure=None) -> Dict[str, Path]:
    """Write metadata, diagnostics, final float64 arrays, and optionally a PNG."""

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(json_safe(metadata(engine)), indent=2), encoding="utf-8")
    diagnostics_path = write_diagnostics_csv(engine.history, output / "diagnostics.csv")
    arrays_path = output / "final_fields.npz"
    np.savez_compressed(arrays_path, **engine.state.fields)
    result = {
        "metadata": metadata_path,
        "diagnostics": diagnostics_path,
        "arrays": arrays_path,
    }
    if figure is not None:
        image_path = output / "snapshot.png"
        figure.savefig(image_path, dpi=180, bbox_inches="tight")
        result["image"] = image_path
    return result


def save_animation(frames: Iterable[np.ndarray], path, fps: int = 15, cmap: str = "viridis") -> Path:
    """Save scalar frames as GIF or MP4 using Matplotlib's available writer."""

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    frames = list(frames)
    if not frames:
        raise ValueError("At least one animation frame is required")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots()
    image = axis.imshow(frames[0], origin="lower", cmap=cmap, animated=True)
    axis.set_axis_off()

    def update(index):
        image.set_data(frames[index])
        return (image,)

    animation = FuncAnimation(figure, update, frames=len(frames), blit=True)
    if output.suffix.lower() == ".gif":
        animation.save(output, writer=PillowWriter(fps=fps))
    else:
        animation.save(output, fps=fps)
    plt.close(figure)
    return output
