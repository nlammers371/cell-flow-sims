"""Launch the continuum phase-separation and aggregation workbench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from cell_sphere_sim.continuum import ContinuumConfig, ContinuumEngine, model_keys
from cell_sphere_sim.continuum.export import json_safe


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "continuum.yaml"


def load_config(path: Path) -> ContinuumConfig:
    with path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError("continuum config must contain a YAML mapping")
    return ContinuumConfig.from_mapping(values)


def run_headless(config: ContinuumConfig, steps: int):
    engine = ContinuumEngine(config)
    engine.step(steps)
    result = engine.diagnostics()
    result.update({"model": engine.model.key, "preset": engine.preset.key})
    print(json.dumps(json_safe(result), sort_keys=True, allow_nan=False))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Continuum aggregation workbench")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", choices=model_keys())
    parser.add_argument("--preset")
    parser.add_argument("--grid-size", type=int)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--substeps", type=int)
    parser.add_argument("--fps", type=float, default=25.0, help="interactive render frames per second")
    parser.add_argument("--dynamic-noise", type=float)
    parser.add_argument("--threshold", type=float, help="cluster threshold")
    parser.add_argument(
        "--initial-condition",
        choices=("uniform_noise", "droplet", "multiple_droplets", "single_interface", "radial"),
    )
    parser.add_argument("--mean", type=float)
    parser.add_argument("--initial-amplitude", type=float)
    parser.add_argument("--droplet-radius", type=float)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps < 0:
        raise ValueError("--steps must be non-negative")
    config = load_config(args.config)
    changes = {
        key: value for key, value in {
            "model": args.model,
            "preset": args.preset,
            "grid_size": args.grid_size,
            "dt": args.dt,
            "seed": args.seed,
            "substeps_per_frame": args.substeps,
            "dynamic_noise": args.dynamic_noise,
            "cluster_threshold": args.threshold,
            "initial_condition": args.initial_condition,
            "mean": args.mean,
            "initial_amplitude": args.initial_amplitude,
            "droplet_radius": args.droplet_radius,
        }.items() if value is not None
    }
    if changes:
        config = ContinuumConfig.from_mapping({**config.to_dict(), **changes})
    if args.headless:
        run_headless(config, args.steps)
    else:
        from cell_sphere_sim.continuum.workbench import MatplotlibContinuumWorkbench

        if args.fps <= 0.0:
            raise ValueError("--fps must be positive")
        MatplotlibContinuumWorkbench(config).show(fps=args.fps)


if __name__ == "__main__":
    main()
