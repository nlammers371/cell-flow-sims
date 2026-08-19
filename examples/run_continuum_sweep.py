"""Run a reproducible two-parameter continuum sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

from cell_sphere_sim.continuum.sweep import SweepRunner, SweepSpec

from run_continuum_workbench import load_config


DEFAULT_SWEEP_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "continuum_mips_sweep.yaml"


def _floats(value):
    return [float(item) for item in value.split(",")]


def _ints(value):
    return [int(item) for item in value.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Continuum operational-map sweep")
    parser.add_argument("--config", type=Path, default=DEFAULT_SWEEP_CONFIG)
    parser.add_argument("--x", default="mean", help="first parameter or shared config key")
    parser.add_argument("--x-values", type=_floats, default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--y", default="v0", help="second parameter or shared config key")
    parser.add_argument("--y-values", type=_floats, default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--seeds", type=_ints, default=[1, 2, 3])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("outputs/continuum_sweep"))
    args = parser.parse_args()
    spec = SweepSpec(args.x, args.x_values, args.y, args.y_values, args.seeds, args.steps)
    runner = SweepRunner(load_config(args.config), spec)

    def progress(done, total, _row):
        print(f"{done}/{total}", flush=True)

    runner.run(progress)
    paths = runner.export(args.output)
    print(f"Wrote {paths['csv']} and {paths['metadata']}")


if __name__ == "__main__":
    main()
