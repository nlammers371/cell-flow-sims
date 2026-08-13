import json
import os
from pathlib import Path
import subprocess
import sys

from examples.run_2d_sandbox import PARAMETER_LABELS


def test_parameter_labels_are_short_and_descriptive():
    expected_keys = {
        "n_cells",
        "seed",
        "motility",
        "diffusion",
        "cil_rate",
        "adhesion",
        "repulsion",
        "hard_core",
        "timestep",
        "clearance",
        "steps_per_frame",
    }
    assert set(PARAMETER_LABELS) == expected_keys
    assert all(1 <= len(label.split()) <= 3 for label in PARAMETER_LABELS.values())


def test_headless_planar_sandbox_succeeds():
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "examples/run_2d_sandbox.py", "--headless", "--steps", "5"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    metrics = json.loads(result.stdout.strip().splitlines()[-1])
    assert metrics["n_cells"] == 500
    assert metrics["steps"] == 5
    assert metrics["all_finite"] is True
