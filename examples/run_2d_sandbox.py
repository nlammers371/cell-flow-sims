from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml

from cell_sphere_sim.config import state_table_from_dict
from cell_sphere_sim.planar import PlanarParams, PlanarSimulationEngine, init_random_periodic


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "sim_2d.yaml"


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("planar config must contain a YAML mapping")
    return config


def build_engine(
    config: dict[str, Any],
    *,
    n_cells: int | None = None,
    seed: int | None = None,
    initial_min_separation_factor: float | None = None,
) -> PlanarSimulationEngine:
    init_cfg = config["initialization"]
    sim_cfg = config["simulation"]
    state_table = state_table_from_dict(config["states"])
    n_cells = int(init_cfg["n_cells"] if n_cells is None else n_cells)
    seed = int(init_cfg["seed"] if seed is None else seed)
    factor = float(
        init_cfg["initial_min_separation_factor"]
        if initial_min_separation_factor is None
        else initial_min_separation_factor
    )
    rng = np.random.default_rng(seed)

    fractions = np.asarray(init_cfg.get("state_fractions", [1.0]), dtype=float)
    if fractions.shape != state_table.R.shape or np.any(fractions < 0.0):
        raise ValueError("state_fractions must be non-negative and match the state table length")
    if not np.isfinite(fractions).all() or fractions.sum() <= 0.0:
        raise ValueError("state_fractions must have a finite positive sum")
    state_id = rng.choice(
        fractions.size,
        size=n_cells,
        p=fractions / fractions.sum(),
    ).astype(np.int32)
    state_vars = np.zeros((n_cells, 0), dtype=float)
    x, p = init_random_periodic(
        n_cells,
        sim_cfg["box_size"],
        state_id,
        state_table,
        rng,
        initial_min_separation_factor=factor,
        max_attempts_per_cell=int(init_cfg.get("max_attempts_per_cell", 5000)),
        eps=float(sim_cfg.get("eps", 1e-8)),
    )
    params = PlanarParams(
        box_size=tuple(sim_cfg["box_size"]),
        gamma_s=float(sim_cfg["gamma_s"]),
        k_rep=float(sim_cfg["k_rep"]),
        alpha_dmin=float(sim_cfg["alpha_dmin"]),
        eps=float(sim_cfg.get("eps", 1e-8)),
        dt=float(sim_cfg["dt"]) if sim_cfg.get("dt") is not None else None,
        record_interval=int(sim_cfg.get("record_interval", 1)),
        neighbor_radius_buffer=float(sim_cfg.get("neighbor_radius_buffer", 0.1)),
        division_enabled=bool(sim_cfg.get("division_enabled", False)),
    )
    return PlanarSimulationEngine(
        x,
        p,
        state_id,
        state_vars,
        state_table,
        params,
        rng=rng,
    )


def run_headless(config: dict[str, Any], steps: int) -> dict[str, float | int | bool]:
    engine = build_engine(config)
    start = time.perf_counter()
    diagnostics = engine.run(steps)
    runtime = time.perf_counter() - start
    final = diagnostics[-1] if diagnostics else {
        "n_cells": int(engine.x.shape[0]),
        "mean_speed": 0.0,
        "mean_contacts": 0.0,
        "polarization": 0.0,
        "nematic_order": 0.0,
    }
    result = dict(final)
    result.update(
        {
            "steps": int(steps),
            "all_finite": bool(
                np.all(np.isfinite(engine.x))
                and np.all(np.isfinite(engine.x_unwrapped))
                and np.all(np.isfinite(engine.p))
                and np.all(np.isfinite(engine.v))
            ),
            "runtime_seconds": runtime,
        }
    )
    print(json.dumps(result, sort_keys=True))
    return result


def run_interactive(config: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Rectangle
    from matplotlib.widgets import Button, Slider

    init_cfg = config["initialization"]
    sim_cfg = config["simulation"]
    view_cfg = config.get("sandbox", {})
    engine = build_engine(config)
    step_number = 0
    last_diag: dict[str, float | int] = {}
    running = False

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_axes((0.06, 0.24, 0.60, 0.70))
    box = engine.box_size
    ax.add_patch(Rectangle((0.0, 0.0), box[0], box[1], fill=False, lw=1.5, color="black"))
    ax.set_xlim(0.0, box[0])
    ax.set_ylim(0.0, box[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Periodic planar cell sandbox")

    radii = engine.state_table.R[engine.state_id]
    cell_sizes = 500.0 * radii**2
    cells = ax.scatter(
        engine.x[:, 0],
        engine.x[:, 1],
        s=cell_sizes,
        c=engine.state_id,
        cmap="tab10",
        alpha=0.75,
        edgecolors="black",
        linewidths=0.35,
    )
    arrow_stride = max(1, int(view_cfg.get("arrow_stride", 10)))
    arrow_indices = np.arange(0, engine.x.shape[0], arrow_stride)
    arrows = ax.quiver(
        engine.x[arrow_indices, 0],
        engine.x[arrow_indices, 1],
        engine.p[arrow_indices, 0],
        engine.p[arrow_indices, 1],
        angles="xy",
        scale_units="xy",
        scale=1.5,
        width=0.003,
        color="black",
        visible=bool(view_cfg.get("show_polarity", True)),
    )
    metrics_text = fig.text(0.70, 0.73, "", family="monospace", va="top")

    slider_specs = [
        ("N (reset)", 10, 1500, init_cfg["n_cells"], 1),
        ("seed (reset)", 0, 10000, init_cfg["seed"], 1),
        ("Fm", 0.0, 3.0, config["states"]["Fm"][0], None),
        ("Dr", 0.0, 1.0, config["states"]["Dr"][0], None),
        ("fcil", 0.0, 10.0, config["states"]["fcil"][0], None),
        ("w", 0.0, 2.0, config["states"]["w"][0], None),
        ("k_rep", 0.0, 10.0, sim_cfg["k_rep"], None),
        ("alpha_dmin", 0.0, 0.95, sim_cfg["alpha_dmin"], None),
        ("dt", 0.001, 0.1, sim_cfg["dt"], None),
        ("init clearance", 0.2, 1.2, init_cfg["initial_min_separation_factor"], None),
        ("steps/frame", 1, 25, view_cfg.get("steps_per_frame", 4), 1),
    ]
    sliders: dict[str, Slider] = {}
    for index, (label, low, high, value, step) in enumerate(slider_specs):
        column = index // 6
        row = index % 6
        slider_ax = fig.add_axes((0.70 + 0.15 * column, 0.62 - 0.065 * row, 0.13, 0.025))
        sliders[label] = Slider(slider_ax, label, low, high, valinit=value, valstep=step)

    run_ax = fig.add_axes((0.70, 0.12, 0.10, 0.05))
    step_ax = fig.add_axes((0.81, 0.12, 0.08, 0.05))
    reset_ax = fig.add_axes((0.90, 0.12, 0.08, 0.05))
    run_button = Button(run_ax, "Run")
    step_button = Button(step_ax, "Step")
    reset_button = Button(reset_ax, "Reset")

    def apply_live_parameters() -> None:
        engine.state_table.Fm[:] = sliders["Fm"].val
        engine.state_table.Dr[:] = sliders["Dr"].val
        engine.state_table.fcil[:] = sliders["fcil"].val
        engine.state_table.w[:] = sliders["w"].val
        engine.params.k_rep = sliders["k_rep"].val
        engine.params.alpha_dmin = sliders["alpha_dmin"].val
        engine.params.dt = sliders["dt"].val

    def update_artists() -> None:
        nonlocal arrows, arrow_indices
        cells.set_offsets(engine.x)
        cells.set_sizes(500.0 * engine.state_table.R[engine.state_id] ** 2)
        cells.set_array(engine.state_id.astype(float))
        arrow_indices = np.arange(0, engine.x.shape[0], arrow_stride)
        if arrows.N != arrow_indices.size:
            arrows.remove()
            arrows = ax.quiver(
                engine.x[arrow_indices, 0],
                engine.x[arrow_indices, 1],
                engine.p[arrow_indices, 0],
                engine.p[arrow_indices, 1],
                angles="xy",
                scale_units="xy",
                scale=1.5,
                width=0.003,
                color="black",
                visible=bool(view_cfg.get("show_polarity", True)),
            )
        else:
            arrows.set_offsets(engine.x[arrow_indices])
            arrows.set_UVC(engine.p[arrow_indices, 0], engine.p[arrow_indices, 1])
        if last_diag:
            metrics_text.set_text(
                f"step: {step_number}\n"
                f"cells: {last_diag['n_cells']}\n"
                f"mean speed: {last_diag['mean_speed']:.4g}\n"
                f"mean contacts: {last_diag['mean_contacts']:.4g}\n"
                f"contact pairs: {last_diag['n_contact_pairs']}\n"
                f"min contact d: {last_diag['min_d_contact']:.4g}\n"
                f"polarization: {last_diag['polarization']:.4g}\n"
                f"nematic order: {last_diag['nematic_order']:.4g}\n"
                f"largest cluster: {last_diag['largest_cluster_fraction']:.4g}\n"
                f"MSD: {last_diag['mean_squared_displacement']:.4g}"
            )

    def advance(count: int = 1) -> None:
        nonlocal step_number, last_diag
        apply_live_parameters()
        for _ in range(count):
            last_diag = engine.step(step_number * float(engine.params.dt))
            step_number += 1
        update_artists()

    def toggle_run(_event: object) -> None:
        nonlocal running
        running = not running
        run_button.label.set_text("Pause" if running else "Run")

    def single_step(_event: object) -> None:
        advance()
        fig.canvas.draw_idle()

    def reset(_event: object) -> None:
        nonlocal engine, step_number, last_diag
        engine = build_engine(
            config,
            n_cells=int(sliders["N (reset)"].val),
            seed=int(sliders["seed (reset)"].val),
            initial_min_separation_factor=sliders["init clearance"].val,
        )
        step_number = 0
        last_diag = {}
        apply_live_parameters()
        update_artists()
        fig.canvas.draw_idle()

    def animate(_frame: int):
        if running:
            advance(int(sliders["steps/frame"].val))
        return cells, arrows, metrics_text

    run_button.on_clicked(toggle_run)
    step_button.on_clicked(single_step)
    reset_button.on_clicked(reset)
    update_artists()
    animation = FuncAnimation(
        fig,
        animate,
        interval=int(view_cfg.get("interval_ms", 30)),
        blit=False,
        cache_frame_data=False,
    )
    fig._planar_animation = animation
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive periodic 2D cell sandbox")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--headless", action="store_true", help="run without importing Matplotlib")
    parser.add_argument("--steps", type=int, default=100, help="steps in headless mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 0:
        raise ValueError("--steps must be non-negative")
    config = load_config(args.config)
    if args.headless:
        run_headless(config, args.steps)
    else:
        run_interactive(config)


if __name__ == "__main__":
    main()
