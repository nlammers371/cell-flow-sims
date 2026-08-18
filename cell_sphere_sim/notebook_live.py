"""ipywidgets-based live controls for the periodic planar simulation.

This module is intentionally separate from the numerical engine so importing
``cell_sphere_sim`` does not require notebook dependencies.  Import it directly
from a notebook when an interactive view is wanted.
"""

from __future__ import annotations

import asyncio
import copy
import io
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from .state import StateTable


class CircularRegionSelector:
    """Click-to-select circular regions on a planar point cloud.

    This class expects Matplotlib's widget backend (``%matplotlib widget`` in
    a notebook). The selected mask is the union of disks centered on each
    left-click, all sharing the editable radius.
    """

    def __init__(
        self,
        points: np.ndarray,
        *,
        radius: float,
        initial_centers: np.ndarray | None = None,
        on_change=None,
        title: str = "Click disk centers for state 1",
    ) -> None:
        try:
            import ipywidgets as widgets
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "CircularRegionSelector requires ipywidgets in the notebook kernel."
            ) from exc

        coordinates = np.asarray(points, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("points must contain only finite values")
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be finite and positive")

        if initial_centers is None:
            centers = np.empty((0, 2), dtype=float)
        else:
            centers = np.asarray(initial_centers, dtype=float)
            if centers.size == 0:
                centers = np.empty((0, 2), dtype=float)
            if centers.ndim != 2 or centers.shape[1] != 2:
                raise ValueError("initial_centers must have shape (K, 2)")
            if not np.all(np.isfinite(centers)):
                raise ValueError("initial_centers must contain only finite values")

        self.widgets = widgets
        self.points = coordinates
        self.centers = [tuple(center) for center in centers]
        self.on_change = on_change
        self.title = title
        self._circle_artists = []

        self.radius_control = widgets.BoundedFloatText(
            value=float(radius),
            min=1e-6,
            max=1e9,
            description="R (um)",
            layout=widgets.Layout(width="180px"),
            style={"description_width": "60px"},
        )
        self.undo_button = widgets.Button(description="Undo last")
        self.clear_button = widgets.Button(description="Clear all", button_style="warning")
        self.status = widgets.HTML()
        self.selection_text = widgets.HTML()
        self.controls = widgets.VBox(
            [
                widgets.HBox(
                    [self.radius_control, self.undo_button, self.clear_button]
                ),
                self.status,
                self.selection_text,
            ]
        )

        self.figure, self.axis = plt.subplots(figsize=(7.0, 7.0), constrained_layout=True)
        if not isinstance(self.figure.canvas, widgets.Widget):
            plt.close(self.figure)
            raise RuntimeError(
                "CircularRegionSelector needs an interactive Matplotlib canvas. "
                "Run `%matplotlib widget` before constructing it."
            )
        self.point_artist = self.axis.scatter(
            self.points[:, 0],
            self.points[:, 1],
            s=16,
            color="#8a8a8a",
            alpha=0.75,
            edgecolors="none",
        )
        self.axis.set(
            aspect="equal",
            xlabel="projected x (um)",
            ylabel="projected y (um)",
            title=self.title,
        )
        self.axis.axhline(0.0, color="0.85", linewidth=0.8, zorder=0)
        self.axis.axvline(0.0, color="0.85", linewidth=0.8, zorder=0)
        self._connection_id = self.figure.canvas.mpl_connect(
            "button_press_event", self._on_click
        )
        self.undo_button.on_click(self._on_undo)
        self.clear_button.on_click(self._on_clear)
        self.radius_control.observe(self._on_radius_change, names="value")
        self._refresh()

    @property
    def radius(self) -> float:
        return float(self.radius_control.value)

    @property
    def centers_array(self) -> np.ndarray:
        if not self.centers:
            return np.empty((0, 2), dtype=float)
        return np.asarray(self.centers, dtype=float).reshape(-1, 2)

    @property
    def mask(self) -> np.ndarray:
        centers = self.centers_array
        if not centers.size:
            return np.zeros(self.points.shape[0], dtype=bool)
        squared_distance = np.sum(
            (self.points[:, None, :] - centers[None, :, :]) ** 2,
            axis=2,
        )
        return np.any(squared_distance <= self.radius**2, axis=1)

    def _notify(self) -> None:
        if self.on_change is not None:
            self.on_change(self.centers_array.copy(), self.radius)

    def _refresh(self) -> None:
        for artist in self._circle_artists:
            artist.remove()
        self._circle_artists = []
        for center in self.centers:
            artist = Circle(
                center,
                self.radius,
                fill=False,
                color="#c44e52",
                linewidth=1.8,
            )
            self.axis.add_patch(artist)
            self._circle_artists.append(artist)

        selected = self.mask
        colors = np.where(selected[:, None], [0.77, 0.31, 0.32, 0.9], [0.54, 0.54, 0.54, 0.7])
        self.point_artist.set_facecolors(colors)
        self.status.value = (
            f"<b>{len(self.centers)} region(s)</b> | radius={self.radius:.3f} um | "
            f"state-1 cells={np.count_nonzero(selected):,}/{len(selected):,}"
        )
        center_literal = self.centers_array.tolist()
        self.selection_text.value = (
            "<details><summary>Current reproducible values</summary>"
            f"<pre>SUBSTATE_REGION_RADIUS_UM = {self.radius!r}\n"
            f"SUBSTATE_REGION_CENTERS_UM = np.array({center_literal!r}, dtype=float)</pre>"
            "</details>"
        )
        self.figure.canvas.draw_idle()
        self._notify()

    def _on_click(self, event) -> None:
        if event.inaxes is not self.axis or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.centers.append((float(event.xdata), float(event.ydata)))
        self._refresh()

    def _on_undo(self, _button) -> None:
        if self.centers:
            self.centers.pop()
            self._refresh()

    def _on_clear(self, _button) -> None:
        if self.centers:
            self.centers.clear()
            self._refresh()

    def _on_radius_change(self, change) -> None:
        if change["new"] != change["old"]:
            self._refresh()

    def show(self):
        """Display the controls and interactive canvas, then return ``self``."""
        from IPython.display import display

        display(self.controls)
        display(self.figure.canvas)
        return self


class LivePlanarSimulation:
    """Interactive in-notebook controller for a ``PlanarSimulationEngine``.

    The controller owns a copy of the supplied engine.  ``Restart`` restores
    its positions, polarities, lineage identifiers, and RNG state, then applies
    the parameter values currently visible in the controls.
    """

    _STATE_COLORS = ("#2a6fb5", "#c44e52", "#55a868", "#8172b2")

    def __init__(
        self,
        engine,
        *,
        disk_center: tuple[float, float] | None = None,
        disk_radius: float | None = None,
        state_names: Sequence[str] | None = None,
        steps_per_frame: int = 20,
        frame_interval_s: float = 0.15,
        max_cells: int = 5000,
    ) -> None:
        try:
            import ipywidgets as widgets
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "LivePlanarSimulation requires ipywidgets in the notebook kernel."
            ) from exc

        self.widgets = widgets
        self._template_engine = copy.deepcopy(engine)
        self.engine = copy.deepcopy(engine)
        self.t = 0.0
        self.running = False
        self._task = None
        self._last_diagnostics = None
        self.disk_center = disk_center
        self.disk_radius = disk_radius
        self.max_cells = int(max_cells)

        n_states = len(self.engine.state_table.R)
        if state_names is None:
            state_names = tuple(f"state {index}" for index in range(n_states))
        if len(state_names) != n_states:
            raise ValueError("state_names must have one entry per state-table row")
        self.state_names = tuple(state_names)

        self._build_controls(steps_per_frame, frame_interval_s)
        self._connect_callbacks()
        self._render()

    def _float_control(self, value: float, description: str):
        return self.widgets.FloatText(
            value=float(value),
            description=description,
            layout=self.widgets.Layout(width="175px"),
            style={"description_width": "70px"},
        )

    def _build_controls(self, steps_per_frame: int, frame_interval_s: float) -> None:
        widgets = self.widgets
        params = self.engine.params
        table = self.engine.state_table

        self.start_button = widgets.Button(description="Start", button_style="success")
        self.pause_button = widgets.Button(description="Pause")
        self.step_button = widgets.Button(description="Step once")
        self.apply_button = widgets.Button(description="Apply parameters")
        self.restart_button = widgets.Button(description="Restart", button_style="warning")
        self.snapshot_button = widgets.Button(description="Save snapshot")

        self.k_rep_control = self._float_control(params.k_rep, "k_rep")
        self.dt_control = self._float_control(float(params.dt), "dt (h)")
        self.division_enabled_control = widgets.Checkbox(
            value=bool(params.division_enabled),
            description="enable division",
            indent=False,
            layout=widgets.Layout(width="175px"),
        )
        self.show_polarity_control = widgets.Checkbox(
            value=True,
            description="show polarity",
            indent=False,
            layout=widgets.Layout(width="175px"),
        )
        self.steps_per_frame_control = widgets.BoundedIntText(
            value=max(1, int(steps_per_frame)),
            min=1,
            max=10000,
            description="steps/frame",
            layout=widgets.Layout(width="175px"),
            style={"description_width": "85px"},
        )
        self.frame_interval_control = widgets.BoundedFloatText(
            value=max(0.01, float(frame_interval_s)),
            min=0.01,
            max=10.0,
            description="refresh (s)",
            layout=widgets.Layout(width="175px"),
            style={"description_width": "85px"},
        )

        self.state_controls = []
        for state_index, state_name in enumerate(self.state_names):
            same_state_adhesion = (
                float(table.w[state_index] ** 2 / params.k_rep)
                if params.k_rep > 0.0
                else 0.0
            )
            controls = {
                "R": self._float_control(table.R[state_index], "R (um)"),
                "speed": self._float_control(
                    table.Fm[state_index] / params.gamma_s, "speed"
                ),
                "Dr": self._float_control(table.Dr[state_index], "Dr (/h)"),
                "fcil": self._float_control(table.fcil[state_index], "CIL (/h)"),
                "adhesion_ratio": self._float_control(same_state_adhesion, "adhesion A"),
                "lambda_div": self._float_control(
                    table.lambda_div[state_index], "division /h"
                ),
                "tau_div": self._float_control(table.tau_div[state_index], "pause (h)"),
            }
            title = widgets.HTML(f"<b>{state_name}</b>")
            self.state_controls.append((title, controls))

        global_controls = widgets.HBox(
            [
                self.k_rep_control,
                self.dt_control,
                self.steps_per_frame_control,
                self.frame_interval_control,
                self.division_enabled_control,
                self.show_polarity_control,
            ],
            layout=widgets.Layout(flex_flow="row wrap", gap="4px 8px"),
        )
        state_boxes = []
        for title, controls in self.state_controls:
            state_boxes.append(
                widgets.VBox(
                    [
                        title,
                        widgets.HBox(
                            list(controls.values()),
                            layout=widgets.Layout(
                                flex_flow="row wrap", gap="4px 8px"
                            ),
                        ),
                    ],
                    layout=widgets.Layout(border="1px solid #ddd", padding="5px"),
                )
            )

        buttons = widgets.HBox(
            [
                self.start_button,
                self.pause_button,
                self.step_button,
                self.apply_button,
                self.restart_button,
                self.snapshot_button,
            ]
        )
        self.status = widgets.HTML()
        self.image = widgets.Image(format="png", layout=widgets.Layout(width="720px"))
        self.snapshot_output = widgets.Output()
        snapshot_accordion = widgets.Accordion(children=[self.snapshot_output])
        snapshot_accordion.set_title(0, "Saved notebook snapshots")
        snapshot_accordion.selected_index = None
        self.widget = widgets.VBox(
            [
                widgets.HTML(
                    "<b>Live periodic planar simulation</b> — parameter edits take "
                    "effect on Apply, Step, Start, or Restart."
                ),
                global_controls,
                *state_boxes,
                buttons,
                self.status,
                self.image,
                snapshot_accordion,
            ]
        )

    def _connect_callbacks(self) -> None:
        self.start_button.on_click(self._on_start)
        self.pause_button.on_click(self._on_pause)
        self.step_button.on_click(self._on_step)
        self.apply_button.on_click(self._on_apply)
        self.restart_button.on_click(self._on_restart)
        self.snapshot_button.on_click(self._on_snapshot)
        self.show_polarity_control.observe(self._on_display_change, names="value")

    def _parameter_arrays(self) -> dict[str, np.ndarray]:
        arrays = {}
        for key in (
            "R",
            "speed",
            "Dr",
            "fcil",
            "adhesion_ratio",
            "lambda_div",
            "tau_div",
        ):
            arrays[key] = np.array(
                [controls[key].value for _, controls in self.state_controls],
                dtype=float,
            )
        return arrays

    def apply_parameters(self) -> None:
        """Apply all visible controls to the owned engine."""
        arrays = self._parameter_arrays()
        k_rep = float(self.k_rep_control.value)
        dt = float(self.dt_control.value)
        if k_rep <= 0.0:
            raise ValueError("k_rep must be positive")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if np.any(arrays["R"] <= 0.0):
            raise ValueError("all cell radii must be positive")
        for key in (
            "speed",
            "Dr",
            "fcil",
            "adhesion_ratio",
            "lambda_div",
            "tau_div",
        ):
            if np.any(arrays[key] < 0.0):
                raise ValueError(f"{key} must be nonnegative")

        self.engine.params.k_rep = k_rep
        self.engine.params.dt = dt
        self.engine.params.division_enabled = bool(self.division_enabled_control.value)
        self.engine.state_table = StateTable(
            R=arrays["R"],
            Fm=self.engine.params.gamma_s * arrays["speed"],
            Dr=arrays["Dr"],
            fcil=arrays["fcil"],
            w=np.sqrt(k_rep * arrays["adhesion_ratio"]),
            lambda_div=arrays["lambda_div"],
            tau_div=arrays["tau_div"],
        )

    def _advance(self, n_steps: int) -> None:
        self.apply_parameters()
        for _ in range(int(n_steps)):
            self._last_diagnostics = self.engine.step(self.t)
            self.t += float(self.engine.params.dt)
            if self.engine.x.shape[0] >= self.max_cells:
                self.running = False
                raise RuntimeError(
                    f"automatic pause at the {self.max_cells:,}-cell safety limit"
                )

    def _make_figure(self):
        figure, axis = plt.subplots(figsize=(6.6, 6.6), constrained_layout=True)
        n_states = len(self.engine.state_table.R)
        for state_index in range(n_states):
            mask = self.engine.state_id == state_index
            if not np.any(mask):
                continue
            radii = self.engine.state_table.R[self.engine.state_id[mask]]
            marker_diameter_points = (
                2.0
                * radii
                / float(np.max(self.engine.box_size))
                * 6.6
                * 72.0
            )
            axis.scatter(
                self.engine.x[mask, 0],
                self.engine.x[mask, 1],
                s=np.maximum(10.0, marker_diameter_points**2),
                color=self._STATE_COLORS[state_index % len(self._STATE_COLORS)],
                alpha=0.72,
                edgecolors="none",
                label=f"{self.state_names[state_index]} (n={mask.sum():,})",
            )

        if self.show_polarity_control.value and self.engine.x.shape[0]:
            stride = max(1, self.engine.x.shape[0] // 100)
            axis.quiver(
                self.engine.x[::stride, 0],
                self.engine.x[::stride, 1],
                self.engine.p[::stride, 0],
                self.engine.p[::stride, 1],
                angles="xy",
                scale_units="xy",
                scale=0.055,
                width=0.002,
                color="0.15",
                alpha=0.75,
            )
        if self.disk_center is not None and self.disk_radius is not None:
            axis.add_patch(
                Circle(
                    self.disk_center,
                    self.disk_radius,
                    fill=False,
                    color="0.35",
                    linewidth=1.1,
                    linestyle="--",
                    label="initial hemisphere footprint",
                )
            )
        axis.set(
            aspect="equal",
            xlim=(0.0, self.engine.box_size[0]),
            ylim=(0.0, self.engine.box_size[1]),
            xlabel="simulation x (um)",
            ylabel="simulation y (um)",
            title=f"t = {self.t:.4f} h | periodic box",
        )
        axis.legend(loc="upper right", fontsize=8)
        return figure

    def _status_text(self, error: str | None = None) -> str:
        counts = np.bincount(
            self.engine.state_id,
            minlength=len(self.engine.state_table.R),
        )
        state_text = ", ".join(
            f"{name}: {counts[index]:,}" for index, name in enumerate(self.state_names)
        )
        run_text = "running" if self.running else "paused"
        diagnostic_text = ""
        if self._last_diagnostics is not None:
            rejected_divisions = int(
                getattr(self.engine, "total_rejected_divisions", 0)
            )
            rejected_text = (
                f" | crowded division attempts rejected: {rejected_divisions:,}"
                if rejected_divisions
                else ""
            )
            diagnostic_text = (
                f" | contacts: {self._last_diagnostics['n_contact_pairs']:,}"
                f" | mean speed: {self._last_diagnostics['mean_speed']:.2f} um/h"
                f" | divisions: {self.engine.total_divisions:,}"
                f"{rejected_text}"
            )
        error_text = f"<br><span style='color:#b22222'>{error}</span>" if error else ""
        return (
            f"<b>{run_text}</b> | t={self.t:.5f} h | N={self.engine.x.shape[0]:,} "
            f"({state_text}){diagnostic_text}{error_text}"
        )

    def _render(self, error: str | None = None) -> None:
        figure = self._make_figure()
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=110)
        plt.close(figure)
        self.image.value = buffer.getvalue()
        self.status.value = self._status_text(error)

    async def _run(self) -> None:
        try:
            while self.running:
                try:
                    self._advance(self.steps_per_frame_control.value)
                    self._render()
                except Exception as exc:  # keep the notebook responsive on failure
                    self.running = False
                    self._render(f"{type(exc).__name__}: {exc}")
                    break
                await asyncio.sleep(float(self.frame_interval_control.value))
        finally:
            if self._task is asyncio.current_task():
                self._task = None

    def _stop_loop(self) -> None:
        self.running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    def _on_start(self, _button) -> None:
        if self.running:
            return
        try:
            self._stop_loop()
            self.apply_parameters()
            self.running = True
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._run())
            self.status.value = self._status_text()
        except Exception as exc:
            self.running = False
            self._render(f"{type(exc).__name__}: {exc}")

    def _on_pause(self, _button) -> None:
        self._stop_loop()
        self.status.value = self._status_text()

    def _on_step(self, _button) -> None:
        self._stop_loop()
        try:
            self._advance(1)
            self._render()
        except Exception as exc:
            self._render(f"{type(exc).__name__}: {exc}")

    def _on_apply(self, _button) -> None:
        try:
            self.apply_parameters()
            self._render()
        except Exception as exc:
            self._render(f"{type(exc).__name__}: {exc}")

    def _on_restart(self, _button) -> None:
        self._stop_loop()
        self.engine = copy.deepcopy(self._template_engine)
        self.t = 0.0
        self._last_diagnostics = None
        try:
            self.apply_parameters()
            self._render()
        except Exception as exc:
            self._render(f"{type(exc).__name__}: {exc}")

    def _on_snapshot(self, _button) -> None:
        figure = self._make_figure()
        with self.snapshot_output:
            from IPython.display import display

            display(figure)
        plt.close(figure)

    def _on_display_change(self, _change) -> None:
        self._render()

    def show(self):
        """Display the controller and return ``self`` for convenient reuse."""
        from IPython.display import display

        display(self.widget)
        return self

    def pause(self) -> None:
        """Programmatically pause an active live loop."""
        self._stop_loop()
        self.status.value = self._status_text()
