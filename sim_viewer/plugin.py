from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QLabel,
    QCheckBox,
)

from cell_sphere_sim.engine import SimulationEngine, SimParams
from cell_sphere_sim.state import StateTable


@dataclass
class _UiState:
    engine: SimulationEngine | None
    points_layer: object | None
    tracks_layer: object | None
    vectors_layer: object | None
    surface_layer: object | None
    diagnostics_label: QLabel
    timer: QTimer
    t: float
    frame: int
    tracks_df: pd.DataFrame


def _random_points_on_sphere(rng: np.random.Generator, n: int, R_E: float) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    return R_E * v


def _random_tangent_polarity(rng: np.random.Generator, x: np.ndarray, R_E: float) -> np.ndarray:
    n = x / R_E
    g = rng.normal(size=x.shape)
    g = g - (np.einsum("ij,ij->i", g, n)[:, None]) * n
    g_norm = np.linalg.norm(g, axis=1)
    g_norm = np.where(g_norm > 1e-12, g_norm, 1.0)
    return g / g_norm[:, None]


def _make_uv_sphere(n_lat: int, n_lon: int, R_E: float) -> tuple[np.ndarray, np.ndarray]:
    phi = np.linspace(0.0, np.pi, n_lat)
    theta = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)
    phi_grid, theta_grid = np.meshgrid(phi, theta, indexing="ij")

    x = R_E * np.sin(phi_grid) * np.cos(theta_grid)
    y = R_E * np.sin(phi_grid) * np.sin(theta_grid)
    z = R_E * np.cos(phi_grid)
    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    faces = []
    for i in range(n_lat - 1):
        for j in range(n_lon):
            a = i * n_lon + j
            b = i * n_lon + (j + 1) % n_lon
            c = (i + 1) * n_lon + (j + 1) % n_lon
            d = (i + 1) * n_lon + j
            faces.append([a, b, d])
            faces.append([b, c, d])
    return verts, np.asarray(faces, dtype=np.int32)


def _density_on_vertices(verts: np.ndarray, x: np.ndarray, R_E: float, sigma: float) -> np.ndarray:
    v_unit = verts / R_E
    x_unit = x / R_E
    dots = v_unit @ x_unit.T
    weights = np.exp((dots - 1.0) / max(sigma, 1e-6))
    return weights.sum(axis=1)


def _normalize_density(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo, hi = np.percentile(values, [5.0, 95.0])
    if hi <= lo:
        return np.zeros_like(values)
    norm = (values - lo) / (hi - lo)
    return np.clip(norm, 0.0, 1.0)


def _tracks_dataframe(track_id: np.ndarray, frame: int, x: np.ndarray, state_id: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "track_id": track_id.astype(np.int64),
            "t": np.full((x.shape[0],), frame, dtype=np.int32),
            "z": x[:, 2],
            "y": x[:, 1],
            "x": x[:, 0],
            "state_id": state_id.astype(np.int32),
        }
    )


def _build_engine(rng: np.random.Generator, n_cells: int, R_E: float, dt: float) -> SimulationEngine:
    x = _random_points_on_sphere(rng, n_cells, R_E)
    p = _random_tangent_polarity(rng, x, R_E)
    state_id = np.zeros((n_cells,), dtype=np.int32)
    state_vars = np.zeros((n_cells, 0), dtype=float)

    state_table = StateTable(
        R=np.array([0.4]),
        Fm=np.array([1.0]),
        Dr=np.array([0.05]),
        fcil=np.array([2.0]),
        w=np.array([0.2]),
        lambda_div=np.array([0.0]),
        tau_div=np.array([1.0]),
    )

    params = SimParams(
        R_E=R_E,
        gamma_s=1.0,
        k_rep=2.0,
        alpha_dmin=0.2,
        eps=1e-8,
        dt=dt,
        neighbor_radius_buffer=0.1,
        record_interval=1,
        division_enabled=False,
    )

    return SimulationEngine(x, p, state_id, state_vars, state_table, params, rng=rng)


def make_dock_widget(viewer):
    widget = QWidget()
    layout = QVBoxLayout()

    row1 = QHBoxLayout()
    row1.addWidget(QLabel("Cells"))
    n_cells = QSpinBox()
    n_cells.setRange(10, 5000)
    n_cells.setValue(500)
    row1.addWidget(n_cells)

    row1.addWidget(QLabel("Seed"))
    seed = QSpinBox()
    seed.setRange(0, 10_000)
    seed.setValue(0)
    row1.addWidget(seed)
    layout.addLayout(row1)

    row2 = QHBoxLayout()
    row2.addWidget(QLabel("dt"))
    dt = QDoubleSpinBox()
    dt.setDecimals(4)
    dt.setRange(0.0001, 0.1)
    dt.setSingleStep(0.001)
    dt.setValue(0.01)
    row2.addWidget(dt)

    row2.addWidget(QLabel("Vector scale"))
    vscale = QDoubleSpinBox()
    vscale.setDecimals(3)
    vscale.setRange(0.01, 5.0)
    vscale.setValue(0.5)
    row2.addWidget(vscale)
    layout.addLayout(row2)

    row3 = QHBoxLayout()
    row3.addWidget(QLabel("Sphere res"))
    sphere_res = QSpinBox()
    sphere_res.setRange(8, 60)
    sphere_res.setValue(24)
    row3.addWidget(sphere_res)

    row3.addWidget(QLabel("Display radius"))
    display_radius_scale = QDoubleSpinBox()
    display_radius_scale.setDecimals(3)
    display_radius_scale.setRange(0.8, 1.0)
    display_radius_scale.setSingleStep(0.01)
    display_radius_scale.setValue(0.97)
    row3.addWidget(display_radius_scale)

    density_toggle = QCheckBox("Density shading")
    density_toggle.setChecked(True)
    row3.addWidget(density_toggle)

    row3.addWidget(QLabel("sigma"))
    sigma = QDoubleSpinBox()
    sigma.setDecimals(3)
    sigma.setRange(0.01, 1.0)
    sigma.setValue(0.15)
    row3.addWidget(sigma)
    layout.addLayout(row3)

    row4 = QHBoxLayout()
    btn_step = QPushButton("Step")
    btn_start = QPushButton("Start")
    btn_stop = QPushButton("Stop")
    btn_reset = QPushButton("Reset")
    row4.addWidget(btn_step)
    row4.addWidget(btn_start)
    row4.addWidget(btn_stop)
    row4.addWidget(btn_reset)
    layout.addLayout(row4)

    diagnostics_label = QLabel("Diagnostics: ready")
    layout.addWidget(diagnostics_label)

    widget.setLayout(layout)

    timer = QTimer()
    timer.setInterval(30)
    state = _UiState(
        engine=None,
        points_layer=None,
        tracks_layer=None,
        vectors_layer=None,
        surface_layer=None,
        diagnostics_label=diagnostics_label,
        timer=timer,
        t=0.0,
        frame=0,
        tracks_df=pd.DataFrame(columns=["track_id", "t", "z", "y", "x", "state_id"]),
    )

    def _set_diag_text(diag: dict | None = None):
        if diag is None:
            diagnostics_label.setText("Diagnostics: ready")
            return
        diagnostics_label.setText(
            " | ".join(
                [
                    f"n={diag['n_cells']}",
                    f"speed={diag['mean_speed']:.3f}",
                    f"contacts={diag['mean_contacts']:.2f}",
                    f"pairs={diag['n_contact_pairs']}",
                    f"min_d={diag['min_d_contact']:.3f}",
                ]
            )
        )

    def _init_layers():
        rng = np.random.default_rng(seed.value())
        engine = _build_engine(rng, n_cells.value(), 10.0, dt.value())
        state.engine = engine
        state.t = 0.0
        state.frame = 0
        state.tracks_df = _tracks_dataframe(engine.track_id, state.frame, engine.x, engine.state_id)
        _set_diag_text(None)

        points = engine.x
        features = {"state_id": engine.state_id}
        if state.points_layer is None:
            state.points_layer = viewer.add_points(
                points,
                size=3.0,
                features=features,
                face_color="state_id",
                face_colormap="tab10",
                shading="spherical",
            )
        else:
            state.points_layer.data = points
            state.points_layer.features = features
            state.points_layer.shading = "spherical"

        if state.tracks_layer is None:
            state.tracks_layer = viewer.add_tracks(
                state.tracks_df[["track_id", "t", "z", "y", "x"]].to_numpy(dtype=float),
                features=state.tracks_df[["state_id"]],
                color_by="state_id",
                colormap="tab10",
                tail_length=40,
            )
        else:
            state.tracks_layer.data = state.tracks_df[["track_id", "t", "z", "y", "x"]].to_numpy(dtype=float)
            state.tracks_layer.features = state.tracks_df[["state_id"]]

        vec_data = np.stack([engine.x, engine.x + vscale.value() * engine.p], axis=1)
        if state.vectors_layer is None:
            state.vectors_layer = viewer.add_vectors(vec_data, edge_width=1.0, edge_color="yellow")
        else:
            state.vectors_layer.data = vec_data

        verts, faces = _make_uv_sphere(
            sphere_res.value(),
            sphere_res.value() * 2,
            10.0 * display_radius_scale.value(),
        )
        values = np.zeros((verts.shape[0],), dtype=float)
        if density_toggle.isChecked():
            values = _normalize_density(_density_on_vertices(verts, engine.x, 10.0, sigma.value()))
        if state.surface_layer is None:
            state.surface_layer = viewer.add_surface((verts, faces, values), colormap="viridis", opacity=0.35)
        else:
            state.surface_layer.data = (verts, faces, values)

    def _update_layers():
        if state.engine is None:
            return
        engine = state.engine
        vec_data = np.stack([engine.x, engine.x + vscale.value() * engine.p], axis=1)
        state.points_layer.data = engine.x
        state.points_layer.features = {"state_id": engine.state_id}
        state.points_layer.shading = "spherical"
        state.vectors_layer.data = vec_data
        state.frame += 1
        frame_df = _tracks_dataframe(engine.track_id, state.frame, engine.x, engine.state_id)
        state.tracks_df = pd.concat([state.tracks_df, frame_df], ignore_index=True)
        state.tracks_layer.data = state.tracks_df[["track_id", "t", "z", "y", "x"]].to_numpy(dtype=float)
        state.tracks_layer.features = state.tracks_df[["state_id"]]
        if density_toggle.isChecked():
            verts = state.surface_layer.data[0]
            values = _normalize_density(_density_on_vertices(verts, engine.x, 10.0, sigma.value()))
            state.surface_layer.data = (state.surface_layer.data[0], state.surface_layer.data[1], values)

    def _step_once():
        if state.engine is None:
            _init_layers()
        diag = state.engine.step(state.t)
        state.t += dt.value()
        _update_layers()
        _set_diag_text(diag)

    def _start():
        if state.engine is None:
            _init_layers()
        if not timer.isActive():
            timer.start(0)

    def _stop():
        if timer.isActive():
            timer.stop()

    def _reset():
        _stop()
        _init_layers()

    def _on_timer():
        if state.engine is None:
            return
        diag = state.engine.step(state.t)
        state.t += dt.value()
        _update_layers()
        _set_diag_text(diag)

    timer.timeout.connect(_on_timer)
    btn_step.clicked.connect(_step_once)
    btn_start.clicked.connect(_start)
    btn_stop.clicked.connect(_stop)
    btn_reset.clicked.connect(_reset)

    _init_layers()
    return widget
