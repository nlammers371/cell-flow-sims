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
    QComboBox,
    QLineEdit,
    QFileDialog,
    QGroupBox,
)

from cell_sphere_sim.engine import SimulationEngine, SimParams
from cell_sphere_sim.state import StateTable

TAIL_LENGTH = 40


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
    # Normalize verts to unit vectors independently so display_radius_scale
    # does not shift the kernel maximum away from 1.
    v_norms = np.linalg.norm(verts, axis=1, keepdims=True)
    v_unit = verts / np.where(v_norms > 1e-12, v_norms, 1.0)
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
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


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


def _build_engine(
    rng: np.random.Generator,
    n_cells: int,
    R_E: float,
    dt: float,
    R: float,
    Fm: float,
    Dr: float,
    fcil: float,
    w: float,
    k_rep: float,
    alpha_dmin: float,
) -> SimulationEngine:
    x = _random_points_on_sphere(rng, n_cells, R_E)
    p = _random_tangent_polarity(rng, x, R_E)
    state_id = np.zeros((n_cells,), dtype=np.int32)
    state_vars = np.zeros((n_cells, 0), dtype=float)

    state_table = StateTable(
        R=np.array([R]),
        Fm=np.array([Fm]),
        Dr=np.array([Dr]),
        fcil=np.array([fcil]),
        w=np.array([w]),
        lambda_div=np.array([0.0]),
        tau_div=np.array([1.0]),
    )

    params = SimParams(
        R_E=R_E,
        gamma_s=1.0,
        k_rep=k_rep,
        alpha_dmin=alpha_dmin,
        eps=1e-8,
        dt=dt,
        neighbor_radius_buffer=0.1,
        record_interval=1,
        division_enabled=False,
    )

    return SimulationEngine(x, p, state_id, state_vars, state_table, params, rng=rng)


def _point_colormap(field_name: str) -> str:
    return "tab10" if field_name == "state_id" else "viridis"


def make_dock_widget(viewer):
    widget = QWidget()
    layout = QVBoxLayout()

    # --- Row 1: n_cells, seed, R_E ---
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

    row1.addWidget(QLabel("R_E"))
    R_E_spin = QDoubleSpinBox()
    R_E_spin.setDecimals(1)
    R_E_spin.setRange(1.0, 200.0)
    R_E_spin.setSingleStep(1.0)
    R_E_spin.setValue(10.0)
    row1.addWidget(R_E_spin)
    layout.addLayout(row1)

    # --- Row 2: config file ---
    row_cfg = QHBoxLayout()
    row_cfg.addWidget(QLabel("Config"))
    config_path = QLineEdit()
    config_path.setPlaceholderText("Path to YAML config (optional)")
    row_cfg.addWidget(config_path)
    btn_browse = QPushButton("Browse")
    btn_load = QPushButton("Load")
    row_cfg.addWidget(btn_browse)
    row_cfg.addWidget(btn_load)
    layout.addLayout(row_cfg)

    # --- Row 3: dt, vector scale, color_by ---
    row2 = QHBoxLayout()
    row2.addWidget(QLabel("dt"))
    dt = QDoubleSpinBox()
    dt.setDecimals(4)
    dt.setRange(0.0001, 0.1)
    dt.setSingleStep(0.001)
    dt.setValue(0.01)
    row2.addWidget(dt)

    row2.addWidget(QLabel("Vec scale"))
    vscale = QDoubleSpinBox()
    vscale.setDecimals(3)
    vscale.setRange(0.01, 5.0)
    vscale.setValue(0.5)
    row2.addWidget(vscale)

    row2.addWidget(QLabel("Color by"))
    color_by = QComboBox()
    color_by.addItems(["state_id", "speed", "contact_count"])
    row2.addWidget(color_by)
    layout.addLayout(row2)

    # --- Row 4: sphere mesh, display radius, density ---
    row3 = QHBoxLayout()
    row3.addWidget(QLabel("Sphere res"))
    sphere_res = QSpinBox()
    sphere_res.setRange(8, 60)
    sphere_res.setValue(24)
    row3.addWidget(sphere_res)

    row3.addWidget(QLabel("Display r"))
    display_radius_scale = QDoubleSpinBox()
    display_radius_scale.setDecimals(3)
    display_radius_scale.setRange(0.8, 1.0)
    display_radius_scale.setSingleStep(0.01)
    display_radius_scale.setValue(0.97)
    row3.addWidget(display_radius_scale)

    density_toggle = QCheckBox("Density")
    density_toggle.setChecked(True)
    row3.addWidget(density_toggle)

    row3.addWidget(QLabel("sigma"))
    sigma = QDoubleSpinBox()
    sigma.setDecimals(3)
    sigma.setRange(0.01, 1.0)
    sigma.setValue(0.15)
    row3.addWidget(sigma)
    layout.addLayout(row3)

    # --- Physics group (live-updates engine in place) ---
    phys_group = QGroupBox("Physics")
    phys_layout = QVBoxLayout()

    phys_row1 = QHBoxLayout()
    phys_row1.addWidget(QLabel("R"))
    R_spin = QDoubleSpinBox()
    R_spin.setDecimals(3)
    R_spin.setRange(0.01, 10.0)
    R_spin.setSingleStep(0.05)
    R_spin.setValue(0.4)
    phys_row1.addWidget(R_spin)

    phys_row1.addWidget(QLabel("Fm"))
    Fm_spin = QDoubleSpinBox()
    Fm_spin.setDecimals(3)
    Fm_spin.setRange(0.0, 20.0)
    Fm_spin.setSingleStep(0.1)
    Fm_spin.setValue(1.0)
    phys_row1.addWidget(Fm_spin)

    phys_row1.addWidget(QLabel("Dr"))
    Dr_spin = QDoubleSpinBox()
    Dr_spin.setDecimals(4)
    Dr_spin.setRange(0.0, 5.0)
    Dr_spin.setSingleStep(0.01)
    Dr_spin.setValue(0.05)
    phys_row1.addWidget(Dr_spin)

    phys_row1.addWidget(QLabel("fcil"))
    fcil_spin = QDoubleSpinBox()
    fcil_spin.setDecimals(3)
    fcil_spin.setRange(0.0, 50.0)
    fcil_spin.setSingleStep(0.5)
    fcil_spin.setValue(2.0)
    phys_row1.addWidget(fcil_spin)
    phys_layout.addLayout(phys_row1)

    phys_row2 = QHBoxLayout()
    phys_row2.addWidget(QLabel("w"))
    w_spin = QDoubleSpinBox()
    w_spin.setDecimals(3)
    w_spin.setRange(0.0, 10.0)
    w_spin.setSingleStep(0.05)
    w_spin.setValue(0.2)
    phys_row2.addWidget(w_spin)

    phys_row2.addWidget(QLabel("k_rep"))
    k_rep_spin = QDoubleSpinBox()
    k_rep_spin.setDecimals(3)
    k_rep_spin.setRange(0.01, 50.0)
    k_rep_spin.setSingleStep(0.5)
    k_rep_spin.setValue(2.0)
    phys_row2.addWidget(k_rep_spin)

    phys_row2.addWidget(QLabel("alpha_dmin"))
    alpha_dmin_spin = QDoubleSpinBox()
    alpha_dmin_spin.setDecimals(3)
    alpha_dmin_spin.setRange(0.01, 0.9)
    alpha_dmin_spin.setSingleStep(0.01)
    alpha_dmin_spin.setValue(0.2)
    phys_row2.addWidget(alpha_dmin_spin)
    phys_layout.addLayout(phys_row2)

    phys_group.setLayout(phys_layout)
    layout.addWidget(phys_group)

    # --- Buttons ---
    row4 = QHBoxLayout()
    btn_step = QPushButton("Step")
    btn_start = QPushButton("Start")
    btn_stop = QPushButton("Stop")
    btn_reset = QPushButton("Reset")
    btn_cam_reset = QPushButton("Reset camera")
    row4.addWidget(btn_step)
    row4.addWidget(btn_start)
    row4.addWidget(btn_stop)
    row4.addWidget(btn_reset)
    row4.addWidget(btn_cam_reset)
    layout.addLayout(row4)

    diagnostics_label = QLabel("Diagnostics: ready")
    layout.addWidget(diagnostics_label)

    widget.setLayout(layout)

    timer = QTimer()
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

    def _set_diag_text(diag: dict | None = None) -> None:
        if diag is None:
            diagnostics_label.setText("Diagnostics: ready")
            return
        min_d = diag["min_d_contact"]
        min_d_str = f"{min_d:.3f}" if not np.isnan(min_d) else "N/A"
        diagnostics_label.setText(
            " | ".join(
                [
                    f"n={diag['n_cells']}",
                    f"speed={diag['mean_speed']:.3f}",
                    f"contacts={diag['mean_contacts']:.2f}",
                    f"pairs={diag['n_contact_pairs']}",
                    f"min_d={min_d_str}",
                ]
            )
        )

    def _get_cell_features(engine: SimulationEngine) -> dict:
        speed = np.linalg.norm(engine.v, axis=1)
        return {
            "state_id": engine.state_id,
            "speed": speed,
            "contact_count": engine.contact_metrics.contact_count.astype(float),
        }

    def _init_layers() -> None:
        rng = np.random.default_rng(seed.value())
        engine = _build_engine(
            rng,
            n_cells.value(),
            R_E_spin.value(),
            dt.value(),
            R_spin.value(),
            Fm_spin.value(),
            Dr_spin.value(),
            fcil_spin.value(),
            w_spin.value(),
            k_rep_spin.value(),
            alpha_dmin_spin.value(),
        )
        state.engine = engine
        state.t = 0.0
        state.frame = 0
        state.tracks_df = _tracks_dataframe(engine.track_id, state.frame, engine.x, engine.state_id)
        _set_diag_text(None)

        R_E = engine.params.R_E
        features = _get_cell_features(engine)
        cb = color_by.currentText()
        cell_size = float(2.0 * np.min(engine.state_table.R))

        if state.points_layer is None:
            state.points_layer = viewer.add_points(
                engine.x,
                size=cell_size,
                features=features,
                face_color=cb,
                face_colormap=_point_colormap(cb),
                shading="spherical",
            )
        else:
            state.points_layer.data = engine.x
            state.points_layer.size = cell_size
            state.points_layer.features = features
            state.points_layer.face_color = cb
            state.points_layer.face_colormap = _point_colormap(cb)

        if state.tracks_layer is None:
            state.tracks_layer = viewer.add_tracks(
                state.tracks_df[["track_id", "t", "z", "y", "x"]].to_numpy(dtype=float),
                features=state.tracks_df[["state_id"]],
                color_by="state_id",
                colormap="tab10",
                tail_length=TAIL_LENGTH,
            )
        else:
            state.tracks_layer.data = state.tracks_df[["track_id", "t", "z", "y", "x"]].to_numpy(dtype=float)
            state.tracks_layer.features = state.tracks_df[["state_id"]]

        vec_data = np.stack([engine.x, engine.x + vscale.value() * engine.p], axis=1)
        if state.vectors_layer is None:
            state.vectors_layer = viewer.add_vectors(vec_data, edge_width=1.0, edge_color="yellow")
        else:
            state.vectors_layer.data = vec_data

        display_R = R_E * display_radius_scale.value()
        verts, faces = _make_uv_sphere(sphere_res.value(), sphere_res.value() * 2, display_R)
        values = np.zeros((verts.shape[0],), dtype=float)
        if density_toggle.isChecked():
            values = _normalize_density(_density_on_vertices(verts, engine.x, R_E, sigma.value()))
        if state.surface_layer is None:
            state.surface_layer = viewer.add_surface((verts, faces, values), colormap="viridis", opacity=0.35)
        else:
            state.surface_layer.data = (verts, faces, values)

    def _update_layers() -> None:
        if state.engine is None:
            return
        engine = state.engine
        R_E = engine.params.R_E
        features = _get_cell_features(engine)

        state.points_layer.data = engine.x
        state.points_layer.features = features

        vec_data = np.stack([engine.x, engine.x + vscale.value() * engine.p], axis=1)
        state.vectors_layer.data = vec_data

        state.frame += 1
        frame_df = _tracks_dataframe(engine.track_id, state.frame, engine.x, engine.state_id)
        state.tracks_df = pd.concat([state.tracks_df, frame_df], ignore_index=True)
        cutoff_t = state.frame - TAIL_LENGTH
        if cutoff_t > 0:
            state.tracks_df = state.tracks_df[state.tracks_df["t"] > cutoff_t].reset_index(drop=True)
        state.tracks_layer.data = state.tracks_df[["track_id", "t", "z", "y", "x"]].to_numpy(dtype=float)
        state.tracks_layer.features = state.tracks_df[["state_id"]]

        if density_toggle.isChecked():
            verts = state.surface_layer.data[0]
            values = _normalize_density(_density_on_vertices(verts, engine.x, R_E, sigma.value()))
            state.surface_layer.data = (state.surface_layer.data[0], state.surface_layer.data[1], values)

    def _update_physics() -> None:
        if state.engine is None:
            return
        st = state.engine.state_table
        st.R[:] = R_spin.value()
        st.Fm[:] = Fm_spin.value()
        st.Dr[:] = Dr_spin.value()
        st.fcil[:] = fcil_spin.value()
        st.w[:] = w_spin.value()
        state.engine.params.k_rep = k_rep_spin.value()
        state.engine.params.alpha_dmin = alpha_dmin_spin.value()
        if state.points_layer is not None:
            state.points_layer.size = float(2.0 * R_spin.value())

    def _on_color_by_changed() -> None:
        if state.points_layer is None:
            return
        cb = color_by.currentText()
        state.points_layer.face_color = cb
        state.points_layer.face_colormap = _point_colormap(cb)

    def _step_once() -> None:
        if state.engine is None:
            _init_layers()
        diag = state.engine.step(state.t)
        state.t += float(state.engine.params.dt)
        _update_layers()
        _set_diag_text(diag)

    def _start() -> None:
        if state.engine is None:
            _init_layers()
        if not timer.isActive():
            timer.start(30)  # ~33 fps

    def _stop() -> None:
        if timer.isActive():
            timer.stop()

    def _reset() -> None:
        _stop()
        _init_layers()

    def _reset_camera() -> None:
        viewer.reset_view()

    def _on_timer() -> None:
        if state.engine is None:
            return
        diag = state.engine.step(state.t)
        state.t += float(state.engine.params.dt)
        _update_layers()
        _set_diag_text(diag)

    def _browse_config() -> None:
        path, _ = QFileDialog.getOpenFileName(widget, "Open config", "", "YAML files (*.yaml *.yml)")
        if path:
            config_path.setText(path)

    def _load_config() -> None:
        import yaml

        path = config_path.text().strip()
        if not path:
            return
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            diagnostics_label.setText(f"Config error: {e}")
            return

        sim_spinbox_map = {
            "R_E": R_E_spin,
            "dt": dt,
            "k_rep": k_rep_spin,
            "alpha_dmin": alpha_dmin_spin,
        }
        for key, spin in sim_spinbox_map.items():
            if key in cfg and cfg[key] is not None:
                spin.setValue(float(cfg[key]))

        state_spinbox_map = {
            "R": R_spin,
            "Fm": Fm_spin,
            "Dr": Dr_spin,
            "fcil": fcil_spin,
            "w": w_spin,
        }
        for key, spin in state_spinbox_map.items():
            if key in cfg:
                val = cfg[key]
                if isinstance(val, list):
                    val = val[0]
                spin.setValue(float(val))

        _reset()

    timer.timeout.connect(_on_timer)
    btn_step.clicked.connect(_step_once)
    btn_start.clicked.connect(_start)
    btn_stop.clicked.connect(_stop)
    btn_reset.clicked.connect(_reset)
    btn_cam_reset.clicked.connect(_reset_camera)
    btn_browse.clicked.connect(_browse_config)
    btn_load.clicked.connect(_load_config)
    color_by.currentTextChanged.connect(_on_color_by_changed)

    for spin in (R_spin, Fm_spin, Dr_spin, fcil_spin, w_spin, k_rep_spin, alpha_dmin_spin):
        spin.valueChanged.connect(_update_physics)

    _init_layers()
    return widget
