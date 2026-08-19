"""Matplotlib continuum workbench built entirely from registry metadata."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from .controller import ContinuumWorkbenchController
from .engine import StepRejected
from .registry import MODEL_REGISTRY


class MatplotlibContinuumWorkbench:
    """Interactive view with non-overlapping registry-generated controls."""

    def __init__(self, config, output_directory="outputs/continuum"):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RadioButtons, Slider

        self.plt = plt
        self.controller = ContinuumWorkbenchController(config)
        self.output_directory = Path(output_directory)
        self.figure = plt.figure(figsize=(16, 9))
        self.primary_axis = self.figure.add_axes((0.18, 0.30, 0.245, 0.61))
        self.secondary_axis = self.figure.add_axes((0.49, 0.30, 0.245, 0.61))
        self.history_axis = self.figure.add_axes((0.18, 0.07, 0.56, 0.15))
        self.model_axis = self.figure.add_axes((0.015, 0.69, 0.14, 0.22))
        self.preset_axis = self.figure.add_axes((0.015, 0.40, 0.14, 0.24))
        self.explanation = self.figure.text(0.015, 0.35, "", va="top", fontsize=8, wrap=True)
        self.metrics = self.figure.text(0.015, 0.22, "", va="top", family="monospace", fontsize=8)
        self.status = self.figure.text(0.77, 0.015, "Ready", va="bottom", fontsize=8)
        self.parameter_title = self.figure.text(
            0.77, 0.925, "Model parameters", fontsize=12, weight="bold"
        )

        self.parameter_axes = []
        self.parameter_group_artists = []
        self.parameter_widgets = {}
        self._axis_help = {}
        self.vector_artist = None
        self.primary_image = self.primary_axis.imshow(
            np.zeros((2, 2)), origin="lower", cmap="coolwarm", interpolation="nearest"
        )
        self.secondary_image = self.secondary_axis.imshow(
            np.zeros((2, 2)), origin="lower", cmap="viridis", interpolation="nearest"
        )
        self.primary_colorbar = self.figure.colorbar(self.primary_image, ax=self.primary_axis, fraction=0.046)
        self.secondary_colorbar = self.figure.colorbar(self.secondary_image, ax=self.secondary_axis, fraction=0.046)

        model_names = [MODEL_REGISTRY[key].name for key in MODEL_REGISTRY]
        self._model_name_to_key = dict(zip(model_names, MODEL_REGISTRY))
        active_index = list(MODEL_REGISTRY).index(config.model)
        self.model_axis.set_title("Continuum model", fontsize=10, loc="left")
        self.model_radio = RadioButtons(self.model_axis, model_names, active=active_index)

        run_axis = self.figure.add_axes((0.77, 0.06, 0.075, 0.05))
        step_axis = self.figure.add_axes((0.85, 0.06, 0.055, 0.05))
        reset_axis = self.figure.add_axes((0.91, 0.06, 0.06, 0.05))
        setup_axis = self.figure.add_axes((0.77, 0.125, 0.095, 0.045))
        export_axis = self.figure.add_axes((0.875, 0.125, 0.095, 0.045))
        self.run_button = Button(run_axis, "Run")
        self.step_button = Button(step_axis, "Step")
        self.reset_button = Button(reset_axis, "Reset")
        self.setup_button = Button(setup_axis, "Setup")
        self.export_button = Button(export_axis, "Export run")
        self.run_button.on_clicked(self._toggle_run)
        self.step_button.on_clicked(self._single_step)
        self.reset_button.on_clicked(self._reset)
        self.setup_button.on_clicked(self._open_setup)
        self.export_button.on_clicked(self._export)
        self.model_radio.on_clicked(self._switch_model)

        # Shared integration controls remain visually separate from model
        # parameters.  Render FPS is passed to show(); neither setting changes
        # the number of PDE steps requested by a button press.
        dt_axis = self.figure.add_axes((0.18, 0.018, 0.20, 0.016))
        substeps_axis = self.figure.add_axes((0.47, 0.018, 0.20, 0.016))
        noise_axis = self.figure.add_axes((0.77, 0.245, 0.20, 0.016))
        threshold_axis = self.figure.add_axes((0.77, 0.195, 0.20, 0.016))
        self.dt_slider = Slider(
            dt_axis, "Time step", -5.0, 0.0, valinit=np.log10(self.engine.current_dt)
        )
        self.substeps_slider = Slider(
            substeps_axis, "Substeps", 1, 50,
            valinit=self.engine.config.substeps_per_frame, valstep=1,
        )
        self.noise_slider = Slider(
            noise_axis, "Dynamic noise", 0.0, 0.5,
            valinit=self.engine.config.dynamic_noise,
        )
        initial_threshold = self.engine.config.cluster_threshold
        if initial_threshold is None:
            initial_threshold = float(np.mean(
                self.engine.state.fields[self.engine.model.primary_field]
            ))
        self.threshold_slider = Slider(
            threshold_axis, "Threshold", -2.0, 4.0, valinit=initial_threshold
        )
        for widget in (
            self.dt_slider, self.substeps_slider, self.noise_slider, self.threshold_slider
        ):
            widget.label.set_position((0.0, 1.35))
            widget.label.set_horizontalalignment("left")
            widget.label.set_fontsize(8)
            widget.valtext.set_position((1.0, 1.35))
            widget.valtext.set_horizontalalignment("right")
            widget.valtext.set_fontsize(7)
        self.dt_slider.valtext.set_text(f"{self.engine.current_dt:.3g}")
        self.dt_slider.on_changed(self._dt_changed)
        self.substeps_slider.on_changed(self._substeps_changed)
        self.noise_slider.on_changed(self._noise_changed)
        self.threshold_slider.on_changed(self._threshold_changed)
        self.figure.canvas.mpl_connect("motion_notify_event", self._show_parameter_help)

        self.preset_radio = None
        self._build_preset_widget()
        self._build_parameter_widgets()
        self._refresh()

    @property
    def engine(self):
        return self.controller.engine

    def _clear_axis_widgets(self, axes):
        for axis in axes:
            axis.remove()
        axes[:] = []

    def _build_preset_widget(self):
        from matplotlib.widgets import RadioButtons

        self.preset_axis.clear()
        self.preset_axis.set_title("Preset", fontsize=10, loc="left")
        presets = self.engine.model.presets
        labels = [preset.name for preset in presets]
        active = next(
            (index for index, preset in enumerate(presets) if preset.key == self.engine.preset.key), 0
        )
        self._preset_name_to_key = {preset.name: preset.key for preset in presets}
        self.preset_radio = RadioButtons(self.preset_axis, labels, active=active)
        for label in self.preset_radio.labels:
            label.set_fontsize(7 if len(labels) > 5 else 8)
        self.preset_radio.on_clicked(self._select_preset)

    def _build_parameter_widgets(self):
        from matplotlib.widgets import RadioButtons, Slider

        self._clear_axis_widgets(self.parameter_axes)
        for artist in self.parameter_group_artists:
            artist.remove()
        self.parameter_group_artists = []
        self.parameter_widgets = {}
        self._axis_help = {}
        specs = self.engine.model.parameter_specs
        available_height = 0.69
        row_height = min(0.061, available_height / max(len(specs), 1))
        top = 0.86
        last_group = None
        for index, spec in enumerate(specs):
            bottom = top - index * row_height
            axis = self.figure.add_axes((0.77, bottom, 0.20, 0.022))
            self.parameter_axes.append(axis)
            self._axis_help[axis] = spec
            if spec.group != last_group:
                self.parameter_group_artists.append(
                    self.figure.text(
                        0.755, bottom + 0.01, spec.group, fontsize=6, color="0.35",
                        rotation=90, ha="center", va="center",
                    )
                )
                last_group = spec.group
            value = self.engine.parameters[spec.key]
            if spec.choices:
                axis.set_position((0.77, bottom - 0.012, 0.20, 0.045))
                widget = RadioButtons(axis, [str(choice) for choice in spec.choices], active=spec.choices.index(value))
                axis.set_title(spec.name, fontsize=8, loc="left", pad=1)
                for label in widget.labels:
                    label.set_fontsize(7)
                widget.on_clicked(lambda selected, key=spec.key: self._choice_changed(key, selected))
            else:
                low = float(spec.minimum)
                high = float(spec.maximum)
                shown = float(value)
                if spec.scale == "log":
                    low, high, shown = np.log10(low), np.log10(high), np.log10(float(value))
                widget = Slider(axis, spec.name, low, high, valinit=shown)
                # Put the short name and value above their own slider track.
                # This is robust even for the ten-control chemotaxis model.
                widget.label.set_position((0.0, 1.32))
                widget.label.set_horizontalalignment("left")
                widget.label.set_fontsize(8)
                widget.valtext.set_position((1.0, 1.32))
                widget.valtext.set_horizontalalignment("right")
                widget.valtext.set_fontsize(7)
                widget.valtext.set_text(f"{float(value):.3g}")
                widget.on_changed(
                    lambda shown_value, key=spec.key, scale=spec.scale, control=widget:
                    self._slider_changed(key, shown_value, scale, control)
                )
            self.parameter_widgets[spec.key] = widget

    def _show_parameter_help(self, event):
        spec = self._axis_help.get(event.inaxes)
        if spec is not None:
            units = "" if spec.units == "dimensionless" else f" [{spec.units}]"
            warning = f" Warning: {spec.stability_warning}" if spec.stability_warning else ""
            self.status.set_text(f"{spec.name}: {spec.description}{units}{warning}")
            self.figure.canvas.draw_idle()

    def _dt_changed(self, shown_value):
        value = 10.0 ** shown_value
        self.engine.current_dt = value
        self.engine.config.dt = value
        self.dt_slider.valtext.set_text(f"{value:.3g}")

    def _substeps_changed(self, value):
        self.engine.config.substeps_per_frame = int(value)

    def _noise_changed(self, value):
        self.engine.config.dynamic_noise = float(value)

    def _threshold_changed(self, value):
        self.engine.config.cluster_threshold = float(value)

    def _slider_changed(self, key, shown_value, scale, widget):
        value = 10.0 ** shown_value if scale == "log" else shown_value
        self.controller.update_parameter(key, value)
        widget.valtext.set_text(f"{value:.3g}")

    def _choice_changed(self, key, selected):
        # Choice-valued controls alter field semantics and therefore reset.
        parameters = dict(self.engine.parameters)
        parameters[key] = selected
        config = self.engine.clone_config(parameters=parameters)
        self.controller._configs[self.engine.model.key] = config
        self.controller.engine = type(self.engine)(config)
        self._refresh()

    def _switch_model(self, label):
        self.controller.pause()
        self.run_button.label.set_text("Run")
        self.controller.switch_model(self._model_name_to_key[label])
        self.dt_slider.set_val(np.log10(self.engine.current_dt))
        self.substeps_slider.set_val(self.engine.config.substeps_per_frame)
        self.noise_slider.set_val(self.engine.config.dynamic_noise)
        threshold = self.engine.config.cluster_threshold
        if threshold is None:
            threshold = float(np.mean(self.engine.state.fields[self.engine.model.primary_field]))
        self.threshold_slider.set_val(threshold)
        self._build_preset_widget()
        self._build_parameter_widgets()
        self._refresh()

    def _select_preset(self, label):
        self.controller.pause()
        self.run_button.label.set_text("Run")
        self.controller.select_preset(self._preset_name_to_key[label])
        self.dt_slider.set_val(np.log10(self.engine.current_dt))
        self._build_parameter_widgets()
        self._refresh()

    def _toggle_run(self, _event):
        running = self.controller.toggle_running()
        self.run_button.label.set_text("Pause" if running else "Run")

    def _single_step(self, _event):
        try:
            self.controller.step()
        except StepRejected as exc:
            self.controller.pause()
            self.run_button.label.set_text("Run")
            self._refresh()
            self.status.set_text(str(exc))
        else:
            self._refresh()
        self.figure.canvas.draw_idle()

    def _reset(self, _event):
        self.controller.pause()
        self.run_button.label.set_text("Run")
        self.controller.reset()
        self._refresh()
        self.figure.canvas.draw_idle()

    def _export(self, _event):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        directory = self.output_directory / f"{self.engine.model.key}-{stamp}"
        paths = self.controller.export(directory, self.figure)
        self.status.set_text(f"Exported {paths['metadata'].parent}")
        self.figure.canvas.draw_idle()

    def _open_setup(self, _event):
        """Open reset-required domain and initialization controls."""

        from matplotlib.widgets import Button, RadioButtons, TextBox

        setup_figure = self.plt.figure(figsize=(6.5, 7.0))
        setup_figure.suptitle("Domain and initial conditions")
        config = self.engine.config
        resolved = self.engine.initial_values
        definitions = [
            ("grid_size", "Grid size", config.grid_size, int),
            ("domain_size", "Domain size", config.domain_size, float),
            ("seed", "Random seed", config.seed, int),
            ("mean", "Field mean", resolved.get("mean", 0.0), float),
            ("initial_amplitude", "Initial amplitude", resolved.get("amplitude", 0.03), float),
            ("droplet_radius", "Droplet radius", resolved.get("radius", 0.1 * config.domain_size), float),
            ("droplet_count", "Droplet count", resolved.get("count", 8), int),
            ("dynamic_noise", "Dynamic noise", config.dynamic_noise, float),
            ("cluster_threshold", "Cluster threshold", config.cluster_threshold, float),
        ]
        boxes = {}
        for index, (key, label, value, converter) in enumerate(definitions):
            axis = setup_figure.add_axes((0.43, 0.86 - 0.07 * index, 0.48, 0.045))
            initial = "" if value is None else str(value)
            boxes[key] = (TextBox(axis, label, initial=initial), converter)

        choices = ("uniform_noise", "droplet", "multiple_droplets", "single_interface", "radial")
        choice_axis = setup_figure.add_axes((0.08, 0.04, 0.38, 0.17))
        choice_axis.set_title("Initial condition", fontsize=9, loc="left")
        active = choices.index(self.engine.initial_condition)
        choice_widget = RadioButtons(choice_axis, choices, active=active)
        for label in choice_widget.labels:
            label.set_fontsize(8)
        selected = {"value": self.engine.initial_condition}
        choice_widget.on_clicked(lambda value: selected.update(value=value))
        apply_axis = setup_figure.add_axes((0.57, 0.08, 0.30, 0.07))
        apply_button = Button(apply_axis, "Apply + reset")

        def apply_setup(_click):
            try:
                changes = {"initial_condition": selected["value"]}
                for key, (box, converter) in boxes.items():
                    text = box.text.strip()
                    changes[key] = None if text == "" else converter(text)
                self.controller.pause()
                self.run_button.label.set_text("Run")
                self.controller.reconfigure(**changes)
                self.dt_slider.set_val(np.log10(self.engine.current_dt))
                self.substeps_slider.set_val(self.engine.config.substeps_per_frame)
                self.noise_slider.set_val(self.engine.config.dynamic_noise)
                threshold = self.engine.config.cluster_threshold
                if threshold is None:
                    threshold = float(np.mean(self.engine.state.fields[self.engine.model.primary_field]))
                self.threshold_slider.set_val(threshold)
                self._refresh()
                self.figure.canvas.draw_idle()
                self.plt.close(setup_figure)
            except (TypeError, ValueError) as exc:
                self.status.set_text(f"Setup error: {exc}")
                self.figure.canvas.draw_idle()

        apply_button.on_clicked(apply_setup)
        setup_figure._continuum_setup_widgets = (boxes, choice_widget, apply_button)
        setup_figure.show()

    def _vector_components(self):
        state = self.engine.state
        if self.engine.model.key == "density_polarization":
            return state.fields["px"], state.fields["py"]
        if self.engine.model.key == "keller_segel":
            return self.engine.grid.gradient(state.fields["c"])
        return None

    def _refresh(self):
        engine = self.engine
        model = engine.model
        primary = engine.state.fields[model.primary_field]
        derived = engine.derived_fields()
        if model.secondary_field in engine.state.fields:
            secondary = engine.state.fields[model.secondary_field]
        elif model.secondary_field in derived:
            secondary = derived[model.secondary_field]
        else:
            secondary = np.zeros_like(primary)

        self.primary_image.set_data(primary)
        self.primary_image.set_clim(float(np.min(primary)), float(np.max(primary)) + 1e-15)
        self.secondary_image.set_data(secondary)
        self.secondary_image.set_clim(float(np.min(secondary)), float(np.max(secondary)) + 1e-15)
        extent = (0.0, engine.grid.length, 0.0, engine.grid.length)
        self.primary_image.set_extent(extent)
        self.secondary_image.set_extent(extent)
        self.primary_axis.set_title(f"{model.name}: {model.primary_field}")
        self.secondary_axis.set_title(model.secondary_field or "derived field")
        for axis in (self.primary_axis, self.secondary_axis):
            axis.set_xlabel("x")
            axis.set_ylabel("y")

        if self.vector_artist is not None:
            self.vector_artist.remove()
            self.vector_artist = None
        vector = self._vector_components()
        if vector is not None and max(float(np.max(np.abs(vector[0]))), float(np.max(np.abs(vector[1])))) > 1e-14:
            stride = max(1, engine.grid.size // 16)
            sample = (slice(None, None, stride), slice(None, None, stride))
            self.vector_artist = self.primary_axis.quiver(
                engine.grid.x[sample], engine.grid.y[sample], vector[0][sample], vector[1][sample],
                color="black", alpha=0.65, pivot="mid",
            )

        diagnostics = engine.diagnostics()
        extra = []
        for key in ("free_energy", "mips_criterion", "signal_length"):
            if key in diagnostics:
                extra.append(f"{key}: {diagnostics[key]:.4g}")
        self.metrics.set_text(
            f"time: {diagnostics['time']:.4g}\n"
            f"step: {diagnostics['step']}\n"
            f"dt: {diagnostics['dt']:.3g}\n"
            f"seed: {engine.config.seed}\n"
            f"mass error: {diagnostics['mass_error']:.2e}\n"
            f"variance: {diagnostics['variance']:.4g}\n"
            f"range: {diagnostics['minimum']:.3g}…{diagnostics['maximum']:.3g}\n"
            f"clusters: {diagnostics['cluster_count']}\n"
            f"threshold: {diagnostics['cluster_threshold']:.3g}\n"
            f"largest: {diagnostics['largest_cluster']:.3g}\n"
            f"length: {diagnostics['length_scale']:.3g}\n" + "\n".join(extra)
        )
        equation_text = "\n".join(model.equations)
        note = ""
        if model.key == "active_model_b":
            note = "\nλ is activity—not speed or temperature. Passive F is not a Lyapunov function."
        self.explanation.set_text(model.description + "\n\n" + equation_text + note)

        self.history_axis.clear()
        times = [row["time"] for row in engine.history]
        self.history_axis.plot(times, [row["variance"] for row in engine.history], label="variance")
        if engine.history and "free_energy" in engine.history[0]:
            self.history_axis.plot(times, [row["free_energy"] for row in engine.history], label="free energy")
        self.history_axis.set_title("Live diagnostics", fontsize=9)
        if len(times) < 2:
            self.history_axis.set_xlim(0.0, max(0.1, engine.current_dt * engine.config.substeps_per_frame))
        self.history_axis.legend(loc="best", fontsize=7)
        latest_warning = engine.warnings[-1] if engine.warnings else "Ready"
        self.status.set_text(latest_warning)

    def show(self, fps=25.0):
        from matplotlib.animation import FuncAnimation

        def animate(_frame):
            if self.controller.running:
                try:
                    self.controller.tick()
                    self._refresh()
                except StepRejected as exc:
                    self.controller.pause()
                    self.run_button.label.set_text("Run")
                    self.status.set_text(str(exc))
            return (self.primary_image, self.secondary_image)

        animation = FuncAnimation(
            self.figure, animate, interval=max(1, int(1000.0 / float(fps))),
            blit=False, cache_frame_data=False
        )
        self.figure._continuum_animation = animation
        self.plt.show()
