from __future__ import annotations

import napari

from sim_viewer.plugin import make_dock_widget


def main() -> None:
    viewer = napari.Viewer()
    widget = make_dock_widget(viewer)
    viewer.window.add_dock_widget(widget, name="Cell Sphere Sim")
    # viewer.points_layer.shading = "spherical"
    napari.run()


if __name__ == "__main__":
    main()
