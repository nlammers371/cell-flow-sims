import numpy as np

from cell_sphere_sim.io.stores import PandasTracksStore


def test_tracks_store_columns_and_rows():
    store = PandasTracksStore()
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    v = np.zeros_like(x)
    state_id = np.array([0, 1], dtype=np.int32)
    track_id = np.array([10, 11], dtype=np.int64)
    parent_id = np.array([-1, 10], dtype=np.int64)

    store.append(t=0.0, x=x, v=v, state_id=state_id, track_id=track_id, extra={"parent_id": parent_id})
    df = store.to_dataframe()

    expected_cols = {
        "track_id",
        "t",
        "z",
        "y",
        "x",
        "vz",
        "vy",
        "vx",
        "state_id",
        "parent_id",
    }
    assert expected_cols.issubset(df.columns)
    assert len(df) == 2
