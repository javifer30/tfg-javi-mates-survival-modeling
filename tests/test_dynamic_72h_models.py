import numpy as np

from src.models.dynamic_72h.data import Dynamic72hSplit, build_model_input, input_metadata
from src.models.dynamic_72h.discretization import discretize_duration_event


def _split():
    durations = np.asarray([0.2, 1.0, 1.1, 10.0], dtype="float32")
    events = np.asarray([1, 1, 0, 0], dtype="int64")
    t_idx, event = discretize_duration_event(durations, events)
    return Dynamic72hSplit(
        name="train",
        patient_ids=np.asarray([1, 2, 3, 4]),
        x_seq=np.ones((4, 72, 3), dtype="float32"),
        m_seq=np.zeros((4, 72, 3), dtype="float32"),
        x_static=np.ones((4, 2), dtype="float32"),
        duration_eval_days=durations,
        duration_rel_days=durations,
        event_eval=events,
        t_idx=t_idx,
        event=event,
    )


def test_dynamic_72h_discretization_daily_bins():
    idx, event = discretize_duration_event([0.2, 1.0, 1.1, 9.9, 10.0], [1, 1, 1, 1, 0])
    assert idx.tolist() == [0, 0, 1, 9, 9]
    assert event.tolist() == [1, 1, 1, 1, 0]


def test_values_plus_mask_plus_static_input_shape():
    split = _split()
    x = build_model_input(split, "values_plus_mask_plus_static")
    assert x.shape == (4, 72, 8)
    meta = input_metadata(split, "values_plus_mask_plus_static")
    assert meta["n_temporal_features"] == 3
    assert meta["n_mask_features"] == 3
    assert meta["n_static_features"] == 2
    assert meta["n_model_input_features"] == 8
