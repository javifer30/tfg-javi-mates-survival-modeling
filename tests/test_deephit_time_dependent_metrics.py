import numpy as np

from src.evaluation.deephit_time_dependent import antolini_ctd, weighted_c_index_at_horizon


def test_antolini_ctd_uses_event_time_specific_cif():
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([1, 1, 0])
    cumulative_incidence = np.array(
        [
            [0.7, 0.9, 1.0],
            [0.2, 0.8, 1.0],
            [0.1, 0.5, 1.0],
        ]
    )

    result = antolini_ctd(time, event, cumulative_incidence, max_horizon_days=3, num_categories=3)

    assert result["ctd"] == 1.0
    assert result["comparable_pairs"] == 3.0


def test_weighted_c_index_at_horizon_matches_reference_pair_logic():
    train_time = np.array([1.0, 2.0, 3.0])
    train_event = np.array([1, 1, 0])
    test_time = np.array([1.0, 2.0, 3.0])
    test_event = np.array([1, 1, 0])
    risk = np.array([0.8, 0.4, 0.1])

    result = weighted_c_index_at_horizon(train_time, train_event, test_time, test_event, risk, horizon=2.0)

    assert result["weighted_c_index"] == 1.0
    assert result["weighted_comparable_pairs"] > 0
