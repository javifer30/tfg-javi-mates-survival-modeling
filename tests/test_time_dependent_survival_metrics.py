import numpy as np
import pandas as pd

from src.evaluation.time_dependent_survival import antolini_ctd, survival_to_cumulative_incidence, time_to_index


def test_survival_to_cumulative_incidence_transposes_patient_curves():
    survival = pd.DataFrame(
        [[0.9, 0.8], [0.7, 0.5], [0.4, 0.2]],
        index=[1.0, 2.0, 3.0],
    )

    result = survival_to_cumulative_incidence(survival)

    expected = np.array([[0.1, 0.3, 0.6], [0.2, 0.5, 0.8]])
    np.testing.assert_allclose(result, expected)


def test_antolini_ctd_uses_time_grid_specific_risk():
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([1, 1, 0])
    time_grid = np.array([1.0, 2.0, 3.0])
    cumulative_incidence = np.array(
        [
            [0.7, 0.9, 1.0],
            [0.2, 0.8, 1.0],
            [0.1, 0.5, 1.0],
        ]
    )

    result = antolini_ctd(time, event, cumulative_incidence, time_grid)

    assert result["ctd"] == 1.0
    assert result["comparable_pairs"] == 3.0


def test_time_to_index_uses_first_grid_time_at_or_after_value():
    time_grid = np.array([0.5, 2.0, 5.0])

    assert time_to_index(0.4, time_grid) == 0
    assert time_to_index(2.0, time_grid) == 1
    assert time_to_index(3.0, time_grid) == 2
