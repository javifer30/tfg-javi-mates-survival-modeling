import numpy as np
import pandas as pd

from src.data.static_dataset import (
    EVENT_COL,
    ID_COL,
    SPLIT_COL,
    TIME_COL,
    StaticPreprocessor,
    feature_columns,
    validate_static_datasets,
)
from src.models.deephit_tfg import (
    build_deephit_masks,
    deephit_event_probability_and_survival,
    deephit_loss,
    discretize_time,
)
from src.models.static_common import split_xy


def _raw_static_df():
    return pd.DataFrame(
        {
            ID_COL: np.arange(1, 13),
            "gender": ["M", "F"] * 6,
            "age": [50, 70, 65, 44, 81, 52, 60, 75, 68, 49, 55, 73],
            "ethnicity": ["WHITE", "BLACK", "WHITE", "ASIAN", "WHITE", "HISPANIC"] * 2,
            "first_careunit": ["MICU", "SICU", "MICU", "CCU", "MICU", "SICU"] * 2,
            "admission_location": ["ER", "ER", "TRANSFER", "ER", "TRANSFER", "ER"] * 2,
            "insurance": ["Medicare", "Private", "Medicaid", "Medicare", "Private", "Medicare"] * 2,
            "hour": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "eyes": [4, 3, 4, np.nan, 2, 4, 3, 4, 2, 1, 4, 3],
            "height": [170, 165, np.nan, 180, 160, 172, 169, np.nan, 175, 168, 171, 166],
            "motor": [6, 5, 6, 4, 6, 5, 4, 6, 5, 6, 4, 6],
            "verbal": [5, 4, 5, 3, 5, 4, 3, 5, 4, 5, 3, 4],
            "weight": [80, 70, 65, np.nan, 90, 78, 82, 77, 69, 73, 88, 66],
            TIME_COL: [1, 2, 3, 4, 5, 6, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            EVENT_COL: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def _preprocessor():
    return StaticPreprocessor(
        categorical_cols=["ethnicity", "first_careunit", "admission_location", "insurance"],
        standard_cols=["height"],
        minmax_cols=["weight", "age", "hour", "eyes", "motor", "verbal"],
        binary_maps={"gender": {"M": 1.0, "F": 0.0}},
        rare_min_count=1,
    )


def _processed_splits():
    raw = _raw_static_df()
    train_raw = raw.iloc[:6].copy()
    val_raw = raw.iloc[6:9].copy()
    test_raw = raw.iloc[9:].copy()
    prep = _preprocessor().fit(train_raw)
    return (
        prep.transform(train_raw, "train"),
        prep.transform(val_raw, "validation"),
        prep.transform(test_raw, "test"),
    )


def test_no_patient_overlap_between_static_splits():
    train, val, test = _processed_splits()
    validate_static_datasets(train, val, test)


def test_observed_event_is_binary_and_time_positive():
    for df in _processed_splits():
        assert set(df[EVENT_COL].unique()).issubset({0, 1})
        assert (df[TIME_COL] > 0).all()


def test_same_columns_and_no_nulls_after_preprocessing():
    train, val, test = _processed_splits()
    assert list(train.columns) == list(val.columns) == list(test.columns)
    assert not train.isna().any().any()
    assert not val.isna().any().any()
    assert not test.isna().any().any()


def test_dataset_is_compatible_with_static_model_inputs():
    train, _, _ = _processed_splits()
    x, time, event, ids = split_xy(train)
    assert x.shape[0] == len(train)
    assert x.shape[1] == len(feature_columns(train))
    assert time.shape[0] == event.shape[0] == ids.shape[0]


def test_deephit_masks_have_expected_shapes():
    time = np.array([0.5, 3.0, 10.0])
    event = np.array([1, 0, 1])
    bins = discretize_time(time, max_horizon_days=10, num_categories=10)
    mask1, mask2 = build_deephit_masks(bins, event, num_events=1, num_categories=10)
    assert mask1.shape == (3, 1, 10)
    assert mask2.shape == (3, 10)
    assert mask1[0, 0, bins[0]] == 1.0
    assert mask2[0, : bins[0] + 1].all()


def test_deephit_tail_mask_covers_censored_horizon():
    time = np.array([10.0])
    event = np.array([0])
    bins = discretize_time(time, max_horizon_days=10, num_categories=10)
    mask1, mask2 = build_deephit_masks(
        bins,
        event,
        num_events=1,
        num_categories=10,
        include_tail_category=True,
    )
    assert mask1.shape == (1, 1, 11)
    assert mask2.shape == (1, 11)
    assert mask1[0, 0, 10] == 1.0
    assert mask1.sum() == 1.0
    assert mask2[0, :10].all()
    assert mask2[0, 10] == 0.0


def test_deephit_tail_probability_preserves_final_survival():
    pred = np.array([[[0.10, 0.20, 0.15, 0.25, 0.30]]])
    event_prob, survival, tail = deephit_event_probability_and_survival(
        pred,
        num_categories=4,
        include_tail_category=True,
    )
    assert event_prob.shape == (1, 4)
    assert np.isclose(event_prob.sum() + tail[0], 1.0)
    assert np.isclose(survival[0, -1], tail[0])


def test_deephit_ranking_loss_uses_event_subject_time():
    import torch

    pred = torch.tensor(
        [
            [[0.1, 0.1, 0.1, 0.7]],
            [[0.2, 0.1, 0.1, 0.6]],
            [[0.4, 0.1, 0.1, 0.4]],
        ],
        dtype=torch.float32,
    )
    time_bins = torch.tensor([0, 1, 2], dtype=torch.long)
    event = torch.tensor([1, 0, 1], dtype=torch.long)
    mask1_np, mask2_np = build_deephit_masks(
        time_bins.numpy(),
        event.numpy(),
        num_events=1,
        num_categories=3,
        include_tail_category=True,
    )
    loss = deephit_loss(
        pred,
        time_bins,
        event,
        torch.tensor(mask1_np, dtype=torch.float32),
        torch.tensor(mask2_np, dtype=torch.float32),
        alpha=0.0,
        beta=1.0,
        gamma=0.0,
        ranking_sigma=1.0,
    )
    expected = (np.exp(0.1) + np.exp(0.3)) / 9.0
    assert np.isclose(float(loss), expected)
