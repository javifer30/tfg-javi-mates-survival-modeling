import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import scripts.landmark_dysurv_static_faithful_tuning_impl as tuning_module
from scripts.landmark_dysurv_static_faithful_tuning_impl import (
    _completed_signatures,
    _next_config_index,
    build_run_config,
    candidate_signature,
    normalize_candidate,
    select_candidates,
)
from src.models.landmark_dynamic.discretization import discretize_duration_event
from src.models.landmark_dynamic.dysurv_static_faithful import DySurvStaticFaithful72h
from src.models.landmark_dynamic.losses import hazards_to_survival, logistic_hazard_nll
from src.models.landmark_dynamic.train_dysurv_static_faithful import (
    StaticFaithfulDataset,
    StaticFaithfulSplit,
    _save_predictions,
    load_static_faithful_split,
    validate_static_splits,
)


def _split(n=4, features=5, name="validation"):
    durations = np.linspace(1.0, 10.0, n, dtype="float32")
    events = np.asarray([i % 2 for i in range(n)], dtype="int64")
    t_idx, events = discretize_duration_event(durations, events)
    return StaticFaithfulSplit(
        name=name,
        patient_ids=np.asarray([f"{name}-p{i}" for i in range(n)]),
        x_static=np.arange(n * features, dtype="float32").reshape(n, features) / 10.0,
        durations=durations,
        events=events,
        t_idx=t_idx,
    )


def test_static_model_shapes_decoder_and_individualization():
    torch.manual_seed(7)
    x = torch.from_numpy(_split().x_static)
    model = DySurvStaticFaithful72h(input_dim=5, latent_dim=6, dropout=0.0)
    model.eval()
    output = model(x)
    survival = hazards_to_survival(output["logits"])
    assert output["mu"].shape == (4, 6)
    assert output["logvar"].shape == (4, 6)
    assert output["z"].shape == (4, 6)
    assert output["reconstruction"].shape == (4, 5)
    assert output["logits"].shape == (4, 10)
    assert survival.shape == (4, 10)
    assert not any(isinstance(module, nn.ReLU) for module in model.decoder.modules())
    assert torch.equal(model.predict_logits(x[:1]), model.predict_logits(x[:1]))
    assert not torch.equal(model.predict_logits(x[:1]), model.predict_logits(x[1:2]))
    perturbed = x[:1].clone()
    perturbed[:, 0] += 5.0
    assert not torch.equal(model.predict_logits(x[:1]), model.predict_logits(perturbed))


def test_static_loader_uses_same_npz_but_exposes_no_temporal_or_target_features(tmp_path):
    patient_ids = np.asarray(["p1", "p2"])
    np.savez_compressed(
        tmp_path / "train_dynamic_landmark.npz",
        patient_ids=patient_ids,
        X_static=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype="float32"),
        X_seq=np.full((2, 72, 3), 9999.0, dtype="float32"),
        M_seq=np.ones((2, 72, 3), dtype="float32"),
        duration_eval_days=np.asarray([2.0, 10.0], dtype="float32"),
        event_eval=np.asarray([1, 0], dtype="int64"),
    )
    split = load_static_faithful_split(tmp_path, "train")
    item = StaticFaithfulDataset(split)[0]
    assert split.patient_ids.tolist() == patient_ids.tolist()
    assert set(item) == {"x_static", "t_idx", "event", "row_index"}
    assert item["x_static"].shape == (2,)
    assert float(item["x_static"].max()) < 9999.0


def test_split_validation_confirms_faithful_source_and_no_overlap(tmp_path):
    (tmp_path / "preprocessing_metadata.json").write_text(
        json.dumps({
            "dataset": "dysurv_faithful_72h",
            "static_preprocessing": "train_mean_standardization",
            "imputation_fit_split": "train",
        }),
        encoding="utf-8",
    )
    train = _split(name="train")
    validation = _split(name="validation")
    checks = validate_static_splits(train, validation, None, tmp_path)
    assert checks["same_faithful_dataset_files"] is True
    assert checks["temporal_input_loaded"] is False
    assert checks["mask_input_loaded"] is False
    assert checks["target_not_in_input"] is True
    assert checks["preprocessing_fit_split"] == "train"


def test_prediction_orientation_preserves_patient_order(tmp_path):
    split = _split(n=3)
    survival = np.asarray([
        np.linspace(0.99, 0.80, 10),
        np.linspace(0.95, 0.60, 10),
        np.linspace(0.90, 0.40, 10),
    ], dtype="float32")
    _save_predictions(tmp_path, split, survival)
    result = pd.read_parquet(tmp_path / "predictions" / "validation_survival_predictions.parquet")
    assert result["patient_id"].tolist() == split.patient_ids.tolist()
    assert np.allclose(result["risk10"], 1.0 - survival[:, -1])
    assert np.allclose(result["survival_day_1"], survival[:, 0])


def test_tuning_config_blocks_test_and_normalizes_weights():
    base = {
        "tuning": {"include_test": False},
        "experiment": {},
        "data": {},
        "evaluation": {},
        "collapse": {},
        "model": {"fixed": {}},
        "paths": {"prepared_dataset_dir": "unused"},
    }
    params = normalize_candidate({"loss_weights": {"w_surv": 0.8, "w_recon": 0.15, "w_kl": 0.05}})
    config = build_run_config(base, "cfg", params, Path("unused"), 42, None, "cpu", False)
    assert config["phase"] == "tuning"
    assert config["include_test"] is False
    assert params["w_surv"] + params["w_recon"] + params["w_kl"] == 1.0


def test_tuning_selection_prefers_noncollapsed_candidate():
    rows = [
        {"status": "completed", "config_id": "collapsed", "validation_ctd_antolini": 0.80, "validation_ibll": 0.40, "collapse_suspected": True},
        {"status": "completed", "config_id": "stable", "validation_ctd_antolini": 0.79, "validation_ibll": 0.42, "collapse_suspected": False},
    ]
    selection = select_candidates(rows)
    assert selection["metric_best"]["config_id"] == "collapsed"
    assert selection["selected"]["config_id"] == "stable"


def test_resume_helpers_continue_static_config_ids():
    params = {"learning_rate": 0.001, "w_surv": 0.7, "w_recon": 0.2, "w_kl": 0.1}
    rows = [{"status": "completed", "config_id": "dysurv_static_faithful_cfg_016", "hyperparameters": json.dumps(params)}]
    assert candidate_signature(params) in _completed_signatures(rows)
    assert _next_config_index(rows) == 17


def test_resume_trains_only_new_candidate(tmp_path, monkeypatch):
    old = {"learning_rate": 0.001, "w_surv": 0.7, "w_recon": 0.2, "w_kl": 0.1}
    pd.DataFrame([{
        "status": "completed",
        "config_id": "dysurv_static_faithful_cfg_016",
        "hyperparameters": json.dumps(old, sort_keys=True),
        "validation_ctd_antolini": 0.70,
        "validation_ibll": 0.60,
        "collapse_suspected": False,
    }]).to_csv(tmp_path / "tuning_results.csv", index=False)
    base = {
        "paths": {"outputs_dir": str(tmp_path), "prepared_dataset_dir": "unused"},
        "tuning": {"seed": 42, "include_test": False, "grid": {
            "learning_rate": [0.001],
            "loss_weights": [old | {}, {"w_surv": 0.8, "w_recon": 0.15, "w_kl": 0.05}],
        }},
        "experiment": {}, "data": {}, "evaluation": {}, "collapse": {}, "model": {"fixed": {}},
    }
    calls = []

    def fake_train(config, logger):
        calls.append(config["run"]["config_id"])
        return {
            "splits": {"validation": {"ctd_antolini": 0.71, "ibs": 0.2, "ibll": 0.5, "nbll": 0.5, "mean_horizon_c_index": 0.72}},
            "collapse": {"collapse_suspected": False, "std_risk10": 0.1, "range_risk10": 0.5, "std_mu": 0.2, "kl_loss": 1.0, "number_unique_risk10_rounded_6": 100},
        }

    monkeypatch.setattr(tuning_module, "load_yaml", lambda _: base)
    monkeypatch.setattr(tuning_module, "train_dysurv_static_faithful", fake_train)
    tuning_module.tune("unused", device="cpu", resume=True)
    assert calls == ["dysurv_static_faithful_cfg_017"]


def test_tiny_overfit_reduces_survival_and_reconstruction_losses():
    torch.manual_seed(11)
    n, features = 64, 6
    signal = torch.linspace(-2.0, 2.0, n)
    x = torch.randn(n, features) * 0.02
    x[:, 0] += signal
    t_idx = torch.where(signal > 0, torch.zeros(n, dtype=torch.long), torch.full((n,), 9, dtype=torch.long))
    event = (signal > 0).long()
    model = DySurvStaticFaithful72h(input_dim=features, latent_dim=5, dropout=0.0, decoder_activation="relu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    with torch.no_grad():
        initial_output = model(x)
        initial_surv = float(logistic_hazard_nll(initial_output["logits"], t_idx, event))
        initial_recon = float(torch.mean((initial_output["reconstruction"] - x) ** 2))
    model.train()
    # Stop before the deliberately binary synthetic target saturates at 0/1.
    for _ in range(40):
        optimizer.zero_grad()
        output = model(x)
        loss_surv = logistic_hazard_nll(output["logits"], t_idx, event)
        loss_recon = torch.mean((output["reconstruction"] - x) ** 2)
        (0.8 * loss_surv + 0.2 * loss_recon).backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        output = model(x)
        final_surv = float(logistic_hazard_nll(output["logits"], t_idx, event))
        final_recon = float(torch.mean((output["reconstruction"] - x) ** 2))
        risk10 = 1.0 - hazards_to_survival(output["logits"])[:, -1]
    assert final_surv < initial_surv * 0.75
    assert final_recon < initial_recon
    assert float(risk10.std()) > 0.005
    assert int(torch.unique(torch.round(risk10 * 1e6)).numel()) > 10
