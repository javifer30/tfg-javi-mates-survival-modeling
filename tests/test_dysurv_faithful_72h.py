import json

import numpy as np
import pandas as pd
import torch

import scripts.tune_dysurv_faithful_72h as tuning_module
from scripts.tune_dysurv_faithful_72h import (
    _completed_signatures,
    _next_config_index,
    build_run_config,
    candidate_signature,
    normalize_candidate,
    select_candidates,
)
from src.data.dysurv_faithful_72h_dataset import prepare_arrays
from src.models.dynamic_72h.discretization import discretize_duration_event
from src.models.dynamic_72h.dysurv_faithful import DySurvFaithful72h
from src.models.dynamic_72h.losses import hazards_to_survival, logistic_hazard_nll
from src.models.dynamic_72h.train_dysurv_faithful import (
    FaithfulDataset,
    FaithfulSplit,
    _save_predictions,
    build_faithful_input,
)


def _split(n=4, temporal_features=3, static_features=2):
    durations = np.linspace(1.0, 10.0, n, dtype="float32")
    events = np.asarray([(i % 2) for i in range(n)], dtype="int64")
    t_idx, events = discretize_duration_event(durations, events)
    x_seq = np.arange(n * 72 * temporal_features, dtype="float32").reshape(n, 72, temporal_features) / 100.0
    return FaithfulSplit(
        name="validation",
        patient_ids=np.asarray([f"p{i}" for i in range(n)]),
        x_seq=x_seq,
        m_seq=np.ones_like(x_seq),
        x_static=np.arange(n * static_features, dtype="float32").reshape(n, static_features),
        durations=durations,
        events=events,
        t_idx=t_idx,
    )


def test_faithful_model_shapes_and_individualization():
    torch.manual_seed(7)
    split = _split()
    x = torch.from_numpy(build_faithful_input(split.x_seq, split.x_static, "temporal_plus_static_repeated"))
    model = DySurvFaithful72h(
        input_dim=5,
        reconstruction_dim=3,
        rnn_hidden_dim=12,
        latent_dim=5,
        encoder_mlp=[18, 24, 18],
        survival_mlp=[18, 24, 18],
        dropout=0.0,
    )
    model.eval()
    output = model(x)
    survival = hazards_to_survival(output["logits"])
    assert output["reconstruction"].shape == (4, 72, 3)
    assert output["mu"].shape == (4, 5)
    assert output["logvar"].shape == (4, 5)
    assert output["logits"].shape == (4, 10)
    assert survival.shape == (4, 10)
    assert torch.equal(model.predict_logits(x[:1]), model.predict_logits(x[:1]))
    assert not torch.equal(model.predict_logits(x[:1]), model.predict_logits(x[1:2]))
    perturbed = x[:1].clone()
    perturbed[:, -1, 0] += 5.0
    assert not torch.equal(model.predict_logits(x[:1]), model.predict_logits(perturbed))


def test_imputation_is_train_only_and_masks_are_not_input_channels():
    def arrays(value, observed):
        x = np.full((2, 72, 2), value, dtype="float32")
        mask = np.zeros_like(x)
        mask[:, observed, :] = 1.0
        return {
            "patient_ids": np.asarray([f"{value}-a", f"{value}-b"]),
            "X_seq": x,
            "M_seq": mask,
            "X_static": np.asarray([[value, 1.0], [value + 1.0, 0.0]], dtype="float32"),
            "duration_eval_days": np.asarray([2.0, 10.0], dtype="float32"),
            "duration_rel_days": np.asarray([2.0, 10.0], dtype="float32"),
            "event_eval": np.asarray([1, 0], dtype="int64"),
        }

    source = {
        "train": arrays(2.0, 5),
        "validation": arrays(1000.0, 5),
        "test": arrays(2000.0, 5),
    }
    prepared, stats = prepare_arrays(source)
    assert stats["temporal_residual_medians"] == [2.0, 2.0]
    assert np.all(prepared["train"]["X_seq"] == 2.0)
    x = build_faithful_input(prepared["train"]["X_seq"], prepared["train"]["X_static"], "temporal_plus_static_repeated")
    assert x.shape[2] == 4
    split = _split()
    item = FaithfulDataset(split, "temporal_only")[0]
    assert "duration" not in item and "event" in item
    assert item["x_input"].shape[-1] == split.x_seq.shape[-1]
    assert item["m_temporal"].shape == item["x_temporal"].shape


def test_prediction_orientation_preserves_patient_order(tmp_path):
    split = _split(n=3)
    survival = np.asarray(
        [
            np.linspace(0.99, 0.80, 10),
            np.linspace(0.95, 0.60, 10),
            np.linspace(0.90, 0.40, 10),
        ],
        dtype="float32",
    )
    _save_predictions(tmp_path, split, survival)
    result = pd.read_parquet(tmp_path / "predictions" / "validation_survival_predictions.parquet")
    assert result["patient_id"].tolist() == split.patient_ids.tolist()
    assert np.allclose(result["risk10"], 1.0 - survival[:, -1])
    assert np.allclose(result["survival_day_1"], survival[:, 0])


def test_tuning_config_never_loads_test():
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
    config = build_run_config(base, "cfg", params, tmp_path := __import__("pathlib").Path("unused"), 42, None, "cpu", False)
    assert config["phase"] == "tuning"
    assert config["include_test"] is False


def test_tuning_selection_prefers_noncollapsed_candidate():
    rows = [
        {
            "status": "completed",
            "config_id": "collapsed",
            "validation_ctd_antolini": 0.80,
            "validation_ibll": 0.40,
            "collapse_suspected": True,
        },
        {
            "status": "completed",
            "config_id": "individualized",
            "validation_ctd_antolini": 0.79,
            "validation_ibll": 0.42,
            "collapse_suspected": False,
        },
    ]
    selection = select_candidates(rows)
    assert selection["metric_best"]["config_id"] == "collapsed"
    assert selection["selected"]["config_id"] == "individualized"


def test_resume_matches_hyperparameters_and_continues_config_ids():
    existing_params = {
        "learning_rate": 0.001,
        "w_surv": 0.7,
        "w_recon": 0.2,
        "w_kl": 0.1,
    }
    rows = [
        {
            "status": "completed",
            "config_id": "dysurv_faithful_cfg_016",
            "hyperparameters": json.dumps(existing_params, sort_keys=True),
        },
        {
            "status": "failed",
            "config_id": "dysurv_faithful_cfg_020",
            "hyperparameters": json.dumps({"learning_rate": 0.0005}, sort_keys=True),
        },
    ]
    assert candidate_signature(existing_params) in _completed_signatures(rows)
    assert candidate_signature({"learning_rate": 0.0005}) not in _completed_signatures(rows)
    assert _next_config_index(rows) == 21


def test_resume_trains_only_new_candidates_and_appends_results(tmp_path, monkeypatch):
    existing_params = {"learning_rate": 0.001, "w_surv": 0.7, "w_recon": 0.2, "w_kl": 0.1}
    existing = {
        "status": "completed",
        "config_id": "dysurv_faithful_cfg_016",
        "hyperparameters": json.dumps(existing_params, sort_keys=True),
        "validation_ctd_antolini": 0.70,
        "validation_ibll": 0.60,
        "collapse_suspected": False,
    }
    pd.DataFrame([existing]).to_csv(tmp_path / "tuning_results.csv", index=False)
    base = {
        "paths": {"outputs_dir": str(tmp_path), "prepared_dataset_dir": "unused"},
        "tuning": {
            "seed": 42,
            "include_test": False,
            "grid": {
                "learning_rate": [0.001],
                "loss_weights": [
                    {"w_surv": 0.7, "w_recon": 0.2, "w_kl": 0.1},
                    {"w_surv": 0.333, "w_recon": 0.333, "w_kl": 0.333},
                ],
            },
        },
        "experiment": {},
        "data": {},
        "evaluation": {},
        "collapse": {},
        "model": {"fixed": {}},
    }
    calls = []

    def fake_train(run_config, logger):
        calls.append(run_config["run"]["config_id"])
        return {
            "splits": {
                "validation": {
                    "ctd_antolini": 0.71,
                    "ibs": 0.20,
                    "ibll": 0.50,
                    "nbll": 0.50,
                    "mean_horizon_c_index": 0.72,
                }
            },
            "collapse": {
                "collapse_suspected": False,
                "std_risk10": 0.1,
                "range_risk10": 0.5,
                "std_mu": 0.2,
                "kl_loss": 1.0,
                "number_unique_risk10_rounded_6": 100,
            },
        }

    monkeypatch.setattr(tuning_module, "load_yaml", lambda _: base)
    monkeypatch.setattr(tuning_module, "train_dysurv_faithful", fake_train)
    planned = tuning_module.tune("unused.yaml", device="cpu", resume=True)

    assert calls == ["dysurv_faithful_cfg_017"]
    assert [run["config_id"] for run in planned] == ["dysurv_faithful_cfg_017"]
    results = pd.read_csv(tmp_path / "tuning_results.csv")
    assert results["config_id"].tolist() == ["dysurv_faithful_cfg_016", "dysurv_faithful_cfg_017"]
    selection = json.loads((tmp_path / "best_hyperparameters.json").read_text())
    assert selection["selected"]["config_id"] == "dysurv_faithful_cfg_017"


def test_tiny_overfit_produces_individual_risk():
    torch.manual_seed(11)
    n = 64
    signal = torch.linspace(-2.0, 2.0, n)
    x = torch.randn(n, 72, 4) * 0.02
    x[:, :, 0] += signal[:, None]
    t_idx = torch.where(signal > 0, torch.zeros(n, dtype=torch.long), torch.full((n,), 9, dtype=torch.long))
    event = (signal > 0).long()
    model = DySurvFaithful72h(
        input_dim=4,
        reconstruction_dim=4,
        rnn_hidden_dim=12,
        latent_dim=6,
        encoder_mlp=[24, 32, 24],
        survival_mlp=[24, 32, 24],
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    initial = None
    for _ in range(60):
        optimizer.zero_grad()
        logits = model(x)["logits"]
        loss = logistic_hazard_nll(logits, t_idx, event)
        initial = float(loss) if initial is None else initial
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model.predict_logits(x)
        final_loss = float(logistic_hazard_nll(logits, t_idx, event))
        risk10 = 1.0 - hazards_to_survival(logits)[:, -1]
    assert final_loss < initial * 0.75
    assert float(risk10.std()) > 0.005
    assert int(torch.unique(torch.round(risk10 * 1e6)).numel()) > 10
