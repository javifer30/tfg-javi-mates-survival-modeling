import json

import numpy as np
import pandas as pd
import torch

import scripts.tune_dynamic_deephit_faithful_72h as tuning_module
from scripts.tune_dynamic_deephit_faithful_72h import (
    _completed_signatures,
    _next_config_index,
    build_run_config,
    candidate_signature,
    normalize_candidate,
    select_candidates,
)
from src.models.dynamic_72h.discretization import discretize_duration_event
from src.models.dynamic_72h.dynamic_deephit_faithful import DynamicDeepHitFaithful72h
from src.models.dynamic_72h.losses import pmf_nll
from src.models.dynamic_72h.train_dynamic_deephit_faithful import _save_predictions
from src.models.dynamic_72h.train_dysurv_faithful import FaithfulSplit, build_faithful_input


def _split(n=4, temporal_features=3, static_features=2):
    durations = np.linspace(1.0, 10.0, n, dtype="float32")
    events = np.asarray([i % 2 for i in range(n)], dtype="int64")
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


def _model(input_dim=5, temporal_dim=3):
    return DynamicDeepHitFaithful72h(
        input_dim=input_dim,
        temporal_dim=temporal_dim,
        output_dim=11,
        hidden_rnn=12,
        long_param={"layers": [12], "dropout": 0.0, "activation": "ReLU"},
        att_param={"layers": [12], "dropout": 0.0, "activation": "ReLU"},
        cs_param={"layers": [12], "dropout": 0.0, "activation": "ReLU"},
    )


def test_model_shapes_pmf_tail_and_individualization():
    torch.manual_seed(3)
    split = _split()
    x = torch.from_numpy(build_faithful_input(split.x_seq, split.x_static, "temporal_plus_static_repeated"))
    model = _model()
    model.eval()
    output = model(x)
    assert output["longitudinal_prediction"].shape == (4, 72, 3)
    assert output["attention"].shape == (4, 72)
    assert output["pmf"].shape == (4, 11)
    assert torch.allclose(output["pmf"].sum(dim=1), torch.ones(4), atol=1e-6)
    assert torch.all(output["pmf"][:, -1] > 0)
    assert torch.equal(model.predict_pmf(x[:1]), model.predict_pmf(x[:1]))
    assert not torch.equal(model.predict_pmf(x[:1]), model.predict_pmf(x[1:2]))
    perturbed = x[:1].clone()
    perturbed[:, -1, 0] += 5.0
    assert not torch.equal(model.predict_pmf(x[:1]), model.predict_pmf(perturbed))


def test_prediction_orientation_preserves_patient_order(tmp_path):
    split = _split(n=3)
    pmf = np.full((3, 11), 0.01, dtype="float32")
    pmf[:, -1] = np.asarray([0.30, 0.20, 0.10], dtype="float32")
    pmf[:, :10] *= ((1.0 - pmf[:, -1]) / pmf[:, :10].sum(axis=1))[:, None]
    survival = 1.0 - np.cumsum(pmf[:, :10], axis=1)
    _save_predictions(tmp_path, split, {"survival": survival, "pmf": pmf}, 10)
    result = pd.read_parquet(tmp_path / "predictions" / "validation_survival_predictions.parquet")
    assert result["patient_id"].tolist() == split.patient_ids.tolist()
    assert np.allclose(result["risk10"], 1.0 - survival[:, -1])
    assert np.allclose(result["tail_probability"], pmf[:, -1])


def test_tuning_config_never_loads_test(tmp_path):
    base = {
        "tuning": {"include_test": False},
        "experiment": {},
        "data": {},
        "evaluation": {},
        "collapse": {},
        "model": {"fixed": {}},
        "paths": {"prepared_dataset_dir": "unused"},
    }
    params = normalize_candidate({"loss_weights": {"alpha_ranking": 0.1, "beta_nll": 0.5}})
    config = build_run_config(base, "cfg", params, tmp_path, 42, None, "cpu", False)
    assert config["phase"] == "tuning"
    assert config["include_test"] is False


def test_selection_prefers_noncollapsed_candidate():
    rows = [
        {"status": "completed", "config_id": "collapsed", "validation_ctd_antolini": 0.80, "validation_ibll": 0.40, "collapse_suspected": True},
        {"status": "completed", "config_id": "stable", "validation_ctd_antolini": 0.79, "validation_ibll": 0.42, "collapse_suspected": False},
    ]
    selection = select_candidates(rows)
    assert selection["metric_best"]["config_id"] == "collapsed"
    assert selection["selected"]["config_id"] == "stable"


def test_resume_helpers_continue_config_ids():
    params = {"learning_rate": 0.001, "alpha_ranking": 0.1, "beta_nll": 0.5}
    rows = [{"status": "completed", "config_id": "dynamic_deephit_faithful_cfg_016", "hyperparameters": json.dumps(params, sort_keys=True)}]
    assert candidate_signature(params) in _completed_signatures(rows)
    assert _next_config_index(rows) == 17


def test_resume_trains_only_new_candidate(tmp_path, monkeypatch):
    existing_params = {"learning_rate": 0.001, "alpha_ranking": 0.1, "beta_nll": 0.5}
    pd.DataFrame([{
        "status": "completed",
        "config_id": "dynamic_deephit_faithful_cfg_016",
        "hyperparameters": json.dumps(existing_params, sort_keys=True),
        "validation_ctd_antolini": 0.70,
        "validation_ibll": 0.50,
        "collapse_suspected": False,
    }]).to_csv(tmp_path / "tuning_results.csv", index=False)
    base = {
        "paths": {"outputs_dir": str(tmp_path), "prepared_dataset_dir": "unused"},
        "tuning": {"seed": 42, "include_test": False, "grid": {
            "learning_rate": [0.001],
            "loss_weights": [
                {"alpha_ranking": 0.1, "beta_nll": 0.5},
                {"alpha_ranking": 0.2, "beta_nll": 0.6},
            ],
        }},
        "experiment": {}, "data": {}, "evaluation": {}, "collapse": {}, "model": {"fixed": {}},
    }
    calls = []
    monkeypatch.setattr(tuning_module, "load_yaml", lambda _: base)
    monkeypatch.setattr(tuning_module, "train_dynamic_deephit_faithful", lambda config, logger: calls.append(config["run"]["config_id"]) or {
        "splits": {"validation": {"ctd_antolini": 0.71, "ibs": 0.2, "ibll": 0.45, "nbll": 0.45, "mean_horizon_c_index": 0.72}},
        "collapse": {"collapse_suspected": False, "std_risk10": 0.1, "range_risk10": 0.4, "mean_tail_probability": 0.2, "number_unique_risk10_rounded_6": 100},
    })
    tuning_module.tune("unused", device="cpu", resume=True)
    assert calls == ["dynamic_deephit_faithful_cfg_017"]


def test_tiny_overfit_reduces_nll_and_individualizes_risk():
    torch.manual_seed(9)
    n = 64
    signal = torch.linspace(-2.0, 2.0, n)
    x = torch.randn(n, 72, 4) * 0.02
    x[:, :, 0] += signal[:, None]
    t_idx = torch.where(signal > 0, torch.zeros(n, dtype=torch.long), torch.full((n,), 9, dtype=torch.long))
    event = (signal > 0).long()
    model = _model(input_dim=4, temporal_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial = None
    for _ in range(70):
        optimizer.zero_grad()
        pmf = model(x)["pmf"]
        loss = pmf_nll(pmf, t_idx, event)
        initial = float(loss) if initial is None else initial
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        pmf = model.predict_pmf(x)
        final = float(pmf_nll(pmf, t_idx, event))
        risk10 = pmf[:, :10].sum(dim=1)
    assert final < initial * 0.75
    assert float(risk10.std()) > 0.005
    assert int(torch.unique(torch.round(risk10 * 1e6)).numel()) > 10
