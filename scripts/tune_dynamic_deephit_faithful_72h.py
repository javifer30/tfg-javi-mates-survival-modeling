import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dynamic_72h.common import expand_grid, load_yaml, save_json, timestamp_utc
from src.models.dynamic_72h.train_dynamic_deephit_faithful import train_dynamic_deephit_faithful
from src.utils.logger import get_logger


def normalize_candidate(candidate: dict) -> dict:
    params = copy.deepcopy(candidate)
    weights = params.pop("loss_weights")
    for source, target in [("alpha_ranking", "alpha_ranking"), ("beta_nll", "beta_nll")]:
        if source not in weights:
            raise ValueError(f"loss_weights missing {source}: {weights}")
        params[target] = float(weights[source])
    if params["alpha_ranking"] + params["beta_nll"] > 1.0:
        raise ValueError("Dynamic-DeepHit loss weights must sum to at most 1")
    return params


def candidate_signature(params: dict) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def _existing_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


def _completed_signatures(rows: list[dict]) -> set[str]:
    signatures = set()
    for row in rows:
        if row.get("status") != "completed":
            continue
        try:
            signatures.add(candidate_signature(json.loads(row["hyperparameters"])))
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
    return signatures


def _next_config_index(rows: list[dict]) -> int:
    indices = []
    for row in rows:
        match = re.fullmatch(r"dynamic_deephit_faithful_cfg_(\d+)", str(row.get("config_id", "")))
        if match:
            indices.append(int(match.group(1)))
    return max(indices, default=0) + 1


def build_run_config(base: dict, config_id: str, params: dict, output_dir: Path, seed: int, sample_size, device: str, include_test: bool) -> dict:
    if base["tuning"].get("include_test", False):
        raise ValueError("The faithful tuning config must keep include_test=false")
    return {
        "seed": int(seed),
        "device": device,
        "phase": "final" if include_test else "tuning",
        "include_test": bool(include_test),
        "sample_size": sample_size,
        "experiment": copy.deepcopy(base["experiment"]),
        "data": copy.deepcopy(base["data"]),
        "evaluation": copy.deepcopy(base["evaluation"]),
        "collapse": copy.deepcopy(base["collapse"]),
        "model_fixed": copy.deepcopy(base["model"]["fixed"]),
        "params": copy.deepcopy(params),
        "paths": {
            "prepared_dataset_dir": base["paths"]["prepared_dataset_dir"],
            "output_dir": str(output_dir),
        },
        "run": {
            "model": "dynamic_deephit_faithful_72h",
            "config_id": config_id,
            "seed": int(seed),
            "timestamp": timestamp_utc(),
            "hyperparameters": copy.deepcopy(params),
        },
    }


def result_row(config_id: str, params: dict, seed: int, output_dir: Path, sample_size=None, metrics=None, error=None) -> dict:
    metrics = metrics or {}
    validation = metrics.get("splits", {}).get("validation", {})
    collapse = metrics.get("collapse", {})
    return {
        "timestamp": timestamp_utc(),
        "model": "dynamic_deephit_faithful_72h",
        "config_id": config_id,
        "hyperparameters": json.dumps(params, sort_keys=True),
        "seed": int(seed),
        "sample_size": sample_size,
        "status": "completed" if error is None else "failed",
        "error": error,
        "validation_ctd_antolini": validation.get("ctd_antolini", math.nan),
        "validation_ibs": validation.get("ibs", math.nan),
        "validation_ibll": validation.get("ibll", math.nan),
        "validation_nbll": validation.get("nbll", math.nan),
        "validation_mean_horizon_c_index": validation.get("mean_horizon_c_index", math.nan),
        "collapse_suspected": collapse.get("collapse_suspected", True),
        "validation_std_risk10": collapse.get("std_risk10", math.nan),
        "validation_range_risk10": collapse.get("range_risk10", math.nan),
        "validation_mean_tail_probability": collapse.get("mean_tail_probability", math.nan),
        "validation_unique_risk10_rounded_6": collapse.get("number_unique_risk10_rounded_6", 0),
        "test_metrics_recorded": "test" in metrics.get("splits", {}),
        "output_dir": str(output_dir),
    }


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _sort_key(row: dict):
    ctd = float(row["validation_ctd_antolini"])
    ibll = float(row["validation_ibll"])
    return (-ctd, ibll if math.isfinite(ibll) else math.inf, row["config_id"])


def select_candidates(rows: list[dict]) -> dict:
    successful = [row for row in rows if row["status"] == "completed" and math.isfinite(float(row["validation_ctd_antolini"]))]
    if not successful:
        raise ValueError("No successful Dynamic-DeepHit faithful candidate with finite validation Ctd")
    metric_best = sorted(successful, key=_sort_key)[0]
    noncollapsed = [row for row in successful if not _as_bool(row["collapse_suspected"])]
    selected = sorted(noncollapsed, key=_sort_key)[0] if noncollapsed else metric_best
    return {
        "status": "selected" if noncollapsed else "review_required_all_candidates_collapsed",
        "selection_rule": "best validation Ctd, validation IBLL tiebreaker, preferring non-collapsed candidates",
        "selected": selected,
        "metric_best": metric_best,
        "non_collapsed_alternative": sorted(noncollapsed, key=_sort_key)[0] if noncollapsed else None,
    }


def tune(config_path: str, dry_run=False, max_runs=None, sample_size=None, device="auto", force=False, resume=False):
    if force and resume:
        raise ValueError("--force and --resume cannot be used together")
    base = load_yaml(config_path)
    logger = get_logger("tune_dynamic_deephit_faithful_72h")
    canonical_root = Path(base["paths"]["outputs_dir"])
    output_root = canonical_root / "smoke" if sample_size is not None else canonical_root
    seed = int(base["tuning"]["seed"])
    results_path = output_root / "tuning_results.csv"
    existing_rows = _existing_results(results_path) if resume else []
    completed_signatures = _completed_signatures(existing_rows)
    next_index = _next_config_index(existing_rows)
    candidates = expand_grid(base["tuning"]["grid"])
    rows, planned = [], []
    for index, raw_candidate in enumerate(candidates, start=1):
        params = normalize_candidate(raw_candidate)
        signature = candidate_signature(params)
        if resume and signature in completed_signatures:
            logger.info("Skipping completed hyperparameter combination")
            continue
        if max_runs is not None and len(planned) >= int(max_runs):
            break
        config_index = next_index + len(planned) if resume else index
        config_id = f"dynamic_deephit_faithful_cfg_{config_index:03d}"
        run_dir = output_root / "tuning" / config_id / f"seed_{seed}"
        planned.append({"config_id": config_id, "seed": seed, "output_dir": str(run_dir), "params": params})
        if dry_run:
            continue
        if run_dir.exists() and not force:
            logger.info("Run directory exists; files will be updated in place: %s", run_dir)
        run_config = build_run_config(base, config_id, params, run_dir, seed, sample_size, device, include_test=False)
        try:
            metrics = train_dynamic_deephit_faithful(run_config, logger)
            rows.append(result_row(config_id, params, seed, run_dir, sample_size, metrics=metrics))
        except Exception as exc:
            logger.exception("Dynamic-DeepHit faithful candidate failed: %s", config_id)
            rows.append(result_row(config_id, params, seed, run_dir, sample_size, error=repr(exc)))

    combined = existing_rows + rows if resume else rows
    if combined and not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(combined).to_csv(results_path, index=False)
        selection = select_candidates(combined)
        covered = len(completed_signatures) + len(rows) if resume else len(rows)
        if covered < len(candidates):
            selection.update({"status": "partial_grid_not_final", "completed_candidates": covered, "total_candidates": len(candidates)})
        if sample_size is not None:
            selection.update({"status": "smoke_only_not_final", "sample_size": int(sample_size)})
        save_json(output_root / "best_hyperparameters.json", selection)
    return planned


def main():
    parser = argparse.ArgumentParser(description="Validation-only tuning for Dynamic-DeepHit faithful 72h.")
    parser.add_argument("--config", default="configs/dynamic_deephit_faithful_72h.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    planned = tune(args.config, args.dry_run, args.max_runs, args.sample_size, args.device, args.force, args.resume)
    if args.dry_run:
        print(json.dumps(planned, indent=2))


if __name__ == "__main__":
    main()
