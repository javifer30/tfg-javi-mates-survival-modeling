import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Consolidate dynamic_72h final model summaries.")
    parser.add_argument("--outputs-dir", default="outputs/dynamic_72h")
    args = parser.parse_args()
    final_root = Path(args.outputs_dir) / "final"
    rows = []
    for path in final_root.glob("*/final_seed_summary.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = data.get("metrics", {})
        rows.append(
            {
                "model": data["model"],
                "test_ctd_antolini_mean": metrics.get("test_ctd_antolini_mean"),
                "test_ctd_antolini_std": metrics.get("test_ctd_antolini_std"),
                "test_ibs_mean": metrics.get("test_ibs_mean"),
                "test_ibs_std": metrics.get("test_ibs_std"),
                "test_ibll_mean": metrics.get("test_ibll_mean"),
                "test_ibll_std": metrics.get("test_ibll_std"),
                "test_mean_horizon_c_index_mean": metrics.get("test_mean_horizon_c_index_mean"),
                "test_mean_horizon_c_index_std": metrics.get("test_mean_horizon_c_index_std"),
                "selected_hyperparameters": json.dumps(data.get("selected_hyperparameters", {}), sort_keys=True),
            }
        )
    out = Path(args.outputs_dir) / "final" / "dynamic_72h_model_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("model").to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
