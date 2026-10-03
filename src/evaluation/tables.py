import json
from pathlib import Path

import pandas as pd

SETTINGS = ["centralized", "iid", "mild_non_iid", "strong_non_iid", "strong_e3"]
LABELS = {
    "centralized": "Centralized",
    "iid": "FedAvg IID",
    "mild_non_iid": "FedAvg mild",
    "strong_non_iid": "FedAvg strong",
    "strong_e3": "Strong E3R10",
}


def read_results(output_dir):
    entries = []
    for setting in SETTINGS:
        for seed in (42, 43, 44):
            path = Path(output_dir) / "runs" / setting / f"seed_{seed}"
            if not (path / "final_metrics.json").exists():
                continue
            manifest = json.loads((path / "manifest.json").read_text())
            if manifest["status"] != "complete":
                continue
            result = json.loads((path / "final_metrics.json").read_text())
            entry = {
                "setting": setting,
                "seed": seed,
                "path": str(path),
                **result["final_test"],
                "best_validation_test_accuracy": result["best_validation_test"]["accuracy"],
                "best_validation_round": result["best_validation_round"],
                "r80_round": result["r80_validation"]["round"] if result["r80_validation"] else None,
                "r80_effective_epochs": result["r80_validation"]["effective_epochs"]
                if result["r80_validation"]
                else None,
                "examples_seen": result["examples_seen"],
                "optimizer_steps": result["optimizer_steps"],
                "training_seconds": result["training_seconds"],
                "raw": result,
                "manifest": manifest,
            }
            entries.append(entry)
    return entries


def make_tables(entries, output_dir):
    directory = Path(output_dir) / "tables"
    directory.mkdir(parents=True, exist_ok=True)
    scalar_keys = [
        "setting",
        "seed",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "loss",
        "best_validation_test_accuracy",
        "best_validation_round",
        "r80_round",
        "r80_effective_epochs",
        "examples_seen",
        "optimizer_steps",
        "training_seconds",
    ]
    frame = pd.DataFrame([{key: entry[key] for key in scalar_keys} for entry in entries])
    frame.to_csv(directory / "per_seed.csv", index=False)
    summary = []
    for setting in SETTINGS:
        group = frame[frame.setting == setting]
        if group.empty:
            continue
        row = {"setting": setting, "n_seeds": len(group), "r80_reached": int(group.r80_round.notna().sum())}
        for key in scalar_keys[2:]:
            row[key + "_mean"] = group[key].mean()
            row[key + "_sd"] = group[key].std(ddof=1)
        summary.append(row)
    pd.DataFrame(summary).to_csv(directory / "summary.csv", index=False)
    per_class, local = [], []
    for entry in entries:
        for metrics in entry["per_class"]:
            per_class.append({"setting": entry["setting"], "seed": entry["seed"], **metrics})
        for metrics in entry["raw"].get("local_validation", []):
            local.append(
                {
                    "setting": entry["setting"],
                    "seed": entry["seed"],
                    "client": metrics["client"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "n_samples": metrics["n_samples"],
                    "class_counts": json.dumps(metrics["class_counts"]),
                }
            )
    class_frame = pd.DataFrame(per_class)
    class_frame.to_csv(directory / "per_class_per_seed.csv", index=False)
    class_frame.groupby(["setting", "class"])[["precision", "recall", "f1"]].agg(["mean", "std"]).to_csv(
        directory / "per_class_summary.csv"
    )
    pd.DataFrame(local).to_csv(directory / "local_validation.csv", index=False)
    return summary
