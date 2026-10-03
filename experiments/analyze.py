import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.tables import make_tables, read_results
from src.evaluation.plots import make_plots
from src.evaluation.report import make_report
from src.utils.logging import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    entries = read_results(args.output_dir)
    if len(entries) != 15:
        raise RuntimeError(
            f"Expected all 15 completed runs, found {len(entries)}. No complete-study figures generated."
        )
    summary = make_tables(entries, args.output_dir)
    make_plots(entries, args.output_dir)
    make_report(entries, summary, args.output_dir)
    atomic_json(
        Path(args.output_dir) / "suite_manifest.json",
        {
            "expected_runs": 15,
            "completed_runs": 15,
            "runs": [
                {"setting": e["setting"], "seed": e["seed"], "path": e["path"], "status": "complete"} for e in entries
            ],
        },
    )
    print("Rebuilt tables, figures and report from 15 completed runs.")


if __name__ == "__main__":
    main()
