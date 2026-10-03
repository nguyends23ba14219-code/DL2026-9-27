import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import load_data
from src.runner import load_config, run
from src.utils.logging import atomic_json

SETTINGS = ["centralized", "iid", "mild_non_iid", "strong_non_iid", "strong_e3"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", nargs="+", choices=SETTINGS, default=SETTINGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    data, entries = None, []
    for setting in args.settings:
        for seed in args.seeds:
            config = load_config(f"configs/{setting}.yaml", seed)
            config.update(device=args.device, output_dir=args.output_dir)
            if data is None:
                data = load_data(config)
            path = run(config, resume=True, data=data)
            entries.append({"setting": setting, "seed": seed, "path": str(path), "status": "complete"})
            atomic_json(
                Path(args.output_dir) / "suite_manifest.json",
                {"runs": entries, "expected_runs": len(args.settings) * len(args.seeds)},
            )


if __name__ == "__main__":
    main()
