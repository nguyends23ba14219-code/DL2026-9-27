import argparse

from src.runner import load_config, run


def main():
    parser = argparse.ArgumentParser(description="Train one locked Fashion-MNIST experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, args.seed)
    if args.device:
        config["device"] = args.device
    if args.output_dir:
        config["output_dir"] = args.output_dir
    run(config, args.resume)


if __name__ == "__main__":
    main()
