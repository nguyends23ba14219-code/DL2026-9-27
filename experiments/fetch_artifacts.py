"""Fetch authorized private-release artifacts with gh, verify checksums, extract safely."""

import argparse
import hashlib
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full", action="store_true", help="Download all-round checkpoints instead of the smaller demo bundle"
    )
    args = parser.parse_args()
    manifest_path = ROOT / "artifacts" / "release_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Release manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    name = "checkpoints.zip" if args.full else "demo-artifacts.zip"
    with tempfile.TemporaryDirectory() as directory:
        subprocess.run(
            [
                "gh",
                "release",
                "download",
                manifest["tag"],
                "--repo",
                manifest["repository"],
                "--pattern",
                name,
                "--dir",
                directory,
            ],
            check=True,
        )
        archive = Path(directory) / name
        digest = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
        if digest != manifest["assets"][name]["sha256"]:
            raise RuntimeError("Artifact SHA-256 does not match release manifest")
        with zipfile.ZipFile(archive) as bundle:
            for item in bundle.infolist():
                destination = (ROOT / item.filename).resolve()
                if not destination.is_relative_to(ROOT) or item.filename.startswith("/"):
                    raise RuntimeError("Unsafe archive path")
            bundle.extractall(ROOT)
        print(f"Verified and extracted {name} into {ROOT}")


if __name__ == "__main__":
    main()
