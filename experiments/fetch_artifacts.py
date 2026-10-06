"""Download public release artifacts without authentication, verify and extract."""

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def verify_and_extract(archive, metadata, root):
    with archive.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != metadata["sha256"] or archive.stat().st_size != metadata["bytes"]:
        raise RuntimeError("Artifact SHA-256 or size does not match release manifest")
    root = root.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            destination = (root / item.filename).resolve()
            if not destination.is_relative_to(root) or item.filename.startswith("/"):
                raise RuntimeError("Unsafe archive path")
        bundle.extractall(root)


def main():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--full", action="store_true", help="Download all-round checkpoints instead of the smaller demo bundle"
    )
    selection.add_argument("--data", action="store_true", help="Download the complete processed study dataset")
    parser.add_argument("--destination", type=Path, default=ROOT, help="Extraction root (default: repository root)")
    args = parser.parse_args()
    manifest_path = ROOT / "artifacts" / "release_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Release manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    name = "processed-data.zip" if args.data else ("checkpoints.zip" if args.full else "demo-artifacts.zip")
    url = f"https://github.com/{manifest['repository']}/releases/download/{manifest['tag']}/{name}"
    with tempfile.TemporaryDirectory() as directory:
        archive = Path(directory) / name
        request = Request(url, headers={"User-Agent": "DL2026-9-27-artifact-fetcher"})
        with urlopen(request, timeout=120) as response, archive.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        verify_and_extract(archive, manifest["assets"][name], args.destination)
        print(f"Verified and extracted {name} into {args.destination.resolve()}")


if __name__ == "__main__":
    main()
