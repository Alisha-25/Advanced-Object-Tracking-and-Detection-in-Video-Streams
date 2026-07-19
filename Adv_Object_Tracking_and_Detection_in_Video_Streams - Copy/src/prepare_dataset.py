"""Safely extract the supplied MOT17 ZIP to the configured dataset directory."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def safe_extract(zip_path: Path, destination: Path) -> None:
    """Extract only safe archive members, rejecting path-traversal attempts."""
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        for member in members:
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe ZIP member rejected: {member.filename}")
        archive.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, help="Path to archive (2).zip")
    parser.add_argument("--destination", required=True, help="Folder that will contain MOT17/")
    args = parser.parse_args()
    zip_path, destination = Path(args.zip), Path(args.destination)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    destination.mkdir(parents=True, exist_ok=True)
    expected = destination / "MOT17"
    if expected.exists():
        raise FileExistsError(f"{expected} already exists; extraction was not repeated.")
    print(f"Extracting {zip_path.name} to {destination} ...")
    safe_extract(zip_path, destination)
    print(f"Done. Configure dataset_root as: {expected}")


if __name__ == "__main__":
    main()
