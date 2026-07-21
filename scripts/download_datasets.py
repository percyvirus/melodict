"""Automated dataset downloader and verification script for Melodict."""

import argparse
import os
import urllib.request
import zipfile
from pathlib import Path

DATASETS = {
    "maestro_mini": {
        "url": "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip",
        "desc": "MAESTRO v3.0.0 (MIDI only for symbolic pipeline testing)",
    },
    "guitarset_meta": {
        "url": "https://zenodo.org/record/3371780/files/annotation.zip",
        "desc": "GuitarSet JAMS/MIDI annotations for monophonic/polyphonic tracking",
    },
}


def download_and_extract(name: str, info: dict, output_dir: Path) -> None:
    """Download a zip dataset and extract it to the target directory."""
    print(f"[{name}] Starting download: {info['desc']}...")
    target_zip = output_dir / f"{name}.zip"

    # Download file
    urllib.request.urlretrieve(info["url"], target_zip)
    print(f"[{name}] Download complete. Extracting...")

    # Extract contents
    with zipfile.ZipFile(target_zip, "r") as zip_ref:
        zip_ref.extractall(output_dir / name)

    # Clean up zip file
    os.remove(target_zip)
    print(f"[{name}] Successfully installed at {output_dir / name}\n")


def main() -> None:
    """Main execution point for dataset acquisition."""
    parser = argparse.ArgumentParser(
        description="Download music datasets for Melodict."
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["maestro_mini", "guitarset_meta", "all"],
        default="all",
        help="Target dataset to download.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent / "datasets"
    base_dir.mkdir(exist_ok=True)

    targets = DATASETS.keys() if args.target == "all" else [args.target]

    for dataset_name in targets:
        if dataset_name in DATASETS:
            download_and_extract(dataset_name, DATASETS[dataset_name], base_dir)


if __name__ == "__main__":
    main()