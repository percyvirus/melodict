"""Automated dataset downloader and verification script for Melodict."""

import argparse
import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

# Restructured to support 'mini' (MIDI/annotations only) and 'full' (Audio + Metadata) modes
DATASETS = {
    "maestro": {
        "desc": "MAESTRO v3.0.0",
        "mini": [
            "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
        ],
        "full": [
            "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip",
            "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.csv"
        ]
    },
    "guitarset": {
        "desc": "GuitarSet JAMS/MIDI and Audio",
        "mini": [
            "https://zenodo.org/record/3371780/files/annotation.zip"
        ],
        "full": [
            "https://zenodo.org/records/3371780/files/annotation.zip",
            "https://zenodo.org/records/3371780/files/audio_mono-mic.zip",
            "https://zenodo.org/records/3371780/files/audio_mono-pickup_mix.zip",
            "https://zenodo.org/records/3371780/files/audio_hex-pickup_debleeded.zip",
            "https://zenodo.org/records/3371780/files/audio_hex-pickup_original.zip"
        ]
    }
}


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)` for urllib."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def download_and_extract(name: str, info: dict, mode: str, output_dir: Path, cleanup: bool) -> None:
    """Download dataset files, skip if they exist, and extract zip archives with optimized progress bars."""
    print(f"\n[{name}] Starting '{mode}' download: {info['desc']}...")
    
    target_dir = output_dir / name
    target_dir.mkdir(parents=True, exist_ok=True)
    
    urls = info[mode]
    
    for url in urls:
        filename = url.split("/")[-1]
        target_path = target_dir / filename
        
        # 1. Download with progress bar or skip if it exists
        if target_path.exists():
            print(f"[{name}] File {filename} already exists at {target_path}. Skipping download.")
        else:
            with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=f"Downloading {filename}") as t:
                urllib.request.urlretrieve(url, filename=target_path, reporthook=t.update_to)
        
        # 2. Extract contents with progress bar if it is a zip archive
        if filename.endswith(".zip"):
            print(f"[{name}] Analyzing {filename} size...")
            # Quickly get the total file count for the progress bar without extracting
            with zipfile.ZipFile(target_path, "r") as zip_ref:
                total_files = len(zip_ref.infolist())
            
            # Check if the native OS 'unzip' tool is available (macOS/Linux) for C-level speed
            if shutil.which("unzip"):
                print(f"[{name}] Using system 'unzip' engine for accelerated extraction...")
                with tqdm(total=total_files, desc=f"Extracting {filename}", unit="file") as pbar:
                    # Run system unzip and pipe output to Python
                    process = subprocess.Popen(
                        ["unzip", "-o", str(target_path), "-d", str(target_dir)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True
                    )
                    # Read the output line by line to advance the tqdm bar
                    for line in process.stdout:
                        # unzip prints these keywords when processing files
                        if any(keyword in line.lower() for keyword in ["inflating:", "extracting:", "creating:"]):
                            pbar.update(1)
                    process.wait()
            else:
                # Fallback for Windows if no native unzip is available
                print(f"[{name}] Native 'unzip' not found. Falling back to Python zipfile (slower)...")
                with zipfile.ZipFile(target_path, "r") as zip_ref:
                    members = zip_ref.infolist()
                    for member in tqdm(members, desc=f"Extracting {filename}", unit="file"):
                        zip_ref.extract(member, target_dir)
            
            # 3. Clean up the zip file only if the user explicitly requested it
            if cleanup:
                os.remove(target_path)
                print(f"[{name}] Extracted and deleted {filename} to save space.")
            else:
                print(f"[{name}] Extracted successfully. Zip file kept at {target_path}")
        else:
            print(f"[{name}] File ready at {target_path}")

    print(f"[{name}] Successfully processed at {target_dir}\n")


def main() -> None:
    """Main execution point for dataset acquisition."""
    parser = argparse.ArgumentParser(
        description="Download music datasets for Melodict."
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["maestro", "guitarset", "all"],
        default="all",
        help="Target dataset to download.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["mini", "full"],
        default="mini",
        help="Download mode: 'mini' for annotations/MIDI only, 'full' for audio and metadata.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(Path(__file__).parent.parent / "datasets"),
        help="Custom target directory for datasets (e.g., an external drive path).",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete zip files after extraction to save space (default: False).",
    )
    args = parser.parse_args()

    # Create the base directory using the provided path
    base_dir = Path(args.data_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    targets = DATASETS.keys() if args.target == "all" else [args.target]

    for dataset_name in targets:
        download_and_extract(dataset_name, DATASETS[dataset_name], args.mode, base_dir, args.cleanup)


if __name__ == "__main__":
    main()