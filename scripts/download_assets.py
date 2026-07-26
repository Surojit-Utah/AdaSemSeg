#!/usr/bin/env python
"""
Download large datasets and trained weights for the AdaSemSeg reproducibility repo.

Assets are hosted externally (e.g., IEEE DataPort). Fill in the URLs below after
uploading the archives, then run:

    python scripts/download_assets.py --data
    python scripts/download_assets.py --weights
    python scripts/download_assets.py --all

Default download directory: ./data (datasets) and ./checkpoints (weights).
Set ADASEMSEG_DATA_ROOT and ADASEMSEG_CKPT_ROOT env vars to override.
"""

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path

try:
    import requests
except ImportError:
    print("requests is required. Install: pip install requests")
    sys.exit(1)


# ---------------------------------------------------------------------------
# TODO: Replace placeholder URLs with the actual IEEE DataPort download links.
# ---------------------------------------------------------------------------
ASSETS = {
    "datasets": {
        "url": "https://ieee-dataport.org/documents/adasemseg-datasets-placeholder",  # noqa
        "filename": "adasemseg_datasets.zip",
        "md5": None,
        "extract_to": "data",
    },
    "simclr_weights": {
        "url": "https://ieee-dataport.org/documents/adasemseg-simclr-weights-placeholder",  # noqa
        "filename": "adasemseg_simclr_weights.zip",
        "md5": None,
        "extract_to": "checkpoints",
    },
    "adasemseg_weights": {
        "url": "https://ieee-dataport.org/documents/adasemseg-weights-placeholder",  # noqa
        "filename": "adasemseg_model_weights.zip",
        "md5": None,
        "extract_to": "checkpoints",
    },
}


def compute_md5(path, chunk_size=8192):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url, dest):
    print(f"Downloading {url} ...")
    print(f"Saving to {dest}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 / total
                        print(f"\r{pct:.1f}%", end="", flush=True)
    print("\nDownload complete.")


def verify_md5(path, expected):
    if expected is None:
        return True
    actual = compute_md5(path)
    if actual != expected:
        print(f"MD5 mismatch: expected {expected}, got {actual}")
        return False
    print("MD5 verified.")
    return True


def extract_archive(archive_path, extract_to):
    print(f"Extracting {archive_path} to {extract_to} ...")
    os.makedirs(extract_to, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(extract_to)
    else:
        print(f"Unsupported archive format: {archive_path}")
        return False
    print("Extraction complete.")
    return True


def download_asset(name, root_dir):
    asset = ASSETS[name]
    url = asset["url"]
    if "placeholder" in url:
        print(f"ERROR: Placeholder URL for '{name}'. Please update scripts/download_assets.py with the real IEEE DataPort link.")
        return False

    dest_dir = os.path.join(root_dir, asset["extract_to"])
    archive_path = os.path.join(root_dir, asset["filename"])

    if os.path.exists(archive_path):
        print(f"Archive already exists: {archive_path}")
    else:
        download_file(url, archive_path)

    if not verify_md5(archive_path, asset["md5"]):
        return False

    return extract_archive(archive_path, dest_dir)


def main():
    parser = argparse.ArgumentParser(description="Download AdaSemSeg datasets and weights")
    parser.add_argument("--data", action="store_true", help="Download datasets")
    parser.add_argument("--weights", action="store_true", help="Download trained weights")
    parser.add_argument("--all", action="store_true", help="Download everything")
    parser.add_argument("--root", default=".", help="Root directory for downloads")
    args = parser.parse_args()

    if not any([args.data, args.weights, args.all]):
        parser.print_help()
        sys.exit(0)

    success = True
    if args.data or args.all:
        success = download_asset("datasets", args.root) and success
    if args.weights or args.all:
        success = download_asset("simclr_weights", args.root) and success
        success = download_asset("adasemseg_weights", args.root) and success

    if success:
        print("All requested assets are ready.")
    else:
        print("Some assets could not be downloaded/extracted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
