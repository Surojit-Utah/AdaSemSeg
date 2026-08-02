#!/usr/bin/env python
"""
Download datasets and trained weights for the AdaSemSeg reproducibility repo.

No large binaries are committed to this repository. Both datasets and model
weights are hosted on Zenodo as two published records, queried at download
time via Zenodo's public records API (no auth token needed -- both records
are open access):

    Checkpoints : https://zenodo.org/records/21762769  (DOI 10.5281/zenodo.21762769)
    Datasets    : https://zenodo.org/records/21764042  (DOI 10.5281/zenodo.21764042)

Usage:
    python scripts/download_assets.py --data
    python scripts/download_assets.py --weights
    python scripts/download_assets.py --all

Default download directory: ./data (datasets) and ./checkpoints (weights).
Each file is extracted into its own subdirectory (e.g. checkpoints/adasemseg/,
data/F3/) -- see the README's "Model weights" and "Download datasets and
weights" sections for the resulting directory layout.
Set ADASEMSEG_DATA_ROOT and ADASEMSEG_CKPT_ROOT env vars to override.
"""

import argparse
import hashlib
import os
import sys
import zipfile

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("requests and tqdm are required. Install: pip install requests tqdm")
    sys.exit(1)

ZENODO_API_URL = "https://zenodo.org/api/records"

# Maps each asset group to its Zenodo record and where each file it contains
# should be extracted to, relative to --root.
ZENODO_RECORDS = {
    "datasets": {
        "record_id": 21764042,
        "doi": "10.5281/zenodo.21764042",
        "files": {
            "F3_data.zip": "data/F3",
            "Parihaka_data.zip": "data/Parihaka",
            "Penobscot_data.zip": "data/Penobscot",
        },
    },
    "weights": {
        "record_id": 21762769,
        "doi": "10.5281/zenodo.21762769",
        "files": {
            "adasemseg_checkpoints.zip": "checkpoints/adasemseg",
            "protosemseg_checkpoints.zip": "checkpoints/protosemseg",
            "simclr_checkpoint.zip": "checkpoints/simclr",
        },
    },
}


def compute_md5(path, chunk_size=8192):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_record_files(record_id):
    """Returns {filename: {"url": ..., "size": ..., "checksum": "md5:<hex>"}} for a published Zenodo record."""
    r = requests.get(f"{ZENODO_API_URL}/{record_id}", timeout=30)
    r.raise_for_status()
    record = r.json()
    files = {}
    for f in record.get("files", []):
        filename = f.get("key") or f.get("filename")
        links = f.get("links", {})
        files[filename] = {
            "url": links.get("self") or links.get("download"),
            "size": f.get("size") or f.get("filesize"),
            "checksum": f.get("checksum"),
        }
    return files


def download_file(url, dest, total_size=None):
    if os.path.exists(dest):
        print(f"Already downloaded: {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = total_size or int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                          unit_divisor=1024, desc=os.path.basename(dest)) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def verify_checksum(path, expected):
    if not expected:
        return True
    algo, _, expected_hex = expected.partition(":")
    if algo != "md5":
        print(f"Skipping checksum verification (unsupported algo: {algo})")
        return True
    actual = compute_md5(path)
    if actual != expected_hex:
        print(f"Checksum mismatch for {path}: expected {expected_hex}, got {actual}")
        return False
    print(f"Checksum verified: {os.path.basename(path)}")
    return True


def extract_archive(archive_path, extract_to):
    print(f"Extracting {archive_path} to {extract_to} ...")
    os.makedirs(extract_to, exist_ok=True)
    if not zipfile.is_zipfile(archive_path):
        print(f"Unsupported archive format: {archive_path}")
        return False
    with zipfile.ZipFile(archive_path, "r") as z:
        z.extractall(extract_to)
    return True


def download_asset(group, root_dir):
    config = ZENODO_RECORDS[group]
    record_id = config["record_id"]
    print(f"Fetching file list for Zenodo record {record_id} (DOI {config['doi']}) ...")
    try:
        remote_files = fetch_record_files(record_id)
    except requests.exceptions.RequestException as exc:
        print(f"ERROR: Could not reach Zenodo record {record_id}: {exc}")
        return False

    success = True
    for filename, extract_subdir in config["files"].items():
        remote = remote_files.get(filename)
        if remote is None:
            print(f"ERROR: '{filename}' not found in Zenodo record {record_id}.")
            success = False
            continue

        archive_path = os.path.join(root_dir, "_downloads", filename)
        download_file(remote["url"], archive_path, total_size=remote["size"])

        if not verify_checksum(archive_path, remote["checksum"]):
            success = False
            continue

        dest_dir = os.path.join(root_dir, extract_subdir)
        if not extract_archive(archive_path, dest_dir):
            success = False

    return success


def main():
    parser = argparse.ArgumentParser(description="Download AdaSemSeg datasets and weights from Zenodo")
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
        success = download_asset("weights", args.root) and success

    if success:
        print("All requested assets are ready.")
    else:
        print("Some assets could not be downloaded/extracted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
