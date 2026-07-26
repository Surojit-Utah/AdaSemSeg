"""
Unified configuration loader.

Reads configs/datasets.yaml and resolves paths relative to ADASEMSEG_DATA_ROOT.
Returns data_info_catalogue dicts compatible with each method's local_config.py.
"""

import os
import re
from pathlib import Path

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required. Install: pip install pyyaml")


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data"


def _resolve_env(value):
    """Resolve ${VAR:-default} placeholders in YAML strings."""
    if not isinstance(value, str):
        return value
    pattern = re.compile(r"\$\{([^}]+)\}")

    def replacer(match):
        expr = match.group(1)
        if ":-" in expr:
            var, default = expr.split(":-", 1)
        else:
            var, default = expr, ""
        return os.environ.get(var, default)

    return pattern.sub(replacer, value)


def get_data_root():
    return Path(os.environ.get("ADASEMSEG_DATA_ROOT", DEFAULT_DATA_ROOT)).resolve()


def load_datasets_config():
    config_path = REPO_ROOT / "configs" / "datasets.yaml"
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    data_root = get_data_root()
    catalogue = {}
    for dataset_key, dataset in raw.items():
        if dataset_key == "data_root":
            continue
        if not isinstance(dataset, dict):
            continue
        for axis_key, axis_info in dataset.items():
            if axis_key in ("name",):
                continue
            if not isinstance(axis_info, dict):
                continue
            class_name = f"{dataset_key}_facies_data_{'inline' if axis_key == 'inline' else 'crossline'}"
            catalogue[class_name] = {
                "data_dir": str(data_root / _resolve_env(axis_info["data_dir"])),
                "data_vol_name": axis_info["data_vol_name"],
                "label_vol_name": axis_info["label_vol_name"],
                "patch_overlap": axis_info["patch_overlap"],
                "axis": axis_info["axis"],
                "train_val_test_split": str(data_root / _resolve_env(axis_info["train_val_test_split"])),
                "train_indices": axis_info["train_indices"],
            }
    return catalogue


def make_data_info_catalogue():
    """Alias for backward compatibility."""
    return load_datasets_config()


def load_eval_datasets_config():
    """Return evaluation-oriented data_info_catalogue with separate train/test paths.

    Compatible with the evaluation datasets in methods/*/Evaluation/.
    """
    config_path = REPO_ROOT / "configs" / "datasets.yaml"
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    data_root = get_data_root()
    catalogue = {}
    test_vol_map = {
        "f3": {"inline": ("F3/test", "test1_seismic.npy", "test1_labels.npy"),
                "crossline": ("F3/test", "test2_seismic.npy", "test2_labels.npy")},
        "parihaka": {"inline": ("Parihaka", "parihaka_facies_train_images.npy", "parihaka_facies_train_labels.npy"),
                     "crossline": ("Parihaka", "parihaka_facies_train_images.npy", "parihaka_facies_train_labels.npy")},
        "penobscot": {"inline": ("Penobscot", "seismic.npy", "seismic_labels.npy"),
                      "crossline": ("Penobscot", "seismic.npy", "seismic_labels.npy")},
    }
    train_vol_map = {
        "f3": ("F3/train", "train_seismic.npy", "train_labels.npy"),
        "parihaka": ("Parihaka", "parihaka_facies_train_images.npy", "parihaka_facies_train_labels.npy"),
        "penobscot": ("Penobscot", "seismic.npy", "seismic_labels.npy"),
    }

    for dataset_key, dataset in raw.items():
        if dataset_key == "data_root":
            continue
        if not isinstance(dataset, dict):
            continue
        train_dir, train_vol, train_label = train_vol_map[dataset_key]
        for axis_key, axis_info in dataset.items():
            if axis_key in ("name",):
                continue
            if not isinstance(axis_info, dict):
                continue
            test_dir, test_vol, test_label = test_vol_map[dataset_key][axis_key]
            class_name = f"{dataset_key}_facies_data_{'inline' if axis_key == 'inline' else 'crossline'}"
            catalogue[class_name] = {
                "train_data_dir": str(data_root / train_dir),
                "train_data_vol_name": train_vol,
                "train_label_vol_name": train_label,
                "train_val_test_split": str(data_root / _resolve_env(axis_info["train_val_test_split"])),
                "train_indices": axis_info["train_indices"],
                "test_data_dir": str(data_root / test_dir),
                "test_data_vol_name": test_vol,
                "test_label_vol_name": test_label,
                "axis": axis_info["axis"],
            }
    return catalogue


def make_eval_data_info_catalogue():
    """Alias for backward compatibility."""
    return load_eval_datasets_config()


if __name__ == "__main__":
    cfg = load_datasets_config()
    for k, v in cfg.items():
        print(k)
        print(v)
        print()
