"""
Quality-of-life helper: erase the tensorboard/visualization/checkpoint/exp_spec
data for a given (shots, run_id) training run, e.g. to redo a run from scratch.

Can be run from anywhere, e.g.:
    python methods/adasemseg/qol/erase_data.py --run_id 1 --shots 5
"""
import argparse
import os
import shutil
import sys

_adasemseg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _adasemseg_dir not in sys.path:
    sys.path.insert(0, _adasemseg_dir)
from config.local_config import create_config


def _remove_if_exists(path, label):
    if os.path.isdir(path):
        shutil.rmtree(path)
        print(f"Removed the {label}.... ({path})")
    else:
        print(f"Skipped {label}: not found ({path})")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Erase logs/checkpoints for a given run, you run experiments from this file")
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--shots", type=int, required=True)
    args = parser.parse_args()
    run_id = args.run_id
    shot = args.shots

    config = create_config(log_dir=os.path.join(_adasemseg_dir, 'logs'))

    tb_dir = os.path.join(config['tb_dir'], str(shot) + '-shot', 'Run_' + str(run_id))
    _remove_if_exists(tb_dir, "tensorboard data")

    visualization_path = os.path.join(config['visualization_path'], str(shot) + '-shot', 'Run_' + str(run_id))
    _remove_if_exists(visualization_path, "visualization directory")

    checkpoint_path = os.path.join(config['checkpoint_path'], str(shot) + '-shot', 'Run_' + str(run_id))
    _remove_if_exists(checkpoint_path, "checkpoint data")

    exp_spec_path = os.path.join(config['exp_spec'], str(shot) + '-shot', 'Run_' + str(run_id))
    _remove_if_exists(exp_spec_path, "exp specification")
