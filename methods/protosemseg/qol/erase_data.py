import argparse
import os
import shutil
import sys
sys.path.append('..')
from config.local_config import create_config


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Experiment runfile, you run experiments from this file")
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--shots", type=int, required=True)
    args = parser.parse_args()
    run_id = args.run_id
    shot = args.shots

    config = create_config(log_dir = os.path.join('..', 'logs'))

    tb_dir = os.path.join(config['tb_dir'], str(shot) + '-shot', 'Run_' + str(run_id))
    shutil.rmtree(tb_dir)
    print("Removed the tensorboard data....")

    visualization_path = os.path.join(config['visualization_path'], str(shot) + '-shot', 'Run_' + str(run_id))
    shutil.rmtree(visualization_path)
    print("Removed the visualization directory....")

    checkpoint_path = os.path.join(config['checkpoint_path'], str(shot) + '-shot', 'Run_' + str(run_id))
    shutil.rmtree(checkpoint_path)
    print("Removed the checkpoint data....")

    exp_spec_path = os.path.join(config['exp_spec'], str(shot) + '-shot', 'Run_' + str(run_id))
    shutil.rmtree(exp_spec_path)
    print("Removed the exp specification....")