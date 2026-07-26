import os
import sys

# Load unified dataset configuration from the repo root.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_scripts_dir = os.path.join(_repo_root, "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from config_loader import make_data_info_catalogue

data_info_catalogue = make_data_info_catalogue()


def create_config(log_dir=os.path.join(os.getcwd(), 'logs')):
    config = {
        'exp_spec': os.path.join(log_dir, 'exp_spec'),
        'tb_dir': os.path.join(log_dir, 'tb_log'),
        'visualization_path': os.path.join(log_dir, 'visualization'),
        'checkpoint_path': os.path.join(log_dir, 'checkpoints'),
        'classes': ['f3_facies_data_inline', 'f3_facies_data_crossline',
                    'penobscot_facies_data_inline', 'penobscot_facies_data_crossline',
                    'parihaka_facies_data_inline', 'parihaka_facies_data_crossline'],
        'data_info': data_info_catalogue,
    }
    return config


if __name__ == "__main__":
    config = create_config()
    for class_name in config['classes']:
        print("Details about the class : " + str(class_name))
        print(config['data_info'][class_name])
        print("\n")