import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_scripts_dir = os.path.join(_repo_root, "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from config_loader import get_data_root

_data_root = str(get_data_root())


def _join(*parts):
    return os.path.normpath(os.path.join(_data_root, *parts))


f3_facies_inline_data_info = {
    'data_dir': _join('F3', 'train'),
    'data_vol_name': 'train_seismic.npy',
    'label_vol_name': 'train_labels.npy',
    'patch_overlap': 0.2,
    'axis': 0,
    'train_val_test_split': _join('F3', 'split_train_val_test_f3.json'),
    'train_indices': 'all',
}
f3_facies_xline_data_info = {
    'data_dir': _join('F3', 'train'),
    'data_vol_name': 'train_seismic.npy',
    'label_vol_name': 'train_labels.npy',
    'patch_overlap': 0.1,
    'axis': 1,
    'train_val_test_split': _join('F3', 'split_train_val_test_f3.json'),
    'train_indices': 'all',
}
parihaka_facies_inline_data_info = {
    'data_dir': _join('Parihaka'),
    'data_vol_name': 'parihaka_facies_train_images.npy',
    'label_vol_name': 'parihaka_facies_train_labels.npy',
    'patch_overlap': 0.5,
    'axis': 0,
    'train_val_test_split': _join('Parihaka', 'split_train_val_test_parihaka.json'),
    'train_indices': '5',
}
parihaka_facies_xline_data_info = {
    'data_dir': _join('Parihaka'),
    'data_vol_name': 'parihaka_facies_train_images.npy',
    'label_vol_name': 'parihaka_facies_train_labels.npy',
    'patch_overlap': 0.5,
    'axis': 1,
    'train_val_test_split': _join('Parihaka', 'split_train_val_test_parihaka.json'),
    'train_indices': '5',
}
penobscot_facies_inline_data_info = {
    'data_dir': _join('Penobscot'),
    'data_vol_name': 'seismic.npy',
    'label_vol_name': 'seismic_labels.npy',
    'patch_overlap': 0.5,
    'axis': 0,
    'train_val_test_split': _join('Penobscot', 'split_train_val_test_penobscot.json'),
    'train_indices': 'all',
}
penobscot_facies_xline_data_info = {
    'data_dir': _join('Penobscot'),
    'data_vol_name': 'seismic.npy',
    'label_vol_name': 'seismic_labels.npy',
    'patch_overlap': 0.5,
    'axis': 1,
    'train_val_test_split': _join('Penobscot', 'split_train_val_test_penobscot.json'),
    'train_indices': 'all',
}
data_info_catalogue = {
    'f3_facies_data_inline': f3_facies_inline_data_info,
    'f3_facies_data_crossline': f3_facies_xline_data_info,
    'parihaka_facies_data_inline': parihaka_facies_inline_data_info,
    'parihaka_facies_data_crossline': parihaka_facies_xline_data_info,
    'penobscot_facies_data_inline': penobscot_facies_inline_data_info,
    'penobscot_facies_data_crossline': penobscot_facies_xline_data_info,
}


def create_config(log_dir=os.path.join(os.getcwd(), 'logs')):
    config = {
        'exp_spec': os.path.join(log_dir, 'exp_spec'),
        'tb_dir': os.path.join(log_dir, 'tb_log'),
        'visualization_path': os.path.join(log_dir, 'visualization'),
        'checkpoint_path': os.path.join(log_dir, 'checkpoints'),
        'classes': ['f3_facies_data_inline', 'f3_facies_data_crossline'],
        'data_info': data_info_catalogue,
    }
    return config


if __name__ == "__main__":
    config = create_config()
    for class_name in config['classes']:
        print("Details about the class : " + str(class_name))
        print(config['data_info'][class_name])
        print("\n")