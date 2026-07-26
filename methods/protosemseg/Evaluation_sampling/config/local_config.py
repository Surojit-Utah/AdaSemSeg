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
    'train_data_dir': _join('F3', 'train'),
    'train_data_vol_name': 'train_seismic.npy',
    'train_label_vol_name': 'train_labels.npy',
    'train_val_test_split': _join('F3', 'split_train_val_test_f3.json'),
    'train_indices': 'all',
    'test_data_dir': _join('F3', 'test'),
    'test_data_vol_name': 'test1_seismic.npy',
    'test_label_vol_name': 'test1_labels.npy',
    'axis': 0,
}
f3_facies_xline_data_info = {
    'train_data_dir': _join('F3', 'train'),
    'train_data_vol_name': 'train_seismic.npy',
    'train_label_vol_name': 'train_labels.npy',
    'train_val_test_split': _join('F3', 'split_train_val_test_f3.json'),
    'train_indices': 'all',
    'test_data_dir': _join('F3', 'test'),
    'test_data_vol_name': 'test2_seismic.npy',
    'test_label_vol_name': 'test2_labels.npy',
    'axis': 1,
}
parihaka_facies_inline_data_info = {
    'train_data_dir': _join('Parihaka'),
    'train_data_vol_name': 'parihaka_facies_train_images.npy',
    'train_label_vol_name': 'parihaka_facies_train_labels.npy',
    'axis': 0,
    'train_val_test_split': _join('Parihaka', 'split_train_val_test_parihaka.json'),
    'train_indices': '5',
    'test_data_dir': _join('Parihaka'),
    'test_data_vol_name': 'parihaka_facies_train_images.npy',
    'test_label_vol_name': 'parihaka_facies_train_labels.npy',
}
parihaka_facies_xline_data_info = {
    'train_data_dir': _join('Parihaka'),
    'train_data_vol_name': 'parihaka_facies_train_images.npy',
    'train_label_vol_name': 'parihaka_facies_train_labels.npy',
    'axis': 1,
    'train_val_test_split': _join('Parihaka', 'split_train_val_test_parihaka.json'),
    'train_indices': '5',
    'test_data_dir': _join('Parihaka'),
    'test_data_vol_name': 'parihaka_facies_train_images.npy',
    'test_label_vol_name': 'parihaka_facies_train_labels.npy',
}
penobscot_facies_inline_data_info = {
    'train_data_dir': _join('Penobscot'),
    'train_data_vol_name': 'seismic.npy',
    'train_label_vol_name': 'seismic_labels.npy',
    'axis': 0,
    'train_val_test_split': _join('Penobscot', 'split_train_val_test_penobscot.json'),
    'train_indices': '5',
    'test_data_dir': _join('Penobscot'),
    'test_data_vol_name': 'seismic.npy',
    'test_label_vol_name': 'seismic_labels.npy',
}
penobscot_facies_xline_data_info = {
    'train_data_dir': _join('Penobscot'),
    'train_data_vol_name': 'seismic.npy',
    'train_label_vol_name': 'seismic_labels.npy',
    'axis': 1,
    'train_val_test_split': _join('Penobscot', 'split_train_val_test_penobscot.json'),
    'train_indices': '5',
    'test_data_dir': _join('Penobscot'),
    'test_data_vol_name': 'seismic.npy',
    'test_label_vol_name': 'seismic_labels.npy',
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
        'visualization_path': os.path.join(log_dir, 'visualization'),
        'results_path': os.path.join(log_dir, 'results'),
        'classes': ['parihaka_facies_data_inline', 'parihaka_facies_data_crossline',
                    'f3_facies_data_inline', 'f3_facies_data_crossline',
                    'penobscot_facies_data_inline', 'penobscot_facies_data_crossline'],
        'data_info': data_info_catalogue,
    }
    return config


if __name__ == "__main__":
    config = create_config()
    for class_name in config['classes']:
        print("Details about the class : " + str(class_name))
        print(config['data_info'][class_name])
        print("\n")