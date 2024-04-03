import os

f3_facies_inline_data_info = {
    'data_dir': 'F3/data/train',
    'data_vol_name': 'train_seismic.npy',
    'label_vol_name': 'train_labels.npy',
    'patch_overlap': 0.2,
    'axis': 0,
    'train_val_test_split': 'F3/data/train/split_train_val_test_f3.json',
    'train_indices': 'all',
}
f3_facies_xline_data_info = {
    'data_dir': 'F3/data/train',
    'data_vol_name': 'train_seismic.npy',
    'label_vol_name': 'train_labels.npy',
    'patch_overlap': 0.1,
    'axis': 1,
    'train_val_test_split': 'F3/data/train/split_train_val_test_f3.json',
    'train_indices': 'all',
}
parihaka_facies_inline_data_info = {
    'data_dir': 'Parihaka/Parihaka_Facies_data',
    'data_vol_name': 'parihaka_facies_train_images.npy',
    'label_vol_name': 'parihaka_facies_train_labels.npy',
    'patch_overlap': 0.8,
    'axis': 0,
    'train_val_test_split': 'Parihaka/Parihaka_Facies_data/split_train_val_test_parihaka.json',
    'train_indices': 'all',
}
parihaka_facies_xline_data_info = {
    'data_dir': 'Parihaka/Parihaka_Facies_data',
    'data_vol_name': 'parihaka_facies_train_images.npy',
    'label_vol_name': 'parihaka_facies_train_labels.npy',
    'patch_overlap': 1.0,
    'axis': 1,
    'train_val_test_split': 'Parihaka/Parihaka_Facies_data/split_train_val_test_parihaka.json',
    'train_indices': 'all',
}
penobscot_facies_inline_data_info = {
    'data_dir': 'Penobscot/Processed',
    'data_vol_name': 'seismic.npy',
    'label_vol_name': 'seismic_labels.npy',
    'patch_overlap': 0.5,
    'axis': 0,
    'train_val_test_split': 'Penobscot/Processed/split_train_val_test_penobscot.json',
    'train_indices': 'all',
}
penobscot_facies_xline_data_info = {
    'data_dir': 'Penobscot/Processed',
    'data_vol_name': 'seismic.npy',
    'label_vol_name': 'seismic_labels.npy',
    'patch_overlap': 0.5,
    'axis': 1,
    'train_val_test_split': 'Penobscot/Processed/split_train_val_test_penobscot.json',
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

def create_config(log_dir = os.path.join(os.getcwd(), 'logs')):
    config = {
        'exp_spec': os.path.join(log_dir, 'exp_spec'),
        'tb_dir': os.path.join(log_dir, 'tb_log'),
        'visualization_path': os.path.join(log_dir, 'visualization'),
        'checkpoint_path': os.path.join(log_dir, 'checkpoints'),
        'classes': ['parihaka_facies_data_inline', 'parihaka_facies_data_crossline'],
                    #'f3_facies_data_inline', 'f3_facies_data_crossline'
                    #'penobscot_facies_data_inline', 'penobscot_facies_data_crossline'
                    #'parihaka_facies_data_inline', 'parihaka_facies_data_crossline'
        'data_info': data_info_catalogue,
    }
    return config


if __name__=="__main__":
    config = create_config()
    for class_name in config['classes']:
        print("Details about the class : " + str(class_name))
        print(config['data_info'][class_name])
        print("\n")