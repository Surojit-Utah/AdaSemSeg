import numpy as np
import os
import h5py


"""
    Processes the raw Penobscot seismic volume into the seismic.npy /
    seismic_labels.npy volumes used by this repository's training and
    evaluation code:
    1. input data (seismic.npy)
    2. labels (seismic_labels.npy)

    Source data: dataset.h5 is the raw Penobscot interpretation dataset from
        Baroni, L., Silva, R.M., Ferreira, R.S., Civitarese, D., Szwarcman, D.,
        Brazil, E.V. "Penobscot dataset: Fostering machine learning development
        for seismic interpretation." arXiv:1903.12060 (2021).
    dataset.h5 itself carries no embedded source/license metadata (checked via
    h5py -- only raw features/label arrays and per-slice index bookkeeping), so
    this citation is the only available provenance; obtain the original file
    from the authors' published release.

    Processing applied (matches Section III-A of the AdaSemSeg paper):
    1. Reorients the volume to inline x crossline x depth.
    2. Crops to a fixed inline/crossline/depth window to remove the corrupted
       edges of the volume and noise at the bottom (601 -> 460 inline slices,
       481 -> 471 crossline slices).
    3. Applies 5-95 percentile clipping and rescales to uint8 [0, 255].
"""
if __name__=='__main__':

    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, "dataset.h5")
    inline_start_index = 70
    inline_end_index = 530
    crossline_start_index = 10
    crossline_end_index = None
    depth_start_index = 30
    depth_end_index = 900


    with h5py.File(filename, "r") as f:

        seismic_data = f['features'][()]  # returns as a h5py dataset object
        seismic_labels = f['label'][()]  # returns as a h5py dataset object

        # volume aligned as inline X crossline X depth
        seismic_data = np.squeeze(seismic_data.transpose(0, 2, 1, 3), axis=-1)
        seismic_labels = seismic_labels.transpose(0, 2, 1)

        ##################################################
        # Percentile-based filtering of the seismic volume
        ##################################################
        # Upper threshold
        higher_percentile_level = 95.0
        higher_percentile_val = np.percentile(seismic_data, higher_percentile_level)

        # Lower threshold
        lower_percentile_level = 100 - higher_percentile_level
        lower_percentile_val = np.percentile(seismic_data, lower_percentile_level)

        # Filtering the numpy array using the higher and lower filter value
        seismic_data = np.where(seismic_data > higher_percentile_val, higher_percentile_val, seismic_data)
        seismic_data = np.where(seismic_data < lower_percentile_val, lower_percentile_val, seismic_data)
        seismic_data = (np.divide((seismic_data - np.min(seismic_data)), (np.max(seismic_data) - np.min(seismic_data)))*255).astype(np.uint8)

        ###############################
        # Selecting the data to be used
        ###############################

        # Set the min and max index to the original value
        if inline_start_index is None:
            inline_start_index = 0
        if inline_end_index is None:
            inline_end_index = seismic_data.shape[0]
        if crossline_start_index is None:
            crossline_start_index = 0
        if crossline_end_index is None:
            crossline_end_index = seismic_data.shape[1]
        if depth_start_index is None:
            depth_start_index = 0
        if depth_end_index is None:
            depth_end_index = seismic_data.shape[2]

        seismic_data = seismic_data[inline_start_index:inline_end_index, crossline_start_index:crossline_end_index, depth_start_index:depth_end_index]
        seismic_labels = seismic_labels[inline_start_index:inline_end_index, crossline_start_index:crossline_end_index, depth_start_index:depth_end_index]

        #####################
        # save the npy files
        #####################
        np.save(os.path.join(script_dir, 'seismic.npy'), seismic_data)
        np.save(os.path.join(script_dir, 'seismic_labels.npy'), seismic_labels)
        print(f"Saved seismic.npy and seismic_labels.npy to {script_dir}")