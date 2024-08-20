from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def relabel_roi_img(
    in_img_file: Path,
    roi_map_file: Path,
    label_from: str,
    label_to: str,
    out_img_file: Path,
) -> None:
    """
    Convert labels in input roi image to new labels based on the mapping
    The mapping file should contain numeric indices for the mapping
    between the input roi image (from) and output roi image (to)

    :param in_img_file: the passed roi image
    :type in_img_file: str
    :param roi_map_file: the passed mapping file(.csv)
    :type roi_map_file: str
    :param label_from: the mapping from the input roi image
    :type label_from: str
    :param label_to: the mapping to the output roi image
    :type label_to: str
    :param out_img_file: the wanted output filename
    :type out_img_file: str
    """

    # Read image
    in_nii = nib.load(in_img_file)
    img_mat = in_nii.get_fdata().astype(int)

    # Read dictionary with roi index mapping
    df_dict = pd.read_csv(roi_map_file)

    # Convert mapping dataframe to dictionaries
    v_from = df_dict[label_from].astype(int)
    v_to = df_dict[label_to].astype(int)

    # Create a mapping with consecutive numbers from dest to target values
    tmp_map = np.zeros(np.max([v_from, v_to]) + 1).astype(int)
    tmp_map[v_from] = v_to

    # Replace each value v in data by the value of tmp_map with the index v
    out_mat = tmp_map[img_mat].astype(np.uint8)

    # Write updated img
    out_nii = nib.Nifti1Image(out_mat, in_nii.affine, in_nii.header)
    nib.save(out_nii, out_img_file)
