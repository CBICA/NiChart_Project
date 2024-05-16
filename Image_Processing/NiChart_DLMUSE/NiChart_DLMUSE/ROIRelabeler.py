import nibabel as nib
import numpy as np
import pandas as pd


def relabel_roi_img(in_img_file, roi_map_file, label_from, label_to, out_img_file):
    '''Convert labels in input roi image to new labels based on the mapping
    The mapping file should contain numeric indices for the mapping
    between the input roi image (from) and output roi image (to)
    '''

    ## Read image
    in_nii = nib.load(in_img_file)
    img_mat = in_nii.get_fdata().astype(int)

    ## Read dictionary with roi index mapping 
    df_dict = pd.read_csv(roi_map_file)
    
    # Convert mapping dataframe to dictionaries
    v_from = df_dict[label_from].astype(int)
    v_to = df_dict[label_to].astype(int)
    
    ## Create a mapping with consecutive numbers from dest to target values
    tmp_map = np.zeros(np.max([v_from, v_to]) + 1).astype(int)
    tmp_map[v_from] = v_to
    
    ## Replace each value v in data by the value of tmp_map with the index v
    out_mat = tmp_map[img_mat].astype(np.uint8)
    
    ## Write updated img
    out_nii = nib.Nifti1Image(out_mat, in_nii.affine, in_nii.header)
    nib.save(out_nii, out_img_file)
