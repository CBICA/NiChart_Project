import csv as csv

import nibabel as nib
import numpy as np
import pandas as pd


def calc_roi_volumes(mrid, in_img_file, label_indices = []):
    '''Creates a dataframe with the volumes of rois.
    Users should provide the mrid (to be added in MRID column) and 
    the input roi image.
    Users can optionally select a set of roi indices (default: all indices in the img)
    '''
    ## Keep input lists as arrays
    label_indices = np.array(label_indices)
    
    ## Read image
    nii = nib.load(in_img_file)
    img_vec = nii.get_fdata().flatten().astype(int)

    ## Get counts of unique indices (excluding 0)
    img_vec = img_vec[img_vec != 0]
    u_ind, u_cnt = np.unique(img_vec, return_counts=True)

    ## Get label indices
    if label_indices.shape[0] == 0:
        # logger.warning('Label indices not provided, generating from data')
        label_indices = u_ind
    
    label_names = label_indices.astype(str)

    ## Get voxel size
    vox_size = np.prod(nii.header.get_zooms()[0:3])

    ## Get volumes for all rois
    tmp_cnt = np.zeros(np.max([label_indices.max(), u_ind.max()]) + 1)
    tmp_cnt[u_ind] = u_cnt

    ## Get volumes for selected rois
    sel_cnt = tmp_cnt[label_indices]
    sel_vol = (sel_cnt * vox_size).reshape(1,-1)
    
    ## Create dataframe
    df_out = pd.DataFrame(index = [mrid], columns = label_names, data = sel_vol)
    df_out = df_out.reset_index().rename({'index' : 'MRID'}, axis = 1)

    ##Return output dataframe
    return df_out

def append_derived_rois(df_in, derived_roi_map_file):
    '''Calculates a dataframe with the volumes of derived rois.
    Users should provide a dataframe with single roi volumes and
    a map file with the list of single roi indices for each derived roi
    '''
    ## Read derived roi map file to a dictionary
    roi_dict = {}
    with open(derived_roi_map_file) as roi_map:
        reader = csv.reader(roi_map, delimiter=',')
        for row in reader:
            key = str(row[0])
            val = [str(x) for x in row[2:]]
            roi_dict[key] = val

    # Calculate volumes for derived rois
    label_names = np.array(list(roi_dict.keys())).astype(str)
    label_vols = np.zeros(label_names.shape[0])
    for i, key in enumerate(roi_dict):
        key_vals = roi_dict[key]
        key_vol = df_in[key_vals].sum(axis=1)
        label_vols[i] = key_vol
        
    ## Create dataframe
    mrid = df_in['MRID'][0]
    df_out = pd.DataFrame(index = [mrid], columns = label_names, data = label_vols.reshape(1,-1))
    df_out = df_out.reset_index().rename({'index' : 'MRID'}, axis = 1)

    ##Return output dataframe
    return df_out

###---------calculate ROI volumes-----------
def create_roi_csv(scan_id, in_roi, list_single_roi, map_derived_roi, out_img, out_csv):

    ## Calculate MUSE ROIs
    df_map = pd.read_csv(list_single_roi)
    
    # Add ROI for cortical CSF with index set to 1
    df_map = df_map.append({'IndexMUSE':1, 'ROINameMUSE':'Cortical CSF'}, ignore_index = True)
    df_map = df_map.sort_values('IndexMUSE')
    
    list_roi = df_map.IndexMUSE.tolist()[1:]
    df_muse = calc_roi_volumes(scan_id, in_roi, list_roi)

    ## Calculate Derived ROIs
    df_dmuse = append_derived_rois(df_muse, map_derived_roi)

    ## Write input roi image as out img
    nii = nib.load(in_roi)
    nii.to_filename(out_img)

    ## Write out csv
    df_dmuse.to_csv(out_csv, index = False)
    
###---------calculate ROI volumes-----------
def extract_roi_masks(in_roi, map_derived_roi, out_pref):
    '''Create individual roi masks for single and derived rois
    '''
    img_ext_type = '.nii.gz'

    ## Read image
    in_nii = nib.load(in_roi)
    img_mat = in_nii.get_fdata().astype(int)

    ## Read derived roi map file to a dictionary
    roi_dict = {}
    with open(map_derived_roi) as roi_map:
        reader = csv.reader(roi_map, delimiter=',')
        for row in reader:
            key = str(row[0])
            val = [int(x) for x in row[2:]]
            roi_dict[key] = val

    # Create an individual roi mask for each roi
    for i, key in enumerate(roi_dict):
        print(i)
        key_vals = roi_dict[key]
        tmp_mask = np.isin(img_mat, key_vals).astype(int)
        out_nii = nib.Nifti1Image(tmp_mask, in_nii.affine, in_nii.header)
        nib.save(out_nii, out_pref + '_' + str(key) + img_ext_type)