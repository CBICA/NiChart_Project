import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import plotly.express as px
from math import ceil
import nibabel as nib
import numpy as np
from nibabel.orientations import axcodes2ornt, ornt_transform

# # Initiate Session State Values
# if 'instantiated' not in st.session_state:
#     st.session_state.plots = pd.DataFrame({'PID':[]})
#     st.session_state.pid = 1
#     st.session_state.instantiated = True


def reorient_nifti(nii_in, ref_orient = 'LPS'):

    # Find transform from current (approximate) orientation to
    # target, in nibabel orientation matrix and affine forms
    orient_in = nib.io_orientation(nii_in.affine)
    orient_out = axcodes2ornt(ref_orient)
    transform = ornt_transform(orient_in, orient_out)

    # Apply transform
    nii_reorient = nii_in.as_reoriented(transform)

    # Return reoriented image
    return nii_reorient

def crop_image(img, mask):
    '''
    Crop img to the foreground of the mask
    '''


    # Detect bounding box
    nz = np.nonzero(mask)
    mn = np.min(nz, axis=1)
    mx = np.max(nz, axis=1)

    # Calculate crop to make all dimensions equal size
    mask_sz = mask.shape
    crop_sz = mx - mn 
    new_sz = max(crop_sz)
    pad_val = (new_sz - crop_sz) // 2
    
    min1 = mn - pad_val
    max1 = mx + pad_val

    min2 = np.max([min1, [0,0,0]], axis=0)
    max2 = np.min([max1, mask_sz], axis=0)
    
    pad1 = list(np.max([min2-min1, [0,0,0]], axis=0))
    pad2 = list(np.max([max1-max2, [0,0,0]], axis=0))
    
    # Crop image
    img = img[min2[0]:max2[0], min2[1]:max2[1], min2[2]:max2[2]]
    mask = mask[min2[0]:max2[0], min2[1]:max2[1], min2[2]:max2[2]]

    # Pad image
    padding = np.array([pad1, pad2]).T
    img = np.pad(img, padding, mode='constant', constant_values=0)
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    
    return img, mask
        

def show_nifti(img, mask, view):
    '''
    Displays the nifti img
    '''
    
    # Set parameters based on orientation
    if view == 'axial':
        sel_axis = 0
        other_axes = (1,2)
        
    if view == 'sagittal':
        sel_axis = 2
        other_axes = (0,1)

    if view == 'coronal':
        sel_axis = 1
        other_axes = (0,2)


    # Detect middle masked slice
    slices_nz = np.where(np.sum(mask, axis = other_axes) > 0)[0]
    sel_slice = slices_nz[len(slices_nz) // 2]
    
    # Create a slider to select the slice index
    slice_index = st.slider(f"{view}", 0, img.shape[sel_axis] - 1,
                            value=sel_slice, key = f'slider_{view}')

    # Extract the slice and display it
    if view == 'axial':
        st.image(img[slice_index, :, :], use_column_width = True)
    elif view == 'sagittal':
        st.image(img[:, :, slice_index], use_column_width = True)
    else:
        st.image(img[:, slice_index, :], use_column_width = True)



# # Config page
# st.set_page_config(page_title="DataFrame Demo", page_icon="📊", layout='wide')

# FIXME: Input data is hardcoded here for now
# fname = "../examples/test_input/vTest1/Study1/StudyTest1_DLMUSE_All.csv"
fname = "../examples/test_input3/ROIS_tmp.csv"
df = pd.read_csv(fname)

f1 = "../examples/test_input3/IXI002-Guys-0828_T1.nii.gz"
f2 = "../examples/test_input3/IXI002-Guys-0828_T1_DLMUSE.nii.gz"

sel_roi = 'Ventricles'
mask_color = (0, 255, 0)  # RGB format

dict_roi = {'Ventricles':51, 'Hippocampus_R':100, 'Hippocampus_L':48}


# Page controls in side bar
with st.sidebar:

    # Show selected id (while providing the user the option to select it from the list of all MRIDs)
    # - get the selected id from the session_state
    # - create a selectbox with all MRIDs
    # -- initialize it with the selected id if it's set
    # -- initialize it with the first id if not
    sel_id = st.session_state.sel_id
    if sel_id == '':
        sel_ind = 0
        sel_type = '(auto)'
    else:
        sel_ind = df.MRID.tolist().index(sel_id)
        sel_type = '(user)'
    sel_id = st.selectbox("Select Subject", df.MRID.tolist(), key=f"selbox_mrid", index = sel_ind)

    # st.sidebar.warning('Selected subject: ' + mrid)
    st.warning(f'Selected {sel_type}: {sel_id}')

    st.write('---')

    ## FIXME: read list of rois from dictionary
    ##        show the roi selected in the plot
    sel_roi = st.selectbox("Select ROI", list(dict_roi.keys()), key=f"selbox_rois", index = 0)

    st.write('---')

    # Create a list of checkbox options
    orient_options = ["axial", "sagittal", "coronal"]
    #list_orient = st.multiselect("Select viewing planes:", orient_options, orient_options[0])
    list_orient = st.multiselect("Select viewing planes:", orient_options, orient_options)

    # Print the selected options (optional)
    if list_orient:
        st.write("Selected options:", list_orient)

# Select roi index
sel_roi_ind = dict_roi[sel_roi]

# Read nifti
nii_ulay = nib.load(f1)
nii_olay = nib.load(f2)

# Reorient nifti
nii_ulay = reorient_nifti(nii_ulay, ref_orient = 'IPL')
nii_olay = reorient_nifti(nii_olay, ref_orient = 'IPL')

# Extract image to matrix
img_ulay = nii_ulay.get_fdata()
img_olay = nii_olay.get_fdata()

# Crop image to ROIs and reshape
img_ulay, img_olay = crop_image(img_ulay, img_olay)

# Select target roi
img_olay = (img_olay == sel_roi_ind).astype(int)

# Merge image and mask
img_ulay = np.stack((img_ulay,)*3, axis=-1)
img_ulay[img_olay == 1] = mask_color

# Scale values
img_ulay = img_ulay / img_ulay.max()


# Show images
blocks = st.columns(len(list_orient))
for i, tmp_orient in enumerate(list_orient):
    with blocks[i]:
        show_nifti(img_ulay, img_olay, tmp_orient)
