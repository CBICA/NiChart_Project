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

# # Initiate Session State Values
# if 'instantiated' not in st.session_state:
#     st.session_state.plots = pd.DataFrame({'PID':[]})
#     st.session_state.pid = 1
#     st.session_state.instantiated = True

def show_nifti(img_ulay, img_olay, roi_ind, view):
    '''
    Displays the nifti img
    '''
    # Detect middle masked slice
    slices_nz = np.where(np.sum(img_olay, axis=(0, 1)) > 0)[0]
    sel_slice = slices_nz[len(slices_nz) // 2]

    # Create a slider to select the slice index
    slice_index = st.slider("Select Slice Index", 0, img_ulay.shape[2] - 1,
                            value=sel_slice, key = f'slider_{view}')

    # Extract the slice and display it
    st.image(img_ulay[:, :, slice_index])
    # st.image(img_ulay[:, :, slice_index], width=800)



# Config page
st.set_page_config(page_title="DataFrame Demo", page_icon="📊", layout='wide')

# FIXME: Input data is hardcoded here for now
# fname = "../examples/test_input/vTest1/Study1/StudyTest1_DLMUSE_All.csv"
fname = "../examples/test_input3/ROIS_tmp.csv"
df = pd.read_csv(fname)

f1 = "../examples/test_input3/IXI002-Guys-0828_T1.nii.gz"
f2 = "../examples/test_input3/IXI002-Guys-0828_T1_DLMUSE.nii.gz"

sel_roi = 'Ventricles'
mask_color = (0, 255, 0)  # RGB format

dict_roi = {'Ventricles':51, 'Hippocampus_R':47, 'Hippocampus_L':48}


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
    sel_id = st.selectbox("Select ROI", list(dict_roi.keys()), key=f"selbox_rois", index = 0)

    st.write('---')

    # Create a list of checkbox options
    orient_options = ["sagittal", "axial", "coronal"]
    list_orient = st.multiselect("Select viewing planes:", orient_options)

    # Print the selected options (optional)
    if list_orient:
        st.write("Selected options:", list_orient)



# Read nifti
nii_ulay = nib.load(f1)
nii_olay = nib.load(f2)

# Extract image to matrix
img_ulay = nii_ulay.get_fdata()
img_olay = nii_olay.get_fdata()

# Reshape img
## FIXME: image matrix should be rotated based on nifti orientation
## Here we just do it ad-hoc
img_ulay = np.rot90(img_ulay)
img_olay = np.rot90(img_olay)

# Merge image and mask
img_ulay = np.stack((img_ulay,)*3, axis=-1)
img_ulay[img_olay == sel_roi] = mask_color

# Scale values
img_ulay = img_ulay / img_ulay.max()

# Select roi index
sel_roi_ind = dict_roi[sel_roi]

# Show images
blocks = st.columns(len(list_orient))
for i, tmp_orient in enumerate(list_orient):
    with blocks[i]:
        show_nifti(img_ulay, img_olay, sel_roi_ind, tmp_orient)
