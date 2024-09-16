import os
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

# Parameters for viewer
VIEWS = ["axial", "sagittal", "coronal"]
VIEW_AXES = [0, 2, 1]
VIEW_OTHER_AXES = [(1,2), (0,1), (0,2)]
MASK_COLOR = (0, 255, 0)  # RGB format
MASK_COLOR = np.array([0.0, 1.0, 0.0])  # RGB format
OLAY_ALPHA = 0.2


def reorient_nifti(nii_in, ref_orient = 'LPS'):
    '''
    Initial img is reoriented to a standard orientation
    '''

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

def detect_mask_bounds(mask):
    '''
    Detect the mask start, end and center in each view
    Used later to set the slider in the image viewer
    '''

    mask_bounds = np.zeros([3,3]).astype(int)
    for i, axis in enumerate(VIEW_AXES):
        mask_bounds[i,0] = 0
        mask_bounds[i,1] = mask.shape[i]
        slices_nz = np.where(np.sum(mask, axis = VIEW_OTHER_AXES[i]) > 0)[0]
        try:
            mask_bounds[i,2] = slices_nz[len(slices_nz) // 2]
        except:
            # Could not detect masked region. Set center to image center
            mask_bounds[i,2] = mask.shape[i] // 2

    return mask_bounds

def read_derived_roi_list(list_derived):
    '''
    Create a dictionary from derived roi list
    '''

    # Read list
    df = pd.read_csv(list_derived, header=None)

    # Create dict
    dict_out = {}
    for i, tmp_ind in enumerate(df[0].values):
        df_tmp = df[df[0] == tmp_ind].drop([0,1], axis =1)
        sel_vals = df_tmp.T.dropna().astype(int).values.flatten()
        dict_out[tmp_ind] = sel_vals

    return dict_out


def show_nifti(img, view, sel_axis_bounds):
    '''
    Displays the nifti img
    '''

    # Create a slider to select the slice index
    slice_index = st.slider(f"{view}", 0, sel_axis_bounds[1] - 1,
                            value=sel_axis_bounds[2], key = f'slider_{view}')

    # Extract the slice and display it
    if view == 'axial':
        st.image(img[slice_index, :, :], use_column_width = True)
    elif view == 'sagittal':
        st.image(img[:, :, slice_index], use_column_width = True)
    else:
        st.image(img[:, slice_index, :], use_column_width = True)

@st.cache_data
def prep_images(f_img, f_mask, sel_roi_ind, dict_derived):
    '''
    Read images from files and create 3D matrices for display
    '''

    # Read nifti
    nii_img = nib.load(f_img)
    nii_mask = nib.load(f_mask)

    # Reorient nifti
    nii_img = reorient_nifti(nii_img, ref_orient = 'IPL')
    nii_mask = reorient_nifti(nii_mask, ref_orient = 'IPL')

    # Extract image to matrix
    img = nii_img.get_fdata()
    mask = nii_mask.get_fdata()

    # Convert image to uint
    img = (img.astype(float) / img.max())

    # Crop image to ROIs and reshape
    img, mask = crop_image(img, mask)

    # Select target roi: derived roi
    list_rois = dict_derived[sel_roi_ind]
    mask = np.isin(mask, list_rois)

    # # Select target roi: single roi
    # mask = (mask == sel_roi_ind).astype(int)

    # Merge image and mask
    img = np.stack((img,)*3, axis=-1)

    img_masked = img.copy()
    img_masked[mask == 1] = (img_masked[mask == 1] * (1 - OLAY_ALPHA) + MASK_COLOR * OLAY_ALPHA)


    return img, mask, img_masked


# Read dataframe with data
df = pd.read_csv(st.session_state.fname_spare_csv)

# Create a dictionary of MUSE indices and names
df_muse = pd.read_csv(st.session_state.list_MUSE_all)
df_muse = df_muse[df_muse.Name.isin(df.columns)]
dict_roi = dict(zip(df_muse.Name, df_muse.Index))

# Read derived roi list and convert to a dict
dict_derived = read_derived_roi_list(st.session_state.list_MUSE_derived)


# Page controls in side bar
with st.sidebar:

    with st.container(border=True):

        # Selection of MRID
        sel_id = st.session_state.sel_id
        if sel_id == '':
            sel_ind = 0
            sel_type = '(auto)'
        else:
            sel_ind = df.MRID.tolist().index(sel_id)
            sel_type = '(user)'
        sel_id = st.selectbox("Selected Subject", df.MRID.tolist(), key=f"selbox_mrid", index = sel_ind)

        # st.sidebar.warning('Selected subject: ' + mrid)
        st.warning(f'Selected {sel_type}: {sel_id}')

        # Selection of ROI
        sel_roi = st.session_state.sel_roi
        if sel_roi == '':
            sel_ind = 0
            sel_type = '(auto)'
        else:
            sel_ind = df_muse.Name.tolist().index(sel_roi)
            sel_type = '(user)'
        sel_roi = st.selectbox("Selected ROI", list(dict_roi.keys()), key=f"selbox_rois", index = sel_ind)

    with st.container(border=True):

        # Create a list of checkbox options
        #list_orient = st.multiselect("Select viewing planes:", VIEWS, VIEWS[0])
        list_orient = st.multiselect("Select viewing planes:", VIEWS, VIEWS)

        # View hide overlay
        is_show_overlay = st.checkbox('Show overlay', True)

        # Print the selected options (optional)
        if list_orient:
            st.write("Selected options:", list_orient)


# Select roi index
sel_roi_ind = dict_roi[sel_roi]


# File names for img and mask
f_img = os.path.join(st.session_state.dir_t1img, sel_id + st.session_state.suffix_t1img)
f_mask = os.path.join(st.session_state.dir_dlmuse, sel_id + st.session_state.suffix_dlmuse)

if os.path.exists(f_img) & os.path.exists(f_mask):

    # Process image and mask to prepare final 3d matrix to display
    img, mask, img_masked = prep_images(f_img, f_mask, sel_roi_ind, dict_derived)

    # Detect mask bounds and center in each view
    mask_bounds = detect_mask_bounds(mask)

    # Show images
    blocks = st.columns(len(list_orient))
    for i, tmp_orient in enumerate(list_orient):
        with blocks[i]:
            if is_show_overlay == False:
                show_nifti(img, tmp_orient, mask_bounds[i,:])
            else:
                show_nifti(img_masked, tmp_orient, mask_bounds[i,:])

else:
    if not os.path.exists(f_img):
        st.sidebar.warning(f'Image not found: {f_img}')
    else:
        st.sidebar.warning(f'Mask not found: {f_mask}')

