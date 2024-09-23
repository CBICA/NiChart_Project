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

import utils_st as utilst
import utils_nifti as utilni
import numpy as np
import nibabel as nib

# Parameters for viewer
VIEWS = ["axial", "sagittal", "coronal"]
VIEW_AXES = [0, 2, 1]
VIEW_OTHER_AXES = [(1,2), (0,1), (0,2)]
MASK_COLOR = (0, 255, 0)  # RGB format
MASK_COLOR = np.array([0.0, 1.0, 0.0])  # RGB format
OLAY_ALPHA = 0.2

def read_derived_roi_list(list_sel_rois, list_derived):
    '''
    Create a dictionary from derived roi list
    '''

    # Read list
    df_sel = pd.read_csv(list_sel_rois)
    df = pd.read_csv(list_derived, header=None)

    # Keep only selected ROIs
    df = df[df[0].isin(df_sel.Index)]

    # Create dict of roi names and indices
    dict_roi = dict(zip(df[1], df[0]))

    # Create dict of roi indices and derived indices
    dict_derived = {}
    for i, tmp_ind in enumerate(df[0].values):
        df_tmp = df[df[0] == tmp_ind].drop([0,1], axis =1)
        sel_vals = df_tmp.T.dropna().astype(int).values.flatten()
        dict_derived[tmp_ind] = sel_vals


    return dict_roi, dict_derived


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
def prep_images(f_img, f_mask, sel_var_ind, dict_derived):
    '''
    Read images from files and create 3D matrices for display
    '''

    # Read nifti
    nii_img = nib.load(f_img)
    nii_mask = nib.load(f_mask)

    # Reorient nifti
    nii_img = utilni.reorient_nifti(nii_img, ref_orient = 'IPL')
    nii_mask = utilni.reorient_nifti(nii_mask, ref_orient = 'IPL')

    # Extract image to matrix
    img = nii_img.get_fdata()
    mask = nii_mask.get_fdata()

    # Convert image to uint
    img = (img.astype(float) / img.max())

    # Crop image to ROIs and reshape
    img, mask = utilni.crop_image(img, mask)

    # Select target roi: derived roi
    list_rois = dict_derived[sel_var_ind]
    mask = np.isin(mask, list_rois)

    # # Select target roi: single roi
    # mask = (mask == sel_var_ind).astype(int)

    # Merge image and mask
    img = np.stack((img,)*3, axis=-1)

    img_masked = img.copy()
    img_masked[mask == 1] = (img_masked[mask == 1] * (1 - OLAY_ALPHA) + MASK_COLOR * OLAY_ALPHA)


    return img, mask, img_masked


# Page controls in side bar
#with st.sidebar:

f_img = ''
f_mask = ''


# Selection of subject list and image paths
with st.expander('Select subject list, image paths and suffixes'):
    
    # DLMUSE file name
    helpmsg = 'Input csv file with DLMUSE ROI volumes.\n\nUsed for selecting the MRID and the ROI name.\n\nChoose the file by typing it into the text field or using the file browser to browse and select it'
    csv_dlmuse, csv_path = utilst.user_input_file("Select file",
                                                    'btn_input_dlmuse',
                                                    "DLMUSE ROI file",
                                                    st.session_state.path_last_sel,
                                                    st.session_state.path_csv_dlmuse,
                                                    helpmsg)
    if os.path.exists(csv_dlmuse):
        st.session_state.path_csv_dlmuse = csv_dlmuse
        st.session_state.path_last_sel = csv_path

    # Input T1 image folder
    helpmsg = 'Path to T1 images.\n\nChoose the path by typing it into the text field or using the file browser to browse and select it'
    path_t1 = utilst.user_input_folder("Select folder",
                                    'btn_indir_t1',
                                    "Input folder",
                                    st.session_state.path_last_sel,
                                    st.session_state.path_t1,
                                    helpmsg)
    st.session_state.path_t1 = path_t1
    
    # Input DLMUSE image folder
    helpmsg = 'Path to DLMUSE images.\n\nChoose the path by typing it into the text field or using the file browser to browse and select it'
    path_dlmuse = utilst.user_input_folder("Select folder",
                                            'btn_indir_dlmuse',
                                            "Input folder",
                                            st.session_state.path_last_sel,
                                            st.session_state.path_dlmuse,
                                            helpmsg)
    st.session_state.path_dlmuse = path_dlmuse

    # T1 suffix
    suff_t1img = utilst.user_input_text("T1 img suffix", 
                                        st.session_state.suff_t1img, 
                                        helpmsg)
    st.session_state.suff_t1img = suff_t1img

    # DLMUSE suffix
    suff_dlmuse = utilst.user_input_text("DLMUSE image suffix", 
                                        st.session_state.suff_dlmuse, 
                                        helpmsg)
    st.session_state.suff_dlmuse = suff_dlmuse
        

# Selection of MRID and ROI name
if os.path.exists(st.session_state.path_csv_dlmuse):

    with st.container(border=True):

        df = pd.read_csv(st.session_state.path_csv_dlmuse)

        # Create a dictionary of MUSE indices and names
        df_muse = pd.read_csv(st.session_state.dict_muse_all)

        #df_muse = df_muse[df_muse.Name.isin(df.columns)]
        #dict_roi = dict(zip(df_muse.Name, df_muse.Index))

        # Read derived roi list and convert to a dict
        dict_roi, dict_derived = read_derived_roi_list(st.session_state.dict_muse_sel,
                                             st.session_state.dict_muse_derived)

        # Selection of MRID
        sel_mrid = st.session_state.sel_mrid
        if sel_mrid == '':
            sel_ind = 0
            sel_type = '(auto)'
        else:
            sel_ind = df.MRID.tolist().index(sel_mrid)
            sel_type = '(user)'
        sel_mrid = st.selectbox("MRID", df.MRID.tolist(), key=f"selbox_mrid", index = sel_ind)

        # Selection of ROI
        #  - The variable will be selected from the active plot
        
        sel_var = ''
        try: 
            sel_var = st.session_state.plots.loc[st.session_state.plot_active, 'yvar']
        except:
            print('Could not detect an active plot')
        if sel_var == '':
            sel_ind = 2
            sel_var = list(dict_roi.keys())[0]
            sel_type = '(auto)'
        else:
            sel_ind = df_muse.Name.tolist().index(sel_var)
            sel_type = '(user)'
        sel_var = st.selectbox("ROI", list(dict_roi.keys()), key=f"selbox_rois", index = sel_ind)

    with st.container(border=True):

        # Create a list of checkbox options
        #list_orient = st.multiselect("Select viewing planes:", VIEWS, VIEWS[0])
        list_orient = st.multiselect("Select viewing planes:", VIEWS, VIEWS)

        # View hide overlay
        is_show_overlay = st.checkbox('Show overlay', True)


    # Select roi index
    sel_var_ind = dict_roi[sel_var]

    # File names for img and mask
    f_img = os.path.join(st.session_state.path_out, 
                        st.session_state.path_t1,
                        sel_mrid + st.session_state.suff_t1img)

    f_mask = os.path.join(st.session_state.path_out, 
                        st.session_state.path_dlmuse,
                        sel_mrid + st.session_state.suff_dlmuse)

if os.path.exists(f_img) & os.path.exists(f_mask):

    # Process image and mask to prepare final 3d matrix to display
    img, mask, img_masked = prep_images(f_img, f_mask, sel_var_ind, dict_derived)

    # Detect mask bounds and center in each view
    mask_bounds = utilni.detect_mask_bounds(mask)

    # Show images
    blocks = st.columns(len(list_orient))
    for i, tmp_orient in enumerate(list_orient):
        with blocks[i]:
            ind_view = VIEWS.index(tmp_orient)
            if is_show_overlay == False:
                show_nifti(img, tmp_orient, mask_bounds[ind_view,:])
            else:
                show_nifti(img_masked, tmp_orient, mask_bounds[ind_view,:])

else:
    if not os.path.exists(f_img):
        st.warning(f'Image not found: {f_img}')
    else:
        st.warning(f'Mask not found: {f_mask}')

