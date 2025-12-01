import os
import shutil
import time
from typing import Any

import pandas as pd
import numpy as np
import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform
from scipy import ndimage
import utils.utils_plots as utilpl
import utils.utils_misc as utilmisc
import utils.utils_user_select as utiluser

import streamlit_antd_components as sac

import streamlit as st
from stqdm import stqdm

img_views = ["axial", "coronal", "sagittal"]
VIEW_AXES = [0, 1, 2]
VIEW_OTHER_AXES = [(1, 2), (0, 2), (0, 1)]
MASK_COLOR = (0, 255, 0)  # RGB format
MASK_COLOR = np.array([0.0, 1.0, 0.0])  # RGB format
OLAY_ALPHA = 0.2

def pad_image(img: np.ndarray) -> np.ndarray:
    """
    Pad img to equal x,y,z
    """

    # Detect max size
    simg = img.shape
    mx = np.max(simg)

    # Calculate padding values to make dims equal
    pad_vals = (mx - np.array(simg)) // 2

    # Create padded image
    out_img = np.zeros([mx, mx, mx])

    # Insert image inside the padded image
    out_img[
        pad_vals[0] : pad_vals[0] + simg[0],
        pad_vals[1] : pad_vals[1] + simg[1],
        pad_vals[2] : pad_vals[2] + simg[2],
    ] = img

    return out_img

def reorient_nifti(nii_in: Any, ref_orient: str = "LPS") -> Any:
    """
    Initial img is reoriented to a standard orientation
    """

    # Find transform from current (approximate) orientation to
    # target, in nibabel orientation matrix and affine forms
    orient_in = nib.io_orientation(nii_in.affine)
    orient_out = axcodes2ornt(ref_orient)
    transform = ornt_transform(orient_in, orient_out)

    # Apply transform
    nii_reorient = nii_in.as_reoriented(transform)

    # Return reoriented image
    return nii_reorient

def crop_image(img: np.ndarray, mask: np.ndarray, crop_to_mask: bool) -> Any:
    """
    Crop img to the foreground of the mask
    """

    # Detect bounding box
    if crop_to_mask:
        nz = np.nonzero(mask)
    else:
        nz = np.nonzero(img)

    mn = np.min(nz, axis=1)
    mx = np.max(nz, axis=1)

    # Calculate crop to make all dimensions equal size
    mask_sz = mask.shape
    crop_sz = mx - mn
    new_sz = max(crop_sz)
    pad_val = (new_sz - crop_sz) // 2

    min1 = mn - pad_val
    max1 = mx + pad_val

    min2 = np.max([min1, [0, 0, 0]], axis=0)
    max2 = np.min([max1, mask_sz], axis=0)

    pad1 = list(np.max([min2 - min1, [0, 0, 0]], axis=0))
    pad2 = list(np.max([max1 - max2, [0, 0, 0]], axis=0))

    # Crop image
    img = img[min2[0] : max2[0], min2[1] : max2[1], min2[2] : max2[2]]
    mask = mask[min2[0] : max2[0], min2[1] : max2[1], min2[2] : max2[2]]

    # Pad image
    padding = np.array([pad1, pad2]).T

    if padding.sum() > 0:
        img = np.pad(img, padding, mode="constant", constant_values=0)
        mask = np.pad(mask, padding, mode="constant", constant_values=0)

    return img, mask

def detect_mask_bounds(mask: Any) -> Any:
    """
    Detect the mask start, end and center in each view
    Used later to set the slider in the image viewer
    """
    mask_bounds = np.zeros([3, 3]).astype(int)
    for i, axis in enumerate(VIEW_AXES):
        mask_bounds[i, 0] = 0
        mask_bounds[i, 1] = mask.shape[i]
        slices_nz = np.where(np.sum(mask, axis=VIEW_OTHER_AXES[i]) > 0)[0]
        try:
            mask_bounds[i, 2] = slices_nz[len(slices_nz) // 2]
        except:
            # Could not detect masked region. Set center to image center
            mask_bounds[i, 2] = mask.shape[i] // 2

    return mask_bounds

def detect_img_bounds(img: np.ndarray) -> np.ndarray:
    """
    Detect the img start, end and center in each view
    Used later to set the slider in the image viewer
    """

    img_bounds = np.zeros([3, 3]).astype(int)
    for i, axis in enumerate(VIEW_AXES):
        img_bounds[i, 0] = 0
        img_bounds[i, 1] = img.shape[i]
        img_bounds[i, 2] = img.shape[i] // 2

    return img_bounds

@st.cache_data(max_entries=1)  # type:ignore
def prep_image_and_olay(f_img: np.ndarray, f_mask: Any, list_rois: list, crop_to_mask: bool) -> Any:
    """
    Read images from files and create 3D matrices for display
    """

    print(f_img)
    print(f_mask)
    print(list_rois)

    # Read nifti
    nii_img = nib.load(f_img)
    nii_mask = nib.load(f_mask)

    # Reorient nifti
    nii_img = reorient_nifti(nii_img, ref_orient="IPL")
    nii_mask = reorient_nifti(nii_mask, ref_orient="IPL")

    # Extract image to matrix
    out_img = nii_img.get_fdata()
    out_mask = nii_mask.get_fdata()

    # Rescale image and out_mask to equal voxel size in all 3 dimensions
    out_img = ndimage.zoom(out_img, nii_img.header.get_zooms(), order=0, mode="nearest")
    out_mask = ndimage.zoom(
        out_mask, nii_mask.header.get_zooms(), order=0, mode="nearest"
    )

    # Shift values in out_img to remove negative values
    out_img = out_img - np.min([0, out_img.min()])

    # Convert image to uint
    out_img = out_img.astype(float) / out_img.max()

    # Crop image to ROIs and reshape
    out_img, out_mask = crop_image(out_img, out_mask, crop_to_mask)

    # Create mask with sel roi
    out_mask = np.isin(out_mask, list_rois)

    # Merge image and out_mask
    out_img = np.stack((out_img,) * 3, axis=-1)

    out_img_out_masked = out_img.copy()
    out_img_out_masked[out_mask == 1] = (
        # WARNING & FIXME:
        # @spirosmaggioros: I don't think this should be like this, something is wrong here with MASK_COLOR * OLAY_ALPHA
        out_img_out_masked[out_mask == 1] * (1 - OLAY_ALPHA)
        + MASK_COLOR * OLAY_ALPHA  # type:ignore
    )

    return out_img, out_mask, out_img_out_masked

@st.cache_data  # type:ignore
def prep_image(f_img: np.ndarray) -> np.ndarray:
    """
    Read image from file and create 3D matrice for display
    """

    # Read nifti
    nii_img = nib.load(f_img)

    # Reorient nifti
    nii_img = reorient_nifti(nii_img, ref_orient="IPL")

    # Extract image to matrix
    out_img = nii_img.get_fdata()

    # Rescale image to equal voxel size in all 3 dimensions
    out_img = ndimage.zoom(out_img, nii_img.header.get_zooms(), order=0, mode="nearest")

    out_img = out_img.astype(float) / out_img.max()

    # Pad image to equal size in x,y,z
    out_img = pad_image(out_img)

    # Convert image to rgb
    out_img = np.stack((out_img,) * 3, axis=-1)

    return out_img

def show_img_slices(img, scroll_axis, sel_axis_bounds, orientation, wimg = None):
    """
    Display 3D mri img slice
    """
    # Create a slider to select the slice index
    slice_index = st.slider(
        f"{orientation}", 
        0,
        sel_axis_bounds[1] - 1,
        value=sel_axis_bounds[2],
        key=f"slider_{orientation}",
    )

    # Extract the slice and display it
    if wimg is None:
        if scroll_axis == 0:
            st.image(img[slice_index, :, :], width='stretch')
        elif scroll_axis == 1:
            st.image(img[:, slice_index, :], width='stretch')
        else:
            st.image(img[:, :, slice_index], width='stretch')
    else:
        if scroll_axis == 0:
            st.image(img[slice_index, :, :], width=wimg)
        elif scroll_axis == 1:
            st.image(img[:, slice_index, :], width=wimg)
        else:
            st.image(img[:, :, slice_index], width=wimg)

def panel_select_var(sel_var_groups, plot_params, var_type, add_none = False):
    '''
    User panel to select a variable
    Variables are grouped in categories
    '''
    df_groups = st.session_state.dicts['df_var_groups'].copy()
    df_groups = df_groups[df_groups.category.isin(sel_var_groups)]

    st.markdown(f'##### Variable: {var_type}')
    cols = st.columns([1,3])
    with cols[0]:

        list_group = df_groups.group.unique().tolist()
        try:
            curr_value = plot_params[f'{var_type}_group']
            curr_index = list_group.index(curr_value)
        except ValueError:
            curr_index = 0

        st.selectbox(
            "Variable Group",
            list_group,
            key = f'_{var_type}_group',
            index = curr_index
        )
        plot_params[f'{var_type}_group'] = st.session_state[f'_{var_type}_group']

    with cols[1]:

        sel_group = plot_params[f'{var_type}_group']
        if sel_group is None:
            return

        sel_atlas = df_groups[df_groups['group'] == sel_group]['atlas'].values[0]
        list_vars = df_groups[df_groups['group'] == sel_group]['values'].values[0]

        # Convert MUSE ROI variables from index to name
        if sel_atlas == 'muse':
            roi_dict = st.session_state.dicts['muse']['ind_to_name']
            list_vars = [roi_dict[k] for k in list_vars]

        if add_none:
            list_vars = ['None'] + list_vars

        try:
            curr_value = plot_params[var_type]
            curr_index = list_vars.index(curr_value)
        except ValueError:
            curr_index = 0

        st.selectbox(
            "Variable Name",
            list_vars,
            key = f'_{var_type}',
            index = curr_index
        )

        plot_params[var_type] = st.session_state[f'_{var_type}']

def panel_set_params(
    plot_params, var_groups_data, atlas, list_vars, flag_hide_settings = False
):
    """
    Panel to set mriview parameters
    """
    if st.session_state.plot_settings['flag_hide_settings'] == 'Hide':
        return

    # Add tabs for parameter settings
    with st.expander():
        tab = sac.tabs(
            items=[
                sac.TabsItem(label='Data'),
                sac.TabsItem(label='Plot Settings')
            ],
            size='sm',
            align='left'
        )
        ## FIXME
        df_vars = st.session_state.dicts['df_var_groups']
        if tab == 'Data':
            # Select roi
            sel_var = utiluser.select_var_from_group(
                'Select ROI variable:',
                df_vars[df_vars.category.isin(['roi'])],
                plot_params['yvargroup'],
                plot_params['yvar'],
                list_vars,
                flag_add_none = False,
                dicts_rename = {
                    'muse': st.session_state.dicts['muse']['ind_to_name']
                }
            )
            plot_params['yvargroup'] = sel_var[0]
            plot_params['yvar'] = sel_var[1]
            plot_params['roi_indices'] = utilmisc.get_roi_indices(
                sel_var[1], 'muse'
            )
            st.session_state['sel_roi'] = sel_var[1]

        elif tab == 'Plot Settings':
            col1, col2, col3 = st.columns(3)
            with col1:
                # Create a list of checkbox options
                plot_params['list_orient'] = st.multiselect(
                    "Select viewing planes:",
                    img_views,
                    img_views,
                    label_visibility = 'collapsed'
                )
            with col2:
                # View hide overlay
                plot_params['is_show_overlay'] = st.checkbox("Show overlay", True, disabled=False)

            with col3:
                # Crop to mask area
                plot_params['crop_to_mask'] = st.checkbox("Crop to mask", True, disabled=False)

def panel_view_seg(ulay, olay, plot_params):
    '''
    Panel to display segmented image overlaid on underlay image
    '''
    sel_roi = st.session_state.sel_roi   # Read sele roi
    if sel_roi is None:
        return
    
    roi_indices = utilmisc.get_roi_indices(sel_roi, 'muse')
    if roi_indices is None:
        return

    # Show images
    with st.container(border=True):
        with st.spinner("Wait for it..."):
            # Process image (and mask) to prepare final 3d matrix to display
            img, mask, img_masked = prep_image_and_olay(
                #ulay, olay, plot_params['roi_indices'], plot_params['crop_to_mask']
                ulay, olay, roi_indices, plot_params['crop_to_mask']
            )
            img_bounds = detect_mask_bounds(mask)

            cols = st.columns(len(plot_params['list_orient']))
            for i, tmp_orient in stqdm(
                enumerate(plot_params['list_orient']),
                desc="Showing images ...",
                total=len(plot_params['list_orient'])
            ):
                with cols[i]:
                    ind_view = img_views.index(tmp_orient)
                    size_auto = True
                    if olay is None or plot_params['is_show_overlay'] is False:
                        show_img_slices(
                            img, ind_view, img_bounds[ind_view, :], tmp_orient
                        )
                    else:
                        show_img_slices(
                            img_masked, ind_view, img_bounds[ind_view, :], tmp_orient
                        )
