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

import streamlit as st
from stqdm import stqdm

img_views = ["axial", "coronal", "sagittal"]
VIEW_AXES = [0, 1, 2]
VIEW_OTHER_AXES = [(1, 2), (0, 2), (0, 1)]
MASK_COLOR = (0, 255, 0)  # RGB format
MASK_COLOR = np.array([0.0, 1.0, 0.0])  # RGB format
OLAY_ALPHA = 0.2

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
            st.image(img[slice_index, :, :], use_container_width=True)
        elif scroll_axis == 1:
            st.image(img[:, slice_index, :], use_container_width=True)
        else:
            st.image(img[:, :, slice_index], use_container_width=True)
    else:
        if scroll_axis == 0:
            st.image(img[slice_index, :, :], width=w_img)
        elif scroll_axis == 1:
            st.image(img[:, slice_index, :], width=w_img)
        else:
            st.image(img[:, :, slice_index], width=w_img)

def panel_view_seg(ulay, olay, method):
    '''
    Panel to display segmented image overlaid on underlay image
    '''
    flag_settings = st.sidebar.checkbox('Hide settings')
    ss_sel = st.session_state.selections
    
    # Add tabs for parameter settings
    with st.container(border=True):
        if not flag_settings:
            ptab1, ptab2, = st.tabs(
                ['Data', 'Plot Settings']
            )        
            with ptab1:
                ss_sel['sel_roi'] = utilpl.panel_select_roi(method, '_seg')
                ss_sel['list_roi_indices'] = utilmisc.get_roi_indices(ss_sel['sel_roi'], method)

            with ptab2:
                col1, col2, col3 = st.columns(3)
                with col1:
                    # Create a list of checkbox options
                    ss_sel['list_orient'] = st.multiselect(
                        "Select viewing planes:",
                        img_views, 
                        img_views,
                        label_visibility = 'collapsed'
                    )
                with col2:
                    # View hide overlay
                    ss_sel['is_show_overlay'] = st.checkbox("Show overlay", True, disabled=False)

                with col3:
                    # Crop to mask area
                    ss_sel['crop_to_mask'] = st.checkbox("Crop to mask", True, disabled=False)

    if ss_sel['list_roi_indices'] is None:
        return

    # Show images
    with st.container(border=True):
        with st.spinner("Wait for it..."):
            # Process image (and mask) to prepare final 3d matrix to display
            img, mask, img_masked = prep_image_and_olay(
                ulay, olay, ss_sel['list_roi_indices'], ss_sel['crop_to_mask']
            )
            img_bounds = detect_mask_bounds(mask)

            cols = st.columns(len(ss_sel['list_orient']))
            for i, tmp_orient in stqdm(
                enumerate(ss_sel['list_orient']),
                desc="Showing images ...",
                total=len(ss_sel['list_orient'])
            ):
                with cols[i]:
                    ind_view = img_views.index(tmp_orient)
                    size_auto = True
                    if olay is None or ss_sel['is_show_overlay'] is False:
                        show_img_slices(
                            img, ind_view, img_bounds[ind_view, :], tmp_orient
                        )
                    else:
                        show_img_slices(
                            img_masked, ind_view, img_bounds[ind_view, :], tmp_orient
                        )
