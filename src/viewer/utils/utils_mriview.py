import os
import shutil
from typing import Any, Optional

import pandas as pd
import numpy as np
import streamlit as st
import utils.utils_io as utilio
import utils.utils_nifti as utilnii
import utils.utils_session as utilses

import plotly.graph_objs as go
import utils.utils_trace as utiltr
from stqdm import stqdm

VIEW_AXES = [0, 1, 2]
VIEW_OTHER_AXES = [(1, 2), (0, 2), (0, 1)]

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

def show_img_slices(
    img, scroll_axis, sel_axis_bounds, orientation, wimg = None,
):
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


def view_mri(
    ulay,
    olay = None,
    df_rois = None,
    sel_roi = None
):
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            # Select ROI name 
            list_roi_names = df_rois.Name.sort_values().tolist()
            sel_var = st.selectbox(
                "ROI", list_roi_names, key="selbox_rois", index=None, disabled=False
            )        
        if sel_var is None:
            return
        
        # Get indice for the selected roi
        df_sel  = df_rois[df_rois.Name == sel_var]
        if 'List' in df_sel:
            list_roi_indices = df_sel.List.values[0]
        else:
            list_roi_indices = [df_sel.Index.values[0]]

        with col2:
            # Create a list of checkbox options
            list_orient = st.multiselect(
                "Select viewing planes:",
                utilnii.img_views, 
                utilnii.img_views,
                disabled=False,
                label_visibility = 'collapsed'
            )

        with col3:
            if olay is not None:
                # View hide overlay
                is_show_overlay = st.checkbox("Show overlay", True, disabled=False)
                # Crop to mask area
                crop_to_mask = st.checkbox("Crop to mask", True, disabled=False)


    with st.container(border=True):
        with st.spinner("Wait for it..."):
            # Process image (and mask) to prepare final 3d matrix to display
            if olay is None:
                img = utilnii.prep_image(ulay)
                img_bounds = detect_img_bounds(img)
                
            else:
                img, mask, img_masked = utilnii.prep_image_and_olay(
                    ulay, olay, list_roi_indices, crop_to_mask
                )
                img_bounds = detect_mask_bounds(mask)

            # Show images
            blocks = st.columns(len(list_orient))
            for i, tmp_orient in stqdm(
                enumerate(list_orient),
                desc="Showing images ...",
                total=len(list_orient),
            ):
                with blocks[i]:
                    ind_view = utilnii.img_views.index(tmp_orient)
                    size_auto = True
                    if olay is None or is_show_overlay is False:
                        show_img_slices(
                            img,
                            ind_view,
                            img_bounds[ind_view, :],
                            tmp_orient,
                        )
                    else:
                        show_img_slices(
                            img_masked,
                            ind_view,
                            img_bounds[ind_view, :],
                            tmp_orient,
                        )
