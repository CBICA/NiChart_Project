import os
import shutil
import time
from typing import Any

import pandas as pd
import streamlit as st
import utils.utils_doc as utildoc
import utils.utils_io as utilio
import utils.utils_session as utilss
import utils.utils_rois as utilroi
import utils.utils_nifti as utilnii
from stqdm import stqdm
import utils.utils_st as utilst

def panel_select_roi(roi_type):
    '''
    User panel to select an ROI
    '''
    ## MUSE ROIs
    if roi_type == 'muse':
        
        # Read dictionaries
        df_derived = st.session_state.rois['muse']['df_derived']
        df_groups = st.session_state.rois['muse']['df_groups']
        
        col1, col2 = st.columns([1,3])
        
        # Select roi group
        with col1:
            list_group = df_groups.Name.unique()
            sel_group = st.selectbox(
                "Select ROI Group",
                list_group,
                None,
                help="Select ROI group"
            )
            if sel_group is None:
                return None
    
        # Select roi
        with col2:
            sel_indices = df_groups[df_groups.Name == sel_group]['List'].values[0]
                    
            list_roi = df_derived[df_derived.Index.isin(sel_indices)].Name.tolist()
            sel_roi = st.selectbox(
                "Select ROI",
                list_roi,
                None,
                help="Select an ROI from the list"
            )
        
        return sel_roi

def get_roi_indices(sel_roi, roi_type):
    '''
    Detect indices for a selected ROI
    '''
    if sel_roi is None:
        return None
    
    # Detect indices
    if roi_type == 'muse':
        df_derived = st.session_state.rois['muse']['df_derived']
        list_roi_indices = df_derived[df_derived.Name == sel_roi].List.values[0]
        return list_roi_indices

    return None

def panel_settings_seg():
    '''
    User panel to select settings for a viewer for segmentation
    '''
    col1, col2, col3 = st.columns(3)
    with col1:
        # Create a list of checkbox options
        list_orient = st.multiselect(
            "Select viewing planes:",
            utilnii.img_views, 
            utilnii.img_views,
            label_visibility = 'collapsed'
        )

    with col2:
        # View hide overlay
        is_show_overlay = st.checkbox("Show overlay", True, disabled=False)

    with col3:
        # Crop to mask area
        crop_to_mask = st.checkbox("Crop to mask", True, disabled=False)

    return list_orient, is_show_overlay, crop_to_mask

def panel_view_img_slices(
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

def panel_view_seg(ulay, olay, roi_type):
    flag_settings = st.sidebar.checkbox('Hide plot settings')
    flag_data = st.sidebar.checkbox('Hide data settings')
    
    # Add settings tabs
    with st.container(border=True):
        if not flag_settings:
            ptab1, ptab2, = st.tabs(
                ['Data', 'Plot Settings']
            )        
            with ptab1:
                sel_roi = panel_select_roi(roi_type)
                list_roi_indices = get_roi_indices(sel_roi, roi_type)

            with ptab2:
                list_orient, is_show_overlay, crop_to_mask = panel_settings_seg()

    if list_roi_indices is None:
        return

    with st.container(border=True):
        with st.spinner("Wait for it..."):
            # Process image (and mask) to prepare final 3d matrix to display
            img, mask, img_masked = utilnii.prep_image_and_olay(
                ulay, olay, list_roi_indices, crop_to_mask
            )
            img_bounds = utilnii.detect_mask_bounds(mask)

            # Show images
            cols = st.columns(len(list_orient))
            for i, tmp_orient in stqdm(
                enumerate(list_orient), desc="Showing images ...", total=len(list_orient)
            ):
                with cols[i]:
                    ind_view = utilnii.img_views.index(tmp_orient)
                    size_auto = True
                    if olay is None or is_show_overlay is False:
                        panel_view_img_slices(
                            img, ind_view, img_bounds[ind_view, :], tmp_orient
                        )
                    else:
                        panel_view_img_slices(
                            img_masked, ind_view, img_bounds[ind_view, :], tmp_orient
                        )
