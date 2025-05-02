import streamlit as st
import utils.utils_pages as utilpg
import utils.utils_panels as utilpn
import utils.utils_doc as utildoc
import utils.utils_io as utilio
import utils.utils_session as utilss
import utils.utils_rois as utilroi
import utils.utils_nifti as utilni
import utils.utils_st as utilst
import utils.utils_plots as utilpl

from stqdm import stqdm

import os

from utils.utils_logger import setup_logger
logger = setup_logger()

logger.debug('Start of setup!')

# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()

def view_dlmuse_seg() -> None:
    """
    Panel for viewing DLMUSE segmentations
    """
    with st.container(border=True):

        # Create combo list for selecting target ROI
        list_roi_names = utilroi.get_roi_names(st.session_state.dicts["muse_sel"])
        sel_var = st.selectbox(
            "ROI", list_roi_names, key="selbox_rois", index=None, disabled=False
        )
        if sel_var is None:
            st.warning("Please select the ROI!")
            return

        # Create a list of checkbox options
        list_orient = st.multiselect(
            "Select viewing planes:", utilni.img_views, utilni.img_views, disabled=False
        )

        if list_orient is None or len(list_orient) == 0:
            st.warning("Please select the viewing plane!")
            return

        # View hide overlay
        is_show_overlay = st.checkbox("Show overlay", True, disabled=False)

        # Crop to mask area
        crop_to_mask = st.checkbox("Crop to mask", True, disabled=False)

        # Get indices for the selected var
        list_rois = utilroi.get_list_rois(
            sel_var,
            st.session_state.rois["roi_dict_inv"],
            st.session_state.rois["roi_dict_derived"],
        )

        if list_rois is None:
            st.warning("ROI list is empty!")
            return

        # Select images
        with st.spinner("Wait for it..."):
            # Process image and mask to prepare final 3d matrix to display
            img, mask, img_masked = utilni.prep_image_and_olay(
                st.session_state.ref_data["t1img"],
                st.session_state.ref_data["dlmuse"],
                list_rois,
                crop_to_mask,
            )

            # Detect mask bounds and center in each view
            mask_bounds = utilni.detect_mask_bounds(mask)

            # Show images
            blocks = st.columns(len(list_orient))
            for i, tmp_orient in stqdm(
                enumerate(list_orient),
                desc="Showing images ...",
                total=len(list_orient),
            ):
                with blocks[i]:
                    ind_view = utilni.img_views.index(tmp_orient)
                    size_auto = True
                    if is_show_overlay is False:
                        utilst.show_img3D(
                            img,
                            ind_view,
                            mask_bounds[ind_view, :],
                            tmp_orient,
                            size_auto,
                        )
                    else:
                        utilpl.show_img3D(
                            img_masked,
                            ind_view,
                            mask_bounds[ind_view, :],
                            tmp_orient,
                            size_auto,
                        )

def view_centiles_dlmuse() -> None:
    
    utilpl.panel_plot()

#st.info(
st.markdown(
    """
    ### Neuroimaging Chart Viewer
    - View NiChart imaging variables and biomarkers derived from reference dataset
    """
)

if 'key_setup_sel_item' not in st.session_state:
    st.session_state.key_setup_sel_item = None

sel_item = st.pills(
    "Select NiChart Item",
    ["Anatomical Brain Segmentation", "Regional Brain Volumes", "ML-Based Brain Signatures"],
    selection_mode="single",
    key='key_setup_sel_item',
    label_visibility="collapsed",
)

## Required to make sure that state of widget is consistent with returned value
#if st.session_state._setup_sel_item != sel_item:
    #st.session_state._setup_sel_item = sel_item
    #st.rerun()    

if sel_item == 'Anatomical Brain Segmentation':
    view_dlmuse_seg()

elif sel_item == 'Regional Brain Volumes':
    view_centiles_dlmuse()
    
elif sel_item == 'ML-Based Brain Signatures':
    view_ml_centiles()
    
