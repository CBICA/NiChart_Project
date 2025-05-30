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
from streamlit_image_select import image_select

from stqdm import stqdm

import os

from utils.utils_logger import setup_logger
logger = setup_logger()

logger.debug('Start of setup!')

# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()

def view_segmentations() -> None:
    """
    Panel for viewing segmentations
    """
    with st.container(border=True):

        # Select method
        list_methods = ['DLMUSE', 'FreeSurfer']
        sel_method = utilpn.panel_select_single(
            list_methods, None, 'Method', 'sel_method'
        )
        if sel_method is None:
            return

        # Select var category
        list_cat = ['Single', 'Derived', 'Deep Structures', 'White Matter']
        sel_cat = utilpn.panel_select_single(
            list_cat, None, 'Category', 'sel_cat'
        )
        if sel_cat is None:
            return

        # Select roi name
        list_roi = ['Hippocampus', 'Thalamus']
        sel_roi = utilpn.panel_select_single(
            list_roi, None, 'Region', 'sel_roi'
        )
        if sel_roi is None:
            return

        # Select roi name
        list_side = ['Left', 'Right', 'None']
        sel_side = utilpn.panel_select_single(
            list_side, None, 'Hemisphere', 'sel_side'
        )
        if sel_side is None:
            return

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
    ### Explore Neuroimaging Chart Values
    """
)

st.markdown("##### Select pipeline")

pdict = st.session_state.pdict
pdir = os.path.join(st.session_state.paths['resources'], 'pipelines')
logo_fnames = [
    os.path.join(pdir, pname, f'logo_{pname}.png') for pname in list(pdict.values())
]
psel = image_select(
    "",
    images = logo_fnames,
    captions=list(pdict.keys()),
    index=-1,
    return_value="index",
    use_container_width = False
)

st.write(logo_fnames)
st.write(psel)

#st.write(img)

#if sel_item == 'Segmentations':
    #view_segmentations()

#elif sel_item == 'Volumes':
    #view_volumes()
    
#elif sel_item == 'Biomarkers':
    #view_biomarkers()
    
