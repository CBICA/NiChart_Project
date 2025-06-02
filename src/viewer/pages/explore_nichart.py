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

def view_segmentation(method) -> None:
    """
    Panel for viewing segmentations
    """
    with st.expander('Display Parameters', expanded=True):
        
        # Select result type        
        list_res_type = ['Segmentation', 'Volumes']
        st.markdown('Output type:')
        sel_res_type = st.pills(
            'Select result type',
            list_res_type,
            default = 'Segmentation',
            selection_mode = 'single',
            label_visibility = 'collapsed',
        )

        if sel_res_type == 'Segmentation':
            # Create a list of checkbox options
            st.markdown('Viewing planes:')
            list_orient = st.multiselect(
                "Select viewing planes:",
                utilni.img_views, 
                utilni.img_views,
                disabled=False,
                label_visibility = 'collapsed'
            )

            # View hide overlay
            is_show_overlay = st.checkbox("Show overlay", True, disabled=False)

            # Crop to mask area
            crop_to_mask = st.checkbox("Crop to mask", True, disabled=False)

        # Create combo list for selecting target ROI
        list_roi_names = utilroi.get_roi_names(st.session_state.dicts["muse_sel"])
        sel_var = st.selectbox(
            "ROI", list_roi_names, key="selbox_rois", index=None, disabled=False
        )
        
        # Get indices for the selected var
        list_rois = utilroi.get_list_rois(
            sel_var,
            st.session_state.rois["roi_dict_inv"],
            st.session_state.rois["roi_dict_derived"],
        )
        if list_rois is None:
            return


    with st.container():
        if sel_res_type == 'Segmentation': 

            # Select images
            if method == 'dlmuse':
                ulay = st.session_state.ref_data["t1img"]
                olay = st.session_state.ref_data["dlmuse"]
            elif method == 'dlwmls':
                ulay = st.session_state.ref_data["flimg"]
                olay = st.session_state.ref_data["dlwmls"]

            # Select images
            with st.spinner("Wait for it..."):
                # Process image and mask to prepare final 3d matrix to display
                img, mask, img_masked = utilni.prep_image_and_olay(
                    ulay,
                    olay,
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

        elif sel_res_type == 'Volumes': 

            utilpl.panel_plot()

def view_dlwmls() -> None:
    st.write('Not there yet!')

def view_spare() -> None:
    st.write('Not there yet!')

#st.info(
st.markdown(
    """
    ### Explore Neuroimaging Chart
    """
)

with st.expander('Pipelines', expanded=True):

    pdict = dict(
        zip(st.session_state.pipelines['Name'], st.session_state.pipelines['Label'])
    )
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

if psel == 0:
    view_segmentation('dlmuse')

elif psel == 1:
    view_segmentation('dlwmls')
    
elif psel == 2:
    view_spare()
    
