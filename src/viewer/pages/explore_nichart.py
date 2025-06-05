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
import utils.utils_mriview as utilmriview

import pandas as pd
from streamlit_image_select import image_select

from stqdm import stqdm

import os

from utils.utils_logger import setup_logger
logger = setup_logger()

logger.debug('Start of setup!')

# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()

def view_description(method) -> None:
    """
    Panel for viewing method description
    """
    with st.container(border=True):
        fdoc = os.path.join(
            st.session_state.paths['resources'],
            'pipelines',
            method,
            'overview_' + method + '.md'
        )
        with open(fdoc, 'r') as f:
            markdown_content = f.read()
        st.markdown(markdown_content)

    
#def view_dlmuse_seg():
    #"""
    #Panel for viewing dlmuse segmentation
    #"""
    #with st.container(border=True):
        #col1, col2, col3 = st.columns(3)

        #with col1:
            ## Create combo list for selecting target ROI
            #list_roi_names = utilroi.get_roi_names(st.session_state.dicts["muse_sel"])
            #sel_var = st.selectbox(
                #"ROI", list_roi_names, key="selbox_rois", index=None, disabled=False
            #)
        
        ## Get indices for the selected var
        #list_rois = utilroi.get_list_rois(
            #sel_var,
            #st.session_state.rois["roi_dict_inv"],
            #st.session_state.rois["roi_dict_derived"],
        #)

        #if list_rois is None:
            #return

        #with col2:
            ## Create a list of checkbox options
            #list_orient = st.multiselect(
                #"Select viewing planes:",
                #utilni.img_views, 
                #utilni.img_views,
                #disabled=False,
                #label_visibility = 'collapsed'
            #)

        #with col3:
            ## View hide overlay
            #is_show_overlay = st.checkbox("Show overlay", True, disabled=False)
            ## Crop to mask area
            #crop_to_mask = st.checkbox("Crop to mask", True, disabled=False)


    #with st.container(border=True):

        ## Select images
        #ulay = st.session_state.ref_data["t1"]
        #olay = st.session_state.ref_data["dlmuse"]

        ## Select images
        #with st.spinner("Wait for it..."):
            ## Process image and mask to prepare final 3d matrix to display
            #img, mask, img_masked = utilni.prep_image_and_olay(
                #ulay,
                #olay,
                #list_rois,
                #crop_to_mask,
            #)

            ## Detect mask bounds and center in each view
            #mask_bounds = utilni.detect_mask_bounds(mask)

            ## Show images
            #blocks = st.columns(len(list_orient))
            #for i, tmp_orient in stqdm(
                #enumerate(list_orient),
                #desc="Showing images ...",
                #total=len(list_orient),
            #):
                #with blocks[i]:
                    #ind_view = utilni.img_views.index(tmp_orient)
                    #size_auto = True
                    #if is_show_overlay is False:
                        #utilst.show_img3D(
                            #img,
                            #ind_view,
                            #mask_bounds[ind_view, :],
                            #tmp_orient,
                            #size_auto,
                        #)
                    #else:
                        #utilpl.show_img3D(
                            #img_masked,
                            #ind_view,
                            #mask_bounds[ind_view, :],
                            #tmp_orient,
                            #size_auto,
                        #)

    #util_view.view_three_planar

def view_dlmuse_seg():
    """
    Panel for viewing dlmuse segmentation
    """
    ulay = st.session_state.ref_data["t1"]
    olay = st.session_state.ref_data["dlmuse"]
    
    utilmriview.view_mri(
        ulay,
        olay,
        df_rois,
        sel_roi = 'GM'
    )


def view_dlmuse_vol():
    df = pd.read_csv('/home/guraylab/GitHub/gurayerus/NiChart_Project/test_data/processed/IXI/DLMUSE/DLMUSE_Volumes.csv')
    
    st.session_state.curr_df = df
    
    utilpl.panel_plot(st.session_state.curr_df)
    
    with st.container(border=True):
        st.write(st.session_state.plots)
        for key in st.session_state:
            if key.startswith('_key'):
                st.write(f"{key}: {st.session_state[key]}")

def view_dlmuse() -> None:
    """
    Panel for viewing dlmuse results
    """
    # Select result type        
    list_res_type = ['Segmentation', 'Volumes']
    sel_res_type = st.pills(
        'Select result type',
        list_res_type,
        default = None,
        selection_mode = 'single',
        label_visibility = 'collapsed',
    )
    
    if sel_res_type == 'Segmentation':
        ulay = st.session_state.ref_data["t1"]
        olay = st.session_state.ref_data["dlmuse"]

        df_muse = st.session_state.rois["muse_derived"]
        
        utilmriview.view_mri(
            ulay,
            olay,
            df_muse,
            sel_roi = 'GM'
        )
        
    elif sel_res_type == 'Volumes':
        view_dlmuse_vol()


def view_dlwmls() -> None:
    """
    Panel for viewing dlwmls segmentation
    """
    with st.container(border=True):
        
        col1, col2, col3 = st.columns(3)

        with col1:
            # Select result type        
            list_res_type = ['Segmentation', 'Volumes']
            st.markdown('Output type:')
            sel_res_type = st.pills(
                'Select result type',
                list_res_type,
                default = None,
                selection_mode = 'single',
                label_visibility = 'collapsed',
            )
        if sel_res_type is None:
            return
        
        # Select 1 for overlay indice (lesion)
        list_rois = [1]

        if sel_res_type == 'Segmentation':

            with col3:
                # Create a list of checkbox options
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
                crop_to_mask = st.checkbox("Crop to mask", False, disabled=False)

    with st.container():
        if sel_res_type == 'Segmentation': 

            # Select images
            ulay = st.session_state.ref_data["fl"]
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


def view_spare() -> None:
    st.write('Not there yet!')

#st.info(
st.markdown(
    """
    ### Explore Neuroimaging Chart
    """
)

tab1, tab2, tab3 = st.tabs(
    ["Select Pipeline", "View Description", "View Outputs"]
)

with tab1:

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
        index=0,
        return_value="index",
        use_container_width = False
    )
    
    print(psel)
    print(pdict)

with tab2:
    view_description(list(pdict.values())[psel])
    
with tab3:
    if psel == 0:
        view_dlmuse()

    elif psel == 1:
        view_dlwmls()
        
    elif psel == 2:
        view_spare()
