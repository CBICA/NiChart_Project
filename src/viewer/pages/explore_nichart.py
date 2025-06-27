import streamlit as st
import utils.utils_pages as utilpg
import utils.utils_plots as utilpl
import utils.utils_mriview as utilmri
import utils.utils_session as utilses
import pandas as pd
from streamlit_image_select import image_select
from stqdm import stqdm
import os
from utils.utils_logger import setup_logger

import streamlit_antd_components as sac

logger = setup_logger()
logger.debug('Page: Explore Nichart')

# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()

def view_description(pipeline) -> None:
    """
    Panel for viewing pipeline description
    """
    with st.container(border=True):
        f_logo = os.path.join(
            st.session_state.paths['resources'], 'pipelines', pipeline, f'logo_{pipeline}.png'
        )
        fdoc = os.path.join(
            st.session_state.paths['resources'], 'pipelines', pipeline, f'overview_{pipeline}.md'
        )
        cols = st.columns([6, 1])
        with cols[0]:
            with open(fdoc, 'r') as f:
                st.markdown(f.read())
        with cols[1]:
            st.image(f_logo)

def pipeline_overviews():
    '''
    Select a pipeline and show overview
    '''
    with st.container(border=True):
        pdict = dict(
            zip(st.session_state.pipelines['Name'], st.session_state.pipelines['Label'])
        )
        
        sitems = []
        colors = ['blue', 'red', 'pink', 'teal', 'grape', 'indigo', 'lime', 'orange']
        #'#25C3B0'
        for i, tmp_key in enumerate(pdict.keys()):
            sitems.append(
                sac.ButtonsItem(
                    label=tmp_key, color = colors[i]
                )
            )
        
        sel_pipeline = sac.buttons(
            items=sitems,
            size='lg',
            radius='xl',
            align='left'
        )
            
        # Show description of the selected pipeline
        view_description(pdict[sel_pipeline])
        
        return(sel_pipeline)


def view_synthseg() -> None:
    """
    Panel for viewing synthseg results
    """    
    # Select result type 
    st.info('Coming soon!')

def view_dlmuse() -> None:
    """
    Panel for viewing dlmuse results
    """
    list_res_type = ['Segmentation', 'Volumes']
    sel_res_type = sac.tabs(
        list_res_type,
        size='lg',
        align='left'
    )   
    if sel_res_type == 'Segmentation':
        ulay = st.session_state.ref_data["t1"]
        olay = st.session_state.ref_data["dlmuse"]        
        utilmri.panel_view_seg(ulay, olay, 'muse')
        
    elif sel_res_type == 'Volumes':
        st.session_state.curr_df = None
        utilpl.panel_view_centiles('dlmuse', 'rois')
        
def view_dlwmls() -> None:
    """
    Panel for viewing dlwmls segmentation
    """
    # Select result type        
    list_res_type = ['Segmentation']
    sel_res_type = st.pills(
        'Select result type',
        list_res_type,
        default = None,
        selection_mode = 'single',
        label_visibility = 'collapsed',
    )
    
    if sel_res_type == 'Segmentation':
        ulay = st.session_state.ref_data["fl"]
        olay = st.session_state.ref_data["dlwmls"]        
        utilmri.panel_view_seg(ulay, olay, 'dlwmls')

def view_dlmuse_biomarkers() -> None:
    """
    Panel for viewing biomarkers
    """
    st.info('Coming soon!')

def view_spare() -> None:
    """
    Panel for viewing biomarkers
    """
    st.info('Coming soon!')

def view_surrealgan() -> None:
    """
    Panel for viewing biomarkers
    """
    st.info('Coming soon!')

#st.info(
st.markdown(
    """
    ### Explore Neuroimaging Chart
    """
)

tab = sac.tabs(
    items=[
        sac.TabsItem(label='Pipelines'),
        sac.TabsItem(label='View Sample Distributions'),
    ],
    size='lg',
    align='left'
)

sel_pipeline = 'dlmuse'

# Select pipeline
if tab == 'Pipelines':
    sel_pipeline = pipeline_overviews()
    
# Show output values for the selected pipeline
if tab == 'View Sample Distributions':
    if sel_pipeline == 'dlmuse':
        view_dlmuse()

    #elif psel == 1:
        #view_dlwmls()
        
    #elif psel == 2:
        #view_dlmuse_biomarkers()

    #elif psel == 3:
        #view_dlmuse_biomarkers()

    #elif psel == 4:
        #view_dlmuse_biomarkers()

    #elif psel == 5:
        #view_dlmuse_biomarkers()

    #elif psel == 6:
        #view_synthseg()

if st.session_state.mode == 'debug':
    utilses.disp_session_state()
