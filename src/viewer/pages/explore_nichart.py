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
    # Select result type        
    list_res_type = ['Segmentation', 'Volumes']
    sel_res_type = st.pills(
        'Select output type',
        list_res_type,
        default = None,
        selection_mode = 'single',
        label_visibility = 'collapsed',
    )
    
    if sel_res_type == 'Segmentation':
        ulay = st.session_state.ref_data["t1"]
        olay = st.session_state.ref_data["dlmuse"]        
        utilmri.panel_view_seg(ulay, olay, 'muse')
        
    elif sel_res_type == 'Volumes':
        # df = pd.read_csv(
        #     '/home/guraylab/GitHub/gurayerus/NiChart_Project/resources/reference_data/centiles/dlmuse_centiles_CN.csv'
        #     #'/home/gurayerus/GitHub/gurayerus/NiChart_Project/resources/reference_data/centiles/dlmuse_centiles_CN.csv'
        # )
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

tab1, tab2 = st.tabs(
    ["Pipeline", "Output"]
)

# Select pipeline
with tab1:
    # Show a thumbnail image for each pipeline
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
    
    # Show description of the selected pipeline
    if psel >= 0 :
        view_description(list(pdict.values())[psel])
    
# Show output values for the selected pipeline
with tab2:
    if psel == 0:
        view_dlmuse()

    elif psel == 1:
        view_dlwmls()
        
    elif psel == 2:
        view_dlmuse_biomarkers()

    elif psel == 3:
        view_dlmuse_biomarkers()

    elif psel == 4:
        view_dlmuse_biomarkers()

    elif psel == 5:
        view_dlmuse_biomarkers()

    elif psel == 6:
        view_synthseg()

if st.session_state.mode == 'debug':
    if st.sidebar.button('Show Session State'):
        utilses.disp_session_state()
