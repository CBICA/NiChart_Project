import streamlit as st
import os
import yaml
from pathlib import Path
from graphviz import Digraph
from collections import defaultdict
import utils.utils_misc as utilmisc
import utils.utils_pages as utilpg
import utils.utils_processes as utilprc
import utils.utils_session as utilses
import utils.utils_io as utilio
import utils.utils_upload as utilup
import utils.utils_pipelines as utilpipe
import utils.utils_data_view as utildv
from utils.utils_styles import inject_global_css 
import utils.utils_navig as utilnav

from streamlit_image_select import image_select
import re
from utils.utils_logger import setup_logger

import streamlit_antd_components as sac

logger = setup_logger()
logger.debug('Page: Select Pipelines')

inject_global_css()

# Page config should be called for each page
#utilpg.config_page()
utilpg.set_global_style()

###############################
# Set session state variables for the reference data workflow
st.session_state.workflow = 'multi_subject'
st.session_state.subject_type = 'multi'

st.session_state.paths['curr_data'] = st.session_state.paths['prj_dir'] 

###############################

if 'instantiated' not in st.session_state or not st.session_state.instantiated:
    utilses.init_session_state()

with st.container(
    horizontal=True, horizontal_alignment="center", width='stretch'
):
    tab1, tab2, tab3 = st.tabs(
        ["Overview", "Data", "Processing"],
        on_change='rerun',
    )

## Info
if tab1.open:
    with tab1:
        # st.markdown("<h5 style='text-align:center; color:#3a3a88;'>Multi-Subject Dataset Analysis\n\n</h1>", unsafe_allow_html=True)

        st.markdown("<h5 style=color:#3a3a88;'>Multi-Subject Dataset Analysis\n\n</h1>", unsafe_allow_html=True)

        cols = st.columns([1,6,1])
        with cols[1]:
            st.markdown(
                '''
                Compute neuroimaging chart values for multi-subject MRI datasets with just a few steps:
                
                - **Data:** Upload image (NIfTI) and non-image (.csv) files required for analysis
                
                - **Processing:** Select processing/analysis pipeline to run on your data

                    ''', unsafe_allow_html=True
            )

## Data Upload
if tab2.open:
    with tab2:

        # with st.container(horizontal=True, horizontal_alignment="center"):
        with st.container(horizontal=True):
            st.markdown("<h5 style='color:#3a3a88;'>Data Upload</h5>", unsafe_allow_html=True, width='content')

    utilup.panel_data()

## Pipelines
if tab3.open:
    with tab3:
        # with st.container(horizontal=True, horizontal_alignment="center"):
        with st.container(horizontal=True):
            st.markdown("<h5 style='color:#3a3a88;'>Select and Run a Pipeline</h5>", unsafe_allow_html=True, width='content')

        utilpipe.panel_pipelines()

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()



