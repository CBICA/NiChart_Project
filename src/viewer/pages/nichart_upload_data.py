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
import utils.utils_upload as utilup
import utils.utils_data_view as utildv
from utils.utils_styles import inject_global_css 

from streamlit_image_select import image_select
import re
from utils.utils_logger import setup_logger

import streamlit_antd_components as sac

logger = setup_logger()
logger.debug("--- STARTING: Upload Data ---")

inject_global_css()

# Page config should be called for each page
#utilpg.config_page()
utilpg.set_global_style()

if 'instantiated' not in st.session_state or not st.session_state.instantiated:
    utilses.init_session_state()

#################################
## Function definitions
def upload_data():

    cols = st.columns([6,1,10,1,10])

    with cols[0]:
        utilup.panel_project_folder()

    with cols[2]:
        if st.session_state.workflow == 'single_subject':
            utilup.panel_upload_single_subject()
        if st.session_state.workflow == 'multi_subject':
            utilup.panel_upload_multi_subject()

    with cols[4]:
        utilup.panel_view_files()
        
    # Show selections
    #utilses.disp_selections()

#################################
## Main

with st.container(horizontal=True, horizontal_alignment='center'):
    st.markdown("<h4 style=color:#3a3a88;'>Upload Data\n\n</h1>", unsafe_allow_html=True, width='content')

if st.session_state.workflow == 'ref_data':
    st.warning('No data upload required for reference data! Please skip to next step')

else:
    upload_data()

sac.divider()

sel_but = sac.chip(
    [
        sac.ChipItem(label = 'Back to Intro', icon='arrow-left', disabled=False),
        sac.ChipItem(label = 'Select and Run Pipeline', icon='arrow-right', disabled=False)
    ],
    label='', align='center', color='#aaeeaa', size='xl', return_index=True
)
    
if sel_but == 0:
    if st.session_state.subject_type == 'single':
        st.switch_page("pages/nichart_single_subject.py")
    if st.session_state.subject_type == 'multi':
        st.switch_page("pages/nichart_multi_subject.py")
    else: # Fallback
        st.switch_page(f'pages/nichart_{st.session_state.workflow}.py')

if sel_but == 1:
    if st.session_state.workflow == 'ref_data':
        st.switch_page("pages/nichart_view_results.py")

    else:
        st.switch_page("pages/nichart_run_pipeline.py")

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()



