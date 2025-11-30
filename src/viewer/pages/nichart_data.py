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
import utils.utils_settings as utilset

import gui.utils_navig as utilnav
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


@st.dialog("Help Information", width="medium")
def my_help():
    st.write(
        """
        **Project Folder Help**
        - All processing steps are performed inside a project folder.
        - By default, NiChart will create and use a current project folder for you.
        - You may also create a new project folder using any name you choose.
        - If needed, you can reset the current project folder (this will remove all files inside it, but keep the folder itself), allowing you to start fresh.
        - You may also switch to an existing project folder.

        **Note:** If you are using the cloud version, stored files will be removed periodically, so previously used project folders might not remain available.
        """
    )


#################################
## Main

with st.container(horizontal=True, horizontal_alignment="center"):
    st.markdown("<h4 style=color:#3a3a88;'>Upload Data\n\n</h1>", unsafe_allow_html=True, width='content')

if st.session_state.workflow == 'ref_data':
    st.info('''
        You’ve selected the **Reference Data** workflow. This option doesn’t require data upload.
        - If you meant to analyze your data, please go back and choose a different workflow.
        - Otherwise, continue to the next step to explore the reference values.
        '''
    )

    # utilnav.main_navig()

else:
    upload_data()

    utilnav.main_navig(
        'Info', f'pages/nichart_{st.session_state.workflow}.py',
        'Pipelines', 'pages/nichart_pipelines.py',
        utilset.edit_settings,
        my_help
    )

#     sac.divider()
#
#     with st.container(horizontal=True, horizontal_alignment="center"):
#         b1 = st.button('', icon=':material/arrow_back:', help = 'Info')
#         b2 = st.button('', icon=':material/arrow_forward:', help = 'Pipeline')
#
#     if b1:
#         st.switch_page(f'pages/nichart_{st.session_state.workflow}.py')
#
#     if b2:
#         st.switch_page("pages/nichart_pipelines.py")
        

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()



