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
import utils.utils_data_view as utildv
from utils.utils_styles import inject_global_css 

from streamlit_image_select import image_select
import re
from utils.utils_logger import setup_logger

import streamlit_antd_components as sac

logger = setup_logger()
logger.debug('Page: Select Pipelines')

inject_global_css()

# Page config should be called for each page
utilpg.config_page()
utilpg.set_global_style()


if 'instantiated' not in st.session_state or not st.session_state.instantiated:
    utilses.init_session_state()

#################################
## Function definitions
def help_message(data_type):
    
    if data_type == 'reference_data':
        st.warning('No data upload for reference data viewer!')
        return
    
    if data_type == 'multi_subject':
        st.warning('Work in progress!')
        return

    if data_type == 'single_subject':
        with st.popover("💡"):
            st.write(
                """
                
                **How to Use This Page**
                
                - **Left Panel:** Upload your data files
                
                  - Nifti (.nii, .nii.gz), compressed Dicom files (.zip), or data file (.csv)
                
                  - Dicom data will be extracted automatically to Nifti
                  
                
                - **Middle Panel:** View contents of the project folder, as data is uploaded
                
                  - Project data is kept in a default folder (user_default)
                  
                  - Users can delete files inside the project folder, create a new project folder and switch to an existing folder
                
                - **Right Panel:** View subject list
                
                  - Subject list is necessary for all pipelines
                  
                  - It's created automatically during data upload
                  
                  - Users can upload their own file or edit the subject file

                """
            )

def upload_data():

    cols = st.columns([1,1,1])

    with cols[0]:
        with st.container(border=True):
            utilio.panel_load_data()

    with cols[1]:
        with st.container(border=True):
            in_dir = st.session_state.paths['project']
            utildv.data_overview(in_dir)

    with cols[2]:
        with st.container(border=True):
            in_dir = st.session_state.paths['project']
            utildv.view_subj_list(in_dir)

        
    # Show selections
    #utilses.disp_selections()

#################################
## Main

data_type = st.session_state.data_type

st.markdown("<h4 style='text-align:center; color:#3a3a88;'>Data Upload\n\n</h1>", unsafe_allow_html=True)

help_message(data_type)

upload_data()

sel_but = sac.chip(
    [
        sac.ChipItem(label = '', icon='arrow-left', disabled=False),
        sac.ChipItem(label = '', icon='arrow-right', disabled=False)
    ],
    label='', align='center', color='#aaeeaa', size='xl', return_index=True
)
    
if sel_but == 0:
    st.switch_page("pages/nichart_single_subject.py")

if sel_but == 1:
    st.switch_page("pages/nichart_run_pipeline.py")

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()



