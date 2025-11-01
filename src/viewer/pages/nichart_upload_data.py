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
logger.debug('Page: Upload Data')

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
        with st.popover("❓", width='content'):
            st.write(
                """
                **How to Use This Page**
                
                - **Upload Files:**
                
                  - Upload MRI scans in **NIfTI** (.nii / .nii.gz) or **DICOM** (either a folder of .dcm files or a single .zip archive).
                  - A **subject list** will be created automatically as MRI scans are added
                  - You may also upload non-imaging data (e.g., clinical variables) as a **CSV** containing an **MRID** column that matches the subject list.
                  
                - **Review & Edit Subject List:**
                
                  - View subject list; edit or add details needed for downstream analysis (e.g., MRID, age, sex).

                - **Review Project Folder:**

                  - View files stored in the project folder.
                  - Delete files if needed (e.g., to restart or replace data).
                  - Switch to a new or existing project folder.

                """
            )

def upload_data():

    # cols = st.columns([8,1,8,1,8])
    cols = st.columns([10,1,10])

    out_dir = os.path.join(
        st.session_state.paths['out_dir'], st.session_state['project']
    )
    
    with cols[0]:
        #with st.container(border=True):
        utilup.panel_upload_single_subject(out_dir)

    # with cols[2]:
    #     #with st.container(border=True):
    #     utilup.panel_edit_participants(
    #         os.path.join(out_dir, 'participants'),
    #         'participants.csv'
    #     )

    # with cols[4]:
    with cols[2]:
        #with st.container(border=True):
        in_dir = st.session_state.paths['project']
        utilup.panel_edit_participants(
            os.path.join(out_dir, 'participants'),
            'participants.csv'
        )
        utilup.panel_view_folder(out_dir)

        
    # Show selections
    #utilses.disp_selections()

#################################
## Main

data_type = st.session_state.data_type

with st.container(horizontal=True, horizontal_alignment="center"):
    st.markdown("<h4 style=color:#3a3a88;'>User Data\n\n</h1>", unsafe_allow_html=True, width='content')
    help_message(data_type)

upload_data()

sac.divider(key='_p0_div1')

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



