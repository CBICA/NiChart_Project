import os
import shutil
import time
from typing import Any

import streamlit as st
import utils.utils_pages as utilpg
import utils.utils_data_upload as utildata
import utils.utils_session as utilses
from stqdm import stqdm
import pandas as pd
import numpy as np

import streamlit_antd_components as sac

# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()

def select_project():
    """
    Panel for selecting a project
    """
    out_dir = st.session_state.paths["out_dir"]
    sel_project = utildata.select_project(out_dir, st.session_state.project)
    st.success(f'Project Name: {st.session_state.project}')
    
def view_project_folder():
    """
    Panel for viewing files in a project folder
    """
    utildata.disp_folder_tree(st.session_state.paths['project'])
    
def upload_data():
    """
    Panel for viewing files in a project folder
    """
    utildata.panel_load_data()
    
st.markdown(
    """
    ### User Data
    """
)

tab = sac.segmented(
    items=[
        sac.SegmentedItem(label='Select Project Name'),
        sac.SegmentedItem(label='View Project Folder'),
        sac.SegmentedItem(label='Upload Data')
    ],
    size='sm',
    radius='lg',
    align='left'
)

if tab == 'Select Project Name':
    select_project()

elif tab == 'View Project Folder':
    view_project_folder()

elif tab == 'Upload Data':
    upload_data()

# Show session state vars
if st.session_state.mode == 'debug':
    if st.sidebar.button('Show Session State'):
        utilses.disp_session_state()
