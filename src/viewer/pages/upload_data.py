import os
import shutil
import time
from typing import Any

import streamlit as st
import utils.utils_pages as utilpg
import utils.utils_data_upload as utildata
import utils.utils_session as utilses
import utils.utils_data_view as utildv
from stqdm import stqdm
import pandas as pd
import numpy as np

import streamlit_antd_components as sac

# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()
utilpg.set_global_style()

def select_project():
    """
    Panel for selecting a project
    """
    with st.container(border=True):
        out_dir = st.session_state.paths["out_dir"]
        sel_project = utildata.select_project(out_dir, st.session_state.project)
        st.success(f'Project Name: {st.session_state.project}')
    
def view_project_folder():
    """
    Panel for viewing files in a project folder
    """
    with st.container(border=True):
        in_dir = st.session_state.paths['project']
        utildv.data_overview(in_dir)

def upload_data():
    """
    Panel for viewing files in a project folder
    """
    with st.container(border=True):    
        utildata.panel_load_data()
    
st.markdown(
    """
    ### User Data
    """
)

tab = sac.tabs(
    items=[
        sac.TabsItem(label='Select Project Name'),
        sac.TabsItem(label='Upload Data'),
        sac.TabsItem(label='View Project Folder'),
    ],
    size='lg',
    align='left'
)

if tab == 'Select Project Name':
    select_project()

if tab == 'Upload Data':
    upload_data()

if tab == 'View Project Folder':
    view_project_folder()

# Show selections
utilses.disp_selections()

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()
