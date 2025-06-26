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

tab1, tab2, tab3 = st.tabs(
    ["Select Project Name", "View Project Folder", "Upload Data"]
)

with tab1:
    select_project()

with tab2:
    view_project_folder()

with tab3:
    upload_data()

# Show session state vars
if st.session_state.mode == 'debug':
    if st.sidebar.button('Show Session State'):
        utilses.disp_session_state()
