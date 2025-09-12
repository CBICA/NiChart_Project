import os
import shutil
import time
from typing import Any

import streamlit as st
import utils.utils_pages as utilpg
import utils.utils_io as utilio
import utils.utils_session as utilses
import utils.utils_data_view as utildv
import utils.utils_io as utilio
import utils.utils_session as utilss
from stqdm import stqdm
import pandas as pd
import numpy as np

import streamlit_antd_components as sac

# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()
utilpg.set_global_style()

if 'instantiated' not in st.session_state or not st.session_state.instantiated:
    utilses.init_session_state()

def select_project():
    """
    Panel for selecting a project
    """
    with st.container(border=True):
        out_dir = st.session_state.paths["out_dir"]
        sel_project = utilio.panel_select_project(out_dir, st.session_state.project)
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
    Panel for uploading project data
    """
    with st.container(border=True):    
        utilio.panel_load_data()
    
def panel_delete_data():
    with st.container(border=True):
        st.success(f'Project Name: {st.session_state.project}')
        proj_dir = st.session_state.paths['project']
        if st.button("Delete this project", help="This will permanently delete all data in this project and invalidate all associated caching."):
            st.warning("Are you sure you want to delete all data associated with this project? This cannot be undone.")
            if st.button("Confirm deletion"):
                shutil.rmtree(st.session_state.paths['project'])
                st.success(f"Project {st.session_state.project} was successfully deleted.")
                list_projects = utilio.get_subfolders(st.session_state.paths["out_dir"])
                if len(list_projects) > 0:
                    sel_project = list_projects[0]
                    utilss.update_project(sel_project)
                
    return
st.markdown(
    """
    ### Manage project data
    
    - Create a project folder for your dataset and upload input data for processing
    
    - Or, switch to an existing project

    """
)

tab = sac.tabs(
    items=[
        sac.TabsItem(label='Select Project Name'),
        sac.TabsItem(label='Upload Data'),
        sac.TabsItem(label='View Project Folder'),
        sac.TabsItem(label='Delete Data')
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

if tab == 'Delete Data':
    panel_delete_data()

# Show selections
utilses.disp_selections()

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()
