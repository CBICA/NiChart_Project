import os
import shutil
import time
from typing import Any

import streamlit as st
import pandas as pd
import utils.utils_dicom as utildcm
import utils.utils_doc as utildoc
import utils.utils_io as utilio
import utils.utils_nifti as utilni
import utils.utils_pages as utilpg
import utils.utils_session as utilss
import utils.utils_data_upload as utildata
import utils.utils_st as utilst
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
    curr_project = st.session_state.navig['project']            

    sel_project = utildata.select_project(out_dir, curr_project)
    
    if sel_project is not None:        
        st.success(f'Selected Project: {sel_project}')
    
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
