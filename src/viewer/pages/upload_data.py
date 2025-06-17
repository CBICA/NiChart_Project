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
    #utildata.folder_viewer(st.session_state.paths['project'])
    
def upload_data():
    list_opt = [
        "Image Data",
        "Covariate File",
    ]
    sel_project = st.pills(
        "Select Task", list_opt, selection_mode="single", label_visibility="collapsed"
    )

    if sel_project == "Image Data":
        #list_opt_img = ["NIfTI", "DICOM", "BIDS", "PACS Server"]
        list_opt_img = ["NIfTI", "DICOM"]
        sel_project_img = st.pills(
            "Select Img Task",
            list_opt_img,
            selection_mode="single",
            label_visibility="collapsed",
            default=None,
            key='_sel_project_img'
        )

        if sel_project_img == "NIfTI":
            with st.container(border=True):
                st.markdown(
                    """
                    ***NIfTI Images***
                    - Upload NIfTI images
                    """
                )
                panel_nifti()

        elif sel_project_img == "DICOM":
            with st.container(border=True):
                st.markdown(
                    """
                    ***DICOM Files***
                    
                    - Upload a folder containing raw DICOM files
                    - DICOM files will be converted to NIfTI scans
                    """
                )
                panel_dicoms()
            
        elif sel_project_img == "BIDS Data":
            with st.container(border=True):
                st.markdown(
                    """
                    ***BIDS Format***
                    - Load a dataset structured according to the ***:red[BIDS standard](https://bids.neuroimaging.io)***, where all imaging modalities and metadata are organized in a single directory.
                    - This is the easiest option if your data is already standardized.
                    """
                )
                st.warning('Work in progress ...')
                

        elif sel_project_img == "Connect to PACS Server":
            with st.container(border=True):
                st.markdown(
                    """
                    ***Connect to PACS Server***
                    - Query and fetch imaging data directly from a hospital PACS server using DICOM networking.
                    - Requires PACS credentials and access permissions.
                    """
                )
                st.warning('Work in progress ...')

    elif sel_project == "Covariate File":
        with st.container(border=True):
            st.markdown(
                """
                ***Covariate File***
                - Upload a ***:red[csv file with covariate info]*** (Age, Sex, DX, etc.)
                """
            )
            panel_in_covars()
    
st.markdown(
    """
    ### User Data
    """
)

tab1, tab2, tab3 = st.tabs(
    ["Select Project", "View Project Folder", "Upload Data"]
)

with tab1:
    select_project()

with tab2:
    view_project_folder()

with tab3:
    upload_data()
