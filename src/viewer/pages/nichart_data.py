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
@st.dialog("Help Information", width="medium")
def my_help():
    tab1, tab2, tab3 = st.tabs(["Project Folder", "Upload Files", "Review Files"])

    with tab1:
        st.write(
            """
            - All processing steps are performed inside a project folder.
            - By default, NiChart will create and use a current project folder for you.
            - You may also create a new project folder using any name you choose.
            - If needed, you can reset the current project folder (this will remove all files inside it, but keep the folder itself), allowing you to start fresh.
            - You may also switch to an existing project folder.

            **Note:** If you are using the cloud version, stored files will be removed periodically, so previously used project folders might not remain available.
            """
        )

    if st.session_state.workflow == 'single_subject':
        with tab2:
            st.write(
                """
                - You may upload MRI scans in any of the following formats:
                    - **NIfTI:** .nii or .nii.gz
                    - **DICOM (compressed):** a single .zip file containing the DICOM series
                    - **DICOM (individual files):** multiple .dcm files

                    *(Note: uploading a folder directly is not currently supported)*

                - If you have multiple imaging modalities (e.g., T1, FLAIR), upload them one at a time.

                - Once uploaded, NiChart will automatically:
                    - Organize the files into the standard input structure
                    - Create a subject list based on the uploaded MRI data

                - You may open and edit the subject list (e.g., to add age, sex, or other metadata needed for analysis).

                - You can also upload non-imaging data (e.g., clinical or cognitive measures) as a CSV file.

                - The CSV must include an MRID column with values that match the subject IDs in the subject list, so the data can be merged correctly.
                """
            )

    if st.session_state.workflow == 'multi_subject':
        with tab2:
            st.write(
                """
                - MRI Scans (NIfTI format):
                    - Upload one or more .nii / .nii.gz files or
                    - Upload a .zip file containing multiple NIfTI files

                    *(Note: uploading a folder directly is not currently supported)*

                - If your dataset includes multiple imaging modalities (e.g., T1, FLAIR), upload each modality separately.

                - Wait for the whole batch to upload before proceeding. (You'll see a few upload bars -- use the arrows to scroll through the batch.)

                - For many pipelines, a participant CSV is required, containing at least one column:

                    **MRID** → subject ID that matches the scan filenames

                - Filename Format Requirement:

                    {MRID}_common_suffix.nii.gz

                    Example: SUB001_T1.nii.gz

                - After upload, NiChart will automatically:

                    - Organize scans into the standard input directory structure

                    - Check consistency between the participants CSV and the uploaded scans

                - You may view and edit participants CSV after upload.

                - Optional: Upload non-imaging data (e.g., clinical or cognitive variables) as an additional CSV (should include the MRID column).
                """
            )

    with tab3:
        st.write(
            """
            - View files stored in the project folder.

            - Click on a file name to:

                - View a scan (.nii.gz, .nii)

                - View/edit a list (.csv)
            """
        )

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

    utilnav.main_navig(
        'Home', 'pages/nichart_home.py',
        'Results', 'pages/nichart_results.py',
    )

else:
    upload_data()

    utilnav.main_navig(
        'Info', f'pages/nichart_{st.session_state.workflow}.py',
        'Pipelines', 'pages/nichart_pipelines.py',
        utilset.edit_settings, my_help
    )

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()



