import os
import shutil
from typing import Any

import numpy as np
import streamlit as st
import utils.utils_io as utilio
import utils.utils_session as utilses

def util_help_dialog(
    s_title: str,
    s_text: str,
    s_warning: str = ''
) -> None:
    @st.dialog(s_title)  # type:ignore
    def show_dialog():
        st.markdown(s_text)
        if s_warning != '':
            st.warning(s_warning)

    col1, col2 = st.columns([0.5, 0.1])
    with col2:
        if st.button(
            "Get help 🤔", key="key_btn_help_" + s_title, use_container_width=True
        ):
            show_dialog()

def util_help_workingdir() -> None:
    s_title="Working Directory"
    s_text= """
        - A NiChart pipeline executes a series of steps, with input/output files organized in a predefined folder structure.

        - Results for an **experiment** (a new analysis on a new dataset) are kept in a dedicated **working directory**.

        - Set an **"output path"** (desktop app only) and an **"experiment name"** to define the **working directory** for your analysis. You only need to set the working directory once.

        - The **experiment name** can be any identifier that describes your analysis or data; it does not need to match the input study or data folder name.

        - You can initiate a NiChart pipeline by selecting the **working directory** from a previously completed experiment.
    """
    s_warning="""
        ❗ On the cloud app, uploaded data and results of experiments are deleted in regular intervals. 
        
        The data from a previous experiment may not be available!
    """
    
    util_help_dialog(s_title, s_text, s_warning)

def util_help_indicoms() -> None:
    s_title = "DICOM Data"
    s_text = """
    - Upload or select the input DICOM folder containing all DICOM files. Nested folders are supported.

    - On the desktop app, a symbolic link named **"Dicoms"** will be created in the **working directory**, pointing to your input DICOM folder.

    - On the cloud platform, you can directly drag and drop your DICOM files or folders and they will be uploaded to the **"Dicoms"** folder within the **working directory**.

    - On the cloud, **we strongly recommend** compressing your DICOM data into a single ZIP archive before uploading. The system will automatically extract the contents of the ZIP file into the **"Dicoms"** folder upon upload.
    """
    
    util_help_dialog(s_title, s_text)

def util_help_dicom_detect() -> None:
    s_title = "DICOM Series"
    s_text = """
    - The system verifies all files within the DICOM folder.
    - Valid DICOM files are processed to extract the DICOM header information, which is used to identify and group images into their respective series
    - The DICOM field **"SeriesDesc"** is used to identify series
    """

    util_help_dialog(s_title, s_text)

def util_help_dicom_extract() -> None:
    s_title = "Nifti Conversion"
    s_text = """
    - The user specifies the desired modality and selects the associated series.
    - Selected series are converted into Nifti image format.
    - Nifti images are renamed with the following format: **{PatientID}_{StudyDate}_{modality}.nii.gz**
    """
    
    util_help_dialog(s_title, s_text)
