import glob
import os
import time
from typing import Optional

import streamlit as st
import utils.utils_st as utilst

title_exp = "Experiment"
def_exp = """
    - A NiChart pipeline executes a series of steps, with input/output files organized in a predefined folder structure.
    - Results for an **experiment** (a new analysis on a new dataset) are kept in a dedicated **working directory**.
    - The **experiment name** can be any identifier that describes your analysis or data; it does not need to match the input study or data folder name.
    - You can initiate a NiChart pipeline by selecting the **working directory** from a previously completed experiment.
"""

title_dicoms = "DICOM Data"
def_dicoms = """
- Upload or select the input DICOM folder containing all DICOM files. Nested folders are supported.

- On the desktop app, a symbolic link named **"Dicoms"** will be created in the **working directory**, pointing to your input DICOM folder.

- On the cloud platform, you can directly drag and drop your DICOM files or folders and they will be uploaded to the **"Dicoms"** folder within the **working directory**.

- On the cloud, **we strongly recommend** compressing your DICOM data into a single ZIP archive before uploading. The system will automatically extract the contents of the ZIP file into the **"Dicoms"** folder upon upload.
"""

title_dicoms_detect = "DICOM Series"
def_dicoms_detect = """
- The system verifies all files within the DICOM folder.
- Valid DICOM files are processed to extract the DICOM header information, which is used to identify and group images into their respective series
- The DICOM field **"SeriesDesc"** is used to identify series
"""

title_dicoms_extract = "Nifti Conversion"
def_dicoms_extract = """
- The user specifies the desired modality and selects the associated series.
- Selected series are converted into Nifti image format.
- Nifti images are renamed with the following format: **{PatientID}_{StudyDate}_{modality}.nii.gz**
"""
