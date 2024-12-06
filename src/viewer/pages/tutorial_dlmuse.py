import os

import streamlit as st
import utils.utils_menu as utilmenu
import utils.utils_session as utilss

# Page config should be called for each page
utilss.config_page()

utilmenu.menu()

@st.dialog("Video tutorial")  # type:ignore
def show_video(f_video):
    video_file = open(f_video, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

st.markdown(
    """
    ### How to use this pipeline ❓
    - **Follow the Steps:** Proceed through the pipeline steps on the left.
    - **Data Flow:** The output of each step automatically becomes the input for the next.
    - **Flexibility:** You can skip steps if you already have the required data for subsequent steps.    
    """
)

st.markdown(
    """
    ### Required user data 🧠 
    - **MRI Scans:** One or more T1-weighted MRI scan(s) as input, provided in either DICOM or NIfti format.
    - **Demographics:** A CSV file containing essential demographic information.
    
        This file must include at least:
        - MRID: Unique identifier for the subject (scan timepoint)
        - Age: Age of the subject.
        - Sex: Sex of the subject (M/F).
        
        Optionally, include:
        - DX: Diagnosis (e.g., Alzheimer's Disease (AD), Control (CN)).
        - SITE: The site where the MRI scans were acquired (used to define scanner batches for harmonization)
        
    - **DLMUSE ROI Volumes (Optional):** If you have already segmented your MRI data using the DLMUSE method, you can provide the resulting ROI (Region of Interest) volumes in CSV format. This can speed up the process.
    """
)

st.markdown(
    """
    ### Show me an example 🤔
    - Example datasets are provided for your convenience. Please watch the videos below for a step-by-step guide.
    """
)

if st.button(':material/play_circle: DICOMs to Biomarkers (small n, complete pipeline)'):
    f_video=os.path.join(st.session_state.paths['root'], 'resources', 'videos', 'tutorial_smri_1.webm')
    show_video(f_video)

if st.button(':material/play_circle: ROIs to Biomarkers (larger n, pre-computed ROIs)'):
    f_video=os.path.join(st.session_state.paths['root'], 'resources', 'videos', 'tutorial_smri_1.webm')
    show_video(f_video)

st.markdown(
    """
    - Before applying the pipeline to your own data, we recommend replicating the provided examples.
    - Example datasets are provided in the folder "test_data". If you are using the desktop version, test_data folder is already in the default search path for file/folder selection.
    - If you are using the cloud version, you can download the test_data from "https://github.com/CBICA/NiChart_Project/test_data".
    """    
)

