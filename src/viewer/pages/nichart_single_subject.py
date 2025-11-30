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
import utils.utils_io as utilio
import utils.utils_data_view as utildv
from utils.utils_styles import inject_global_css 
import gui.utils_navig as utilnav

from streamlit_image_select import image_select
import re
from utils.utils_logger import setup_logger

import streamlit_antd_components as sac

logger = setup_logger()
logger.debug("--- STARTING: Single Subject ---")

inject_global_css()

# Page config should be called for each page
#utilpg.config_page()
utilpg.set_global_style()

# Set data type
st.session_state.workflow = 'single_subject'

if 'instantiated' not in st.session_state or not st.session_state.instantiated:
    utilses.init_session_state()

st.session_state.subject_type = 'single'

st.markdown("<h5 style='text-align:center; color:#3a3a88;'>Single-Subject Analysis\n\n</h1>", unsafe_allow_html=True)

cols = st.columns([1,6,1])
with cols[1]:
    st.markdown(
        '''
        Welcome! This is where you can calculate neuroimaging chart values from a single subject's MRI scan(s) in a few simple actions:
        
        - **Data:** Upload image (Nifti, Dicom) and non-image (.csv) files required for analysis
        
        - **Pipelines:** Select processing/analysis pipeline to run on your data

        - **Results:** View/download results of the pipeline
        
        ''', unsafe_allow_html=True
    )

utilnav.main_navig(None, None, 'Data', 'pages/nichart_data.py')

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()



