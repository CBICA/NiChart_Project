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
import utils.utils_data_view as utildv
import gui.utils_pipelines as utilpipe
from utils.utils_styles import inject_global_css
import gui.utils_navig as utilnav
import utils.utils_settings as utilset

from streamlit_image_select import image_select
import re
from utils.utils_logger import setup_logger

import streamlit_antd_components as sac

logger = setup_logger()
logger.debug("--- STARTING: Run Pipelines ---")

inject_global_css()

# Page config should be called for each page
utilpg.config_page()
utilpg.set_global_style()

@st.dialog("Help Information", width="medium")
def my_help():
    st.write(
        """
        **How to Use This Page**

        - Select a pipeline
        - Run the pipeline
        - View progress
        """
    )


if 'instantiated' not in st.session_state or not st.session_state.instantiated:
    utilses.init_session_state()

utilpipe.panel_pipelines()

if st.session_state.workflow == 'Reference Data':
    utilnav.main_navig(
        'Home', 'pages/nichart_ref_data.py',
        'Results', 'pages/nichart_results.py',
    )

else:
    utilnav.main_navig(
        'Data', 'pages/nichart_data.py',
        'Results', 'pages/nichart_results.py',
        utilset.edit_settings, my_help
    )

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()



