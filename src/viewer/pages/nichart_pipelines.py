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


if 'instantiated' not in st.session_state or not st.session_state.instantiated:
    utilses.init_session_state()

utilpipe.panel_pipelines()

sac.divider()

with st.container(horizontal=True, horizontal_alignment="center"):
    if st.session_state.workflow == 'Reference Data':
        #b1 = st.button('', icon=':material/arrow_back:', help = 'Pipeline')
        b2 = st.button('', icon=':material/arrow_forward:', help = 'Results')
        if b2:
            st.switch_page("pages/nichart_results.py")

    else:
        b1 = st.button('', icon=':material/arrow_back:', help = 'Data')
        b2 = st.button('', icon=':material/arrow_forward:', help = 'Results')
        if b1:
            st.switch_page("pages/nichart_data.py")

        if b2:
            st.switch_page("pages/nichart_results.py")
        

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()



