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
import utils.utils_plots as utilpl
import utils.utils_mriview as utilmri
import utils.utils_data_view as utildv
import gui.utils_results as utilres
from utils.utils_styles import inject_global_css 
import pandas as pd
import gui.utils_navig as utilnav
import utils.utils_settings as utilset

from streamlit_image_select import image_select
import re
from utils.utils_logger import setup_logger
import utils.utils_settings as utilset

import streamlit_antd_components as sac

logger = setup_logger()
logger.debug('Page: Results')

inject_global_css()

# Page config should be called for each page
#utilpg.config_page()
utilpg.set_global_style()

@st.dialog("Help Information", width="medium")
def my_help():
    st.write(
        """
        **How to Use This Page**

        - See results
        """
    )

if 'instantiated' not in st.session_state or not st.session_state.instantiated:
    utilses.init_session_state()

utilres.panel_results()

if st.session_state.workflow == 'Reference Data':
    utilnav.main_navig(
        'Info', 'pages/nichart_ref_data.py',
        'Home', 'pages/nichart_home.py',
        utilset.edit_settings, my_help
    )

else:
    utilnav.main_navig(
        'Pipelines', 'pages/nichart_pipelines.py',
        'Home', 'pages/nichart_home.py',
        utilset.edit_settings, my_help
    )

# with st.container(horizontal=True, horizontal_alignment="center"):
#     b1 = st.button('', icon=':material/arrow_back:', help = 'Pipeline')
#     b2 = st.button('', icon=':material/arrow_forward:', help = 'Home')
#     b3 = st.button('', icon=':material/settings:')
#
# if b1:
#     st.switch_page("pages/nichart_pipelines.py")
#
# if b2:
#     st.switch_page("pages/nichart_home.py")
#
# if b3:
#     utilset.edit_settings()
    
# # Show session state vars
# if st.session_state.mode == 'debug':
#     with st.sidebar:
#         utilses.disp_session_state()



