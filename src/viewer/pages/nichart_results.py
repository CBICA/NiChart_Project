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

with st.container(horizontal=False, horizontal_alignment="center"):
    st.markdown("<h4 style=color:#3a3a88;'>View Results\n\n</h1>", unsafe_allow_html=True, width='content')
    st.markdown('Please select a viewing option from the sidebar!', width="content")


utilres.panel_results()

if st.session_state.workflow == 'ref_data':
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


