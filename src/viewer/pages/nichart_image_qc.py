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
from utils.utils_styles import inject_global_css
import utils.utils_toolloader as utiltl
import utils.utils_stlogbox as stlogbox

from streamlit_image_select import image_select
from stqdm import stqdm
import re
from utils.utils_logger import setup_logger

import streamlit_antd_components as sac


logger = setup_logger()
logger.debug("--- STARTING: Run Pipelines ---")

inject_global_css()

# Page config should be called for each page
#utilpg.config_page()
utilpg.set_global_style()


if 'instantiated' not in st.session_state or not st.session_state.instantiated:
    utilses.init_session_state()

def help_message():

    with st.popover("❓", width='content'):
        st.write(
            """
            **How to Use This Page**

            Here you can run quality checks on your images and detect any issues that would prevent our pipelines from executing.

            We'll check each image in the selected project and give the associated errors below.
            """
        )    

with st.container(horizontal=True, horizontal_alignment="center"):
    st.markdown("<h4 style=color:#3a3a88;'>Image Quality Checker\n\n</h1>", unsafe_allow_html=True, width='content')
    help_message()

