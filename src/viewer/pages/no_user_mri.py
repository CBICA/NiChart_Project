import streamlit as st
import os
import yaml
from pathlib import Path
from graphviz import Digraph
from collections import defaultdict
import utils.utils_pages as utilpg
import utils.utils_processes as utilprc
import utils.utils_session as utilses
from streamlit_image_select import image_select
import re
from utils.utils_logger import setup_logger

import streamlit_antd_components as sac

logger = setup_logger()
logger.debug('Page: Select Pipelines')

## Page config should be called for each page
#utilpg.config_page()
#utilpg.set_global_style()

if 'instantiated' not in st.session_state or not st.session_state.instantiated:
    utilses.init_session_state()

st.markdown(
    """
    ### No User MRI

    - View existing results

    """
)

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()



