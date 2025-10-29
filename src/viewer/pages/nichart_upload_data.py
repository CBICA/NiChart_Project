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

from streamlit_image_select import image_select
import re
from utils.utils_logger import setup_logger

import streamlit_antd_components as sac

logger = setup_logger()
logger.debug('Page: Select Pipelines')

inject_global_css()

# Page config should be called for each page
utilpg.config_page()
utilpg.set_global_style()


if 'instantiated' not in st.session_state or not st.session_state.instantiated:
    utilses.init_session_state()

def upload_data():

    with st.container(border=True):
        col1, col2, col3 = st.columns([10,1,10])
    
        with col1:
            utilio.panel_load_data()

        with col3:
            in_dir = st.session_state.paths['project']
            utildv.data_overview(in_dir)

        
    # Show selections
    #utilses.disp_selections()
     
st.markdown("<h5 style='text-align:center; color:#3a3a88;'>Data Upload\n\n</h1>", unsafe_allow_html=True)

upload_data()

sel_but = sac.chip(
    [
        sac.ChipItem(label = '', icon='arrow-left', disabled=False),
        sac.ChipItem(label = '', icon='arrow-right', disabled=False)
    ],
    label='', align='center', color='#aaeeaa', size='xl', return_index=True
)
    
if sel_but == 0:
    st.switch_page("pages/nichart_single_subject.py")

if sel_but == 1:
    st.switch_page("pages/nichart_run_pipeline.py")

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()



