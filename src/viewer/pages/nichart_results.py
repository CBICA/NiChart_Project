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

from streamlit_image_select import image_select
import re
from utils.utils_logger import setup_logger
import utils.utils_settings as utilset

import streamlit_antd_components as sac

logger = setup_logger()
logger.debug('Page: Select Pipelines')

inject_global_css()

# Page config should be called for each page
utilpg.config_page()
utilpg.set_global_style()

if 'instantiated' not in st.session_state or not st.session_state.instantiated:
    utilses.init_session_state()
     

## Show setting button
#utilset.settings_button()

sac.divider(key='_p0_div1')

st.markdown("<h4 style='text-align:center; color:#3a3a88;'>Results\n\n</h1>", unsafe_allow_html=True)


if st.session_state.workflow == 'ref_data':
    utilres.panel_ref_data()

else:
    #sel_container = st.sidebar()
    layout = st.sidebar if st.session_state.layout == "Sidebar" else st.container(border=False)
    utilres.panel_user_data()

sac.divider(key='_p0_div2')

sel_but = sac.chip(
    [
        sac.ChipItem(label = '', icon='arrow-left', disabled=False),
        sac.ChipItem(label = '', icon='square', disabled=False),
        sac.ChipItem(label = '', icon='house', disabled=False),
        sac.ChipItem(label = '', icon='gear', disabled=False),
    ],
    key='_chip_navig', label='', align='center', color='#aaeeaa', size='xl',
    multiple=False,
    return_index=True, index=1
)
if sel_but == 0:
    st.switch_page("pages/nichart_pipelines.py")

if sel_but == 2:
    st.switch_page("pages/nichart_home.py")
    
if sel_but == 3:
#if st.button('Settings'):
    utilset.edit_settings()

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()



