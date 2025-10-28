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
from streamlit_image_select import image_select
import re
from utils.utils_logger import setup_logger

import streamlit_antd_components as sac

logger = setup_logger()
logger.debug('Page: NiChart Reference Data')

# Page config should be called for each page
utilpg.config_page()
utilpg.set_global_style()


if 'instantiated' not in st.session_state or not st.session_state.instantiated:
    utilses.init_session_state()

def view_overview():
    with st.container(border=True):
        st.markdown(
            f'NiChart is a {utilmisc.styled_text('free, open-source framework')} built specifically for deriving {utilmisc.styled_text('machine learning biomarkers')} from {utilmisc.styled_text('MRI imaging data')}', unsafe_allow_html=True
        )
        st.image("../resources/nichart1.png", width=300)
        st.markdown(
            f'- NiChart platform offers tools for {utilmisc.styled_text('image processing')} and {utilmisc.styled_text('data analysis')}', unsafe_allow_html=True
        )
        st.markdown(
            f'- Users can extract {utilmisc.styled_text('imaging phenotypes')} and {utilmisc.styled_text('machine learning (ML) indices')} of disease and aging', unsafe_allow_html=True
        )
        st.markdown(
            f'- Pre-trained {utilmisc.styled_text('ML models')} allow users to quantify complex brain changes and compare results against {utilmisc.styled_text('normative and disease-specific reference ranges')}', unsafe_allow_html=True
        )

def upload_data():
    st.info('Work in progress!')

def select_pipeline():
    st.info('Work in progress!')

def view_results():
    st.info('Work in progress!')

def download_results():
    st.info('Work in progress!')

st.markdown(
    """
    ### NiChart Reference Distributions
    """
)

sel = sac.tabs([
    sac.TabsItem(label='Overview'),
    sac.TabsItem(label='Select Pipeline'),
    sac.TabsItem(label='View Results'),
    sac.TabsItem(label='Download Results'),
    sac.TabsItem(label='Go Back Home'),
], align='center',  size='xl', color='grape')

if sel == 'Overview':
    view_overview()
    
if sel == 'Select Pipeline':
    select_pipeline()
    
if sel == 'View Results':
    view_results()

if sel == 'Download Results':
    download_results()

if sel == 'Go Back Home':
    st.switch_page("pages/nichart_home.py")


# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()



