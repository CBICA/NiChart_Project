
import utils.utils_pages as utilpg
# Page config should be called for each page
utilpg.config_page()

import os
import numpy as np
import pandas as pd
import streamlit as st
import utils.utils_misc as utilmisc
import utils.utils_plots as utilpl
import utils.utils_session as utilses
import utils.utils_mriview as utilmri
import utils.utils_alerts as utils_alerts
import utils.utils_survey as utils_survey
from streamlit_image_select import image_select
import logging
from stqdm import stqdm
from utils.utils_logger import setup_logger

import streamlit_antd_components as sac

utilpg.config_page()
utilpg.set_global_style()

def view_overview():
    #with st.container(border=True):
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

def view_quick_start():
    #with st.container(border=True):

    st.markdown(
        '''
        You can begin using NiChart by choosing an option on the Home page and following the guided steps:
        
        - **Analyze Single-Subject MRI Data**:
        
        Create a neuroimaging chart for an individual subject. Process their MRI scans to compute imaging features, apply pre-trained machine learning models to derive biomarkers, and visualize results against reference population distributions.
        
        - **Process Multiple-Subject Dataset**: 
        
        Run the NiChart processing pipeline on a group or study dataset to generate imaging biomarkers and perform large-scale or comparative analyses.
        
        - **Explore NiChart**:
        
        Browse and visualize population-level distributions of precomputed imaging features and biomarkers from the NiChart reference dataset.
        '''
    )

def view_links():
    #with st.container(border=True):
    st.markdown(
        """
        - Check out [NiChart Web page](https://neuroimagingchart.com)
        - Visit [NiChart GitHub](https://github.com/CBICA/NiChart_Project)
        - Jump into [our documentation](https://cbica.github.io/NiChart_Project)
        - Ask a question in [our community discussions](https://github.com/CBICA/NiChart_Project/discussions)
        """
        , unsafe_allow_html=True
    )
    st.markdown("And fill out our 1-minute user demographics survey to gain **permanent, free** access to NiChart Cloud!")
    take_survey = st.button("Take Survey")
    if take_survey:
        st.switch_page("pages/survey.py")

def view_installation():
    #with st.container(border=True):
    #with st.expander(label='Installation'):
    st.markdown(
        """
        - You can install NiChart Project desktop
        ```
        pip install NiChart_Project
        ```

        - Run the application
        ```
        cd src/viewer
        streamlit run NiChartProject.py
        ```

        - Alternatively, the cloud app can be launched at
        ```
        https://cloud.neuroimagingchart.com
        ```
        """
        , unsafe_allow_html=True
    )

st.markdown("<h5 style='text-align:center; color:#3a3a88;'>NiChart: Neuroimaging Chart\n\n</h1>", unsafe_allow_html=True)

sel = sac.tabs([
    sac.TabsItem(label='Overview'),
    sac.TabsItem(label='Quick Start'),
    sac.TabsItem(label='Links'),
    sac.TabsItem(label='Installation'),
    sac.TabsItem(label="Start Using NiChart"),
], align='center',  size='xl', color='grape')

if sel == 'Overview':
    view_overview()
    
if sel == 'Quick Start':
    view_quick_start()

if sel == 'Links':
    view_links()
    
if sel == 'Installation':
    view_installation()

if sel == 'Start Using NiChart':
    st.switch_page("pages/nichart_home.py")
    
    



