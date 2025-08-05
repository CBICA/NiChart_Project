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
from streamlit_image_select import image_select
import logging
from stqdm import stqdm
from utils.utils_logger import setup_logger

import streamlit_antd_components as sac

utilpg.show_menu()
utilpg.add_sidebar_options()
utilpg.set_global_style()

logger = setup_logger()

logger.debug('Start of Home Screen!')

#def styled_text(text):
    #return f'<span style="color:darkgreen; font-weight:bold;">{text}</span>'

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

def view_quick_start():
    with st.container(border=True):
        st.markdown(
            f'###### Explore NiChart {utilmisc.styled_text('(No Data Upload Required)')}', unsafe_allow_html=True
        )
        st.markdown(
            f'- Visualize distributions of imaging variables and biomarkers (pre-computed using NiChart reference data)', unsafe_allow_html=True
        )

        st.markdown(
            '''
            ###### Apply NiChart to Your Data
            
            - Select Your Pipeline
            
            - Upload Your Data
            
            - Run pipeline
            
            - View / Download Your Results
            ''', unsafe_allow_html=True
        )

def view_links():
    with st.container(border=True):
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
            st.switch_page("survey.py")

def view_installation():
    with st.container(border=True):
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
if 'instantiated' not in st.session_state or not st.session_state.instantiated:
    utilses.init_session_state()

utils_alerts.render_alert()

st.markdown(
    """
    ### Welcome to NiChart Project!!!
    """
    , unsafe_allow_html=True
)

tab = sac.tabs(
    items=[
        sac.TabsItem(label='Overview'),
        sac.TabsItem(label='Quick Start'),
        sac.TabsItem(label='Links'),
        sac.TabsItem(label='Installation'),
        #sac.TabsItem(label='Test'),
    ],
    size='lg',
    align='left'
)


if tab == 'Overview':
    view_overview()

if tab == 'Quick Start':
    view_quick_start()

if tab == 'Links':
    view_links()

if tab == 'Installation':
    view_installation()

#if tab == 'Test':
    #view_test()

# Show selections
utilses.disp_selections()
    
# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()
