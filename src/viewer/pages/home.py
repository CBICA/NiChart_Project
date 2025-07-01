import os
import numpy as np
import pandas as pd
import streamlit as st
import utils.utils_pages as utilpg
import utils.utils_misc as utilmisc
import utils.utils_plots as utilpl
import utils.utils_session as utilses
import utils.utils_mriview as utilmri
from streamlit_image_select import image_select
import logging
from stqdm import stqdm
from utils.utils_logger import setup_logger

import streamlit_antd_components as sac

# Page config should be called for each page
utilpg.config_page()
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
            f'NiChart is an {utilmisc.styled_text('open-source framework')} built specifically for deriving {utilmisc.styled_text('machine learning biomarkers')} from {utilmisc.styled_text('MRI imaging data')}', unsafe_allow_html=True
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
            """
            ##### Explore NiChart (No Data Upload Required):
            
            - `Visualize distributions` of imaging variables and biomarkers from the large `NiChart reference dataset`, processed through various pipelines.
            
            - This module is for `visualization of the output values` from processing pipelines and `does not require any user data`.

            ##### Analyze Your Own Data:

            - **<u>Upload Your Data: </u>** Navigate to the "Data" page to upload the files you wish to analyze.

            - **<u>Select Your Pipeline: </u>** Go to the "Pipelines" page and choose the analysis workflow you want to apply to your data.

            - **<u>View and/or Download Your Results: </u>** Once the pipeline has finished processing, your findings will be available to view or to download.


            """
            , unsafe_allow_html=True
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
    

st.markdown(
    """
    ### Welcome to NiChart Project!
    """
    , unsafe_allow_html=True
)

tab = sac.tabs(
    items=[
        sac.TabsItem(label='Overview'),
        sac.TabsItem(label='Quick Start'),
        sac.TabsItem(label='Links'),
        sac.TabsItem(label='Installation'),
    ],
    size='lg',
    align='left'
)


if tab == 'Overview':
    view_overview()

elif tab == 'Quick Start':
    view_quick_start()

elif tab == 'Links':
    view_links()

elif tab == 'Installation':
    view_installation()

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()
