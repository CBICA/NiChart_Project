
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

import streamlit as st
from utils.nav import top_nav

print("--- RERUN: HOME PAGE STARTING ---") 

# Page config should be called for each page
utilpg.config_page()
utilpg.set_global_style()

# Inject custom CSS once
st.markdown(
    """
    <style>
    .centered-text {
        text-align: center;
        font-size: 40px;
        color: #53AB23;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="centered-text">Welcome to NiChart Project</p>', unsafe_allow_html=True)

sel_opt = sac.segmented(
    items=[
        sac.SegmentedItem(label='About NiChart'),
        sac.SegmentedItem(label='Single-Subject MRI Data'),
        sac.SegmentedItem(label='Multi-Subject MRI Dataset'),
        sac.SegmentedItem(label='No MRI/View Results'),
    ], label='', align='center', size='md', radius='md', color='indigo', divider=True
) 


sel_but = sac.buttons([
    sac.ButtonsItem(label='Go'),
], label='', align='center', index=None)

if sel_but == 'Go':

    print(f'Selected page {sel_opt}')

    if sel_opt == 'About NiChart':
        st.switch_page("pages/info.py")

    if sel_opt == 'Single Subject MRI':
        st.switch_page("pages/single_subject.py")

    if sel_opt == 'Multi-Subject MRI Dataset':
        st.switch_page("pages/multi_subject.py")

    if sel_opt == 'No MRI/View Results':
        st.switch_page("pages/no_user_mri.py")

