
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
        font-size: 80px;
        color: #53AB23;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#st.markdown('<h1 class="centered-text">Welcome to NiChart Project</p>', unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center; color:#5e5fad;'>Welcome to NiChart Project\n\n</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align:center; color:#3a3a88;'>What would you like to explore today?\n\n</h1>", unsafe_allow_html=True)

sel = sac.chip(
    items=[
        sac.ChipItem(label='What is NiChart?'),
        sac.ChipItem(label='Analyze Single Subject MRI Data'),
        sac.ChipItem(label='Analyze a Group of Scans'),
        sac.ChipItem(label='Explore Results Only (No MRI Needed!)'),
    ], label='', align='center', size='lg', radius='lg', direction='vertical', color='cyan'
) 
flag_disabled = sel is None

sel_but = sac.chip(
    [sac.ChipItem(label='Go!', disabled=flag_disabled)],
    label='', align='center', color='#aaeeaa'
)
    
if sel_but == 'Go!':
    print(f'Selected page {sel}')

    if sel == 'What is NiChart?':
        st.switch_page("pages/info.py")

    if sel == 'Analyze Single Subject MRI Data':
        st.switch_page("pages/single_subject.py")

    if sel == 'Analyze a Group of Scans':
        st.switch_page("pages/multi_subject.py")

    if sel == 'Explore Results Only (No MRI Needed!)':
        st.switch_page("pages/no_user_mri.py")

