
import utils.utils_pages as utilpg
# Page config should be called for each page
# utilpg.config_page()

import os
import numpy as np
import pandas as pd
import streamlit as st
import utils.utils_misc as utilmisc
import utils.utils_session as utilses
import utils.utils_alerts as utils_alerts
import utils.utils_survey as utils_survey
import gui.utils_mriview as utilmri
import gui.utils_plots as utilpl
import time

from streamlit_image_select import image_select
import logging
from stqdm import stqdm
from utils.utils_logger import setup_logger
from utils.utils_styles import inject_global_css 

import streamlit_antd_components as sac
import streamlit.components.v1 as components

import streamlit as st
from utils.nav import top_nav

from utils.utils_logger import setup_logger
logger = setup_logger()

logger.debug("--- STARTING: Home ---")

inject_global_css()

# Page config should be called for each page
#utilpg.config_page() # Done earlier above
utilpg.set_global_style()

html_style = '''
    <style>
    div:has( >.element-container div.floating) {
        display: flex;
        flex-direction: column;
        position: fixed;
        top: 4rem;        /* distance from the top */
        left: 0.75rem;
        z-index: 9999;    /* keep it above content */
    }

    div.floating {
        height:0%;
    }
    </style>
    '''
st.markdown(html_style, unsafe_allow_html=True)
if st.session_state.has_cloud_session:
    user_email = st.session_state.cloud_user_email
    with st.container():
        st.markdown('<div class="floating"></div>', unsafe_allow_html=True)
        col1, col2 = st.columns([6, 1])
        with col1: 
            logout_url = 'https://cbica-nichart.auth.us-east-1.amazoncognito.com/logout?client_id=4shr6mm2h0p0i4o9uleqpu33fj&logout_uri=https://neuroimagingchart.com'
            st.markdown(
                f""" Logged in as: {user_email}""",
                unsafe_allow_html=True
            )
        with col2:
            do_logout = st.button("Logout", type='primary')
            if do_logout:
                components.html(f"""
                    <script>
                    window.top.location.href = "{logout_url}";
                    </script>"""
                )

# Redirect users to survey page until it is completed or otherwise temporarily skipped
if not utils_survey.is_survey_completed():
    if 'skip_survey' in st.session_state:
        if not st.session_state.skip_survey:
            print("Activating survey page.")
            st.switch_page("pages/survey.py")
    else:
        print("Skipping survey due to session state.")
        st.switch_page("pages/survey.py")
else:
    print("Skipping survey, it's already completed.")
utils_alerts.render_alert()

with st.container(horizontal_alignment="center"):
    st.markdown("<h2 style='color:#5e5fad;'>Welcome to NiChart Project\n\n</h1>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h5 style='color:#3a3a88;'>What would you like to explore today?\n\n</h1>", unsafe_allow_html=True)

    sel = sac.chip(
        items=[
            sac.ChipItem(label='A Quick Introduction'),
            sac.ChipItem(label='Process a Single Subject'),
            sac.ChipItem(label='Process a Dataset'),
            sac.ChipItem(label='View NeuroImaging Charts'),
            sac.ChipItem(label='Ask NiChart Chatbot')
        ], label='', size='lg', radius='lg', direction='vertical', color='cyan'
    ) 
        
    if sel == 'A Quick Introduction':
        time.sleep(0.4)
        st.switch_page("pages/nichart_info.py")

    if sel == 'Process a Single Subject':
        time.sleep(0.4)
        st.switch_page("pages/nichart_single_subject.py")

    if sel == 'Process a Dataset':
        time.sleep(0.4)
        st.switch_page("pages/nichart_multi_subject.py")

    if sel == 'View NeuroImaging Charts':
        time.sleep(0.4)
        st.switch_page("pages/nichart_results.py")

    if sel == 'Ask NiChart Chatbot':
        time.sleep(0.4)
        st.switch_page("pages/nichart_chatbot.py")



# #st.markdown('<h1 class="centered-text">Welcome to NiChart Project</p>', unsafe_allow_html=True)
# st.markdown("<h2 style='text-align:center; color:#5e5fad;'>Welcome to NiChart Project\n\n</h1>", unsafe_allow_html=True)
# st.markdown("<br>", unsafe_allow_html=True)
# st.markdown("<h5 style='text-align:center; color:#3a3a88;'>What would you like to explore today?\n\n</h1>", unsafe_allow_html=True)

# sel = sac.chip(
#     items=[
#         sac.ChipItem(label='What is NiChart?'),
#         sac.ChipItem(label='Process a Single Subject'),
#         sac.ChipItem(label='Process a Dataset'),
#         sac.ChipItem(label='View NeuroImaging Charts'),
#     ], label='', align='center', size='lg', radius='lg', direction='vertical', color='cyan'
# ) 
    
# if sel == 'What is NiChart?':
#     time.sleep(0.6)
#     st.switch_page("pages/nichart_info.py")

# if sel == 'Process a Single Subject':
#     time.sleep(0.6)
#     st.switch_page("pages/nichart_single_subject.py")

# if sel == 'Process a Dataset':
#     time.sleep(0.6)
#     st.switch_page("pages/nichart_multi_subject.py")

# if sel == 'View NeuroImaging Charts':
#     time.sleep(0.6)
#     st.switch_page("pages/nichart_results.py")


