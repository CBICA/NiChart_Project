
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


with st.form('Select:'):
    sel_layout = st.radio("Choose layout:", ["Main", "Sidebar"], horizontal=True)

    submitted = st.form_submit_button('Submit')
    if submitted:
        st.success('Selected: {sel_layout}')
        st.session_state.layout = sel_layout
