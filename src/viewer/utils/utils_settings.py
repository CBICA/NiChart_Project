# -*- coding: utf-8 -*-
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import streamlit_antd_components as sac

@st.dialog('Settings', width='medium')
def edit_settings():
    with st.form('Select:'):
        sel_layout_label = st.radio("Choose layout:", ["Main", "Sidebar"], horizontal=True)

        submitted = st.form_submit_button('Submit')
        
        if submitted:        
            if sel_layout_label == "Sidebar":
                layout = st.sidebar 
            else:
                layout = st.container(border=False)
            st.session_state.sel_layout_label = sel_layout_label
            st.session_state.layout = layout
            st.toast('Selected Layout: {sel_layout_label}')
            st.rerun()

def settings_button():
    # Top-right subtle “Settings” link
    but_set = sac.chip(
        [sac.ChipItem(label = 'Settings', icon='gear', disabled=False)],
        label='', align='right', color='#aaeeaa',
        size='sm', return_index=False,
        index=None
    )
    if but_set == 'Settings':
        edit_settings()
