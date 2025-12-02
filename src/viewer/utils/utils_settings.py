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
        list_opts = ["Main", "Sidebar"]
        sel_ind = list_opts.index(st.session_state.layout_plots) 
        sel_layout = st.radio(
            "Choose layout for plotting parameters:",
            list_opts,
            index = sel_ind,
            horizontal=True
        )
        submitted = st.form_submit_button('Submit')
        
    if submitted:
        st.session_state.layout_plots = sel_layout
        st.toast('Selected Layout: {sel_layout}')
        #if st.session_state.get('_chip_navig'):
            #st.session_state._chip_navig = 1
        st.rerun()

#def settings_button():
    ## Top-right subtle “Settings” link
    #but_set = sac.chip(
        #[sac.ChipItem(label = 'Settings', icon='gear', disabled=False)],
        #label='', align='right', color='#aaeeaa',
        #size='sm', return_index=False,
        #index=None
    #)
    #if but_set == 'Settings':
        #edit_settings()

