import streamlit as st
import utils.utils_pages as utilpg
import utils.utils_io as utilio
import utils.utils_plots as utilpl
import utils.utils_mriview as utilmri
import utils.utils_data_view as utildv
import utils.utils_session as utilses
import pandas as pd
import os
from pathlib import Path

# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()

def panel_data_overview():
    '''
    Detect all csv files and merge them
    '''
    with st.container(border=True):
        in_dir = st.session_state.paths['project']
        utildv.data_overview(in_dir)

def panel_data_merge():
    '''
    Detect all csv files and merge them
    '''
    in_dir = st.session_state.paths['project']
    utildv.data_merge(in_dir)

def plot_vars():
    """
    Panel for viewing dlmuse results
    """    
    # st.write(st.session_state.plot_params)

    st.session_state.plot_data['df_data'] = utilpl.read_data(st.session_state.paths['plot_data']) 
    var_groups_data = ['demog', 'roi']
    var_groups_hue = ['cat_vars']
    pipeline = 'dlmuse'

    utilpl.panel_set_plot_params(
        st.session_state.plot_params,
        var_groups_data,
        var_groups_hue,
        pipeline
    )

    print(st.session_state.plot_params)
        
    utilpl.panel_show_plots()

    with st.expander('hello2'):
        for i in [0,1,2,3]:
            if f'_key_plot_sel_{i}' in st.session_state:
                st.write(st.session_state[f'_key_plot_sel_{i}'])

    with st.expander('hello'):
        # st.write(st.session_state.plot_settings)
        #st.write(st.session_state.plots)
        #st.write(st.session_state.selections)
        st.write(st.session_state.plot_params)


def view_images():
    ulay = st.session_state.ref_data["t1"]
    olay = st.session_state.ref_data["dlmuse"]        
    #utilmri.panel_view_seg(ulay, olay, 'muse')

st.markdown(
    """
    ### View Results 
    """
)

my_tabs = st.tabs(
    ["Overview", "Prepare Data", "Plot Data", "View Images"]
)

with my_tabs[0]:
    panel_data_overview()

with my_tabs[1]:
    panel_data_merge()

with my_tabs[2]:
    plot_vars()

with my_tabs[3]:
    view_images()

# Show session state vars
if st.session_state.mode == 'debug':
    if st.sidebar.button('Show Session State'):
        utilses.disp_session_state()
