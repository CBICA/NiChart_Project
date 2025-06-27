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
from utils.utils_logger import setup_logger

import streamlit_antd_components as sac

logger = setup_logger()
logger.debug('Page: View Results')

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

    utilpl.panel_show_plots()

def view_images():
    ulay = st.session_state.ref_data["t1"]
    olay = st.session_state.ref_data["dlmuse"]        
    #utilmri.panel_view_seg(ulay, olay, 'muse')

st.markdown(
    """
    ### View Results 
    """
)

tab = sac.segmented(
    items=[
        sac.SegmentedItem(label='Overview'),
        sac.SegmentedItem(label='Prepare Data'),
        sac.SegmentedItem(label='Plot Data'),
        sac.SegmentedItem(label='View Images'),
    ],
    size='sm',
    radius='lg',
    align='left'
)

if tab == 'Overview':
    panel_data_overview()

elif tab == 'Prepare Data':
    panel_data_merge()

elif tab == 'Plot Data':
    plot_vars()

elif tab == 'View Images':
    view_images()

# Show session state vars
if st.session_state.mode == 'debug':
    if st.sidebar.button('Show Session State'):
        utilses.disp_session_state()

    if st.sidebar.button('Show Plot'):
        st.dataframe(st.session_state.plots)
