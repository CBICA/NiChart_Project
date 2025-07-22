import utils.utils_pages as utilpg
utilpg.config_page()

import streamlit as st
import utils.utils_plots as utilpl
import utils.utils_io as utilio
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
utilpg.show_menu()
utilpg.set_global_style()

def select_data_files():
    """
    Panel for merging selected data files
    """    
    tab = sac.tabs(
        items=[
            sac.TabsItem(label='Select'),
            sac.TabsItem(label='View'),
            sac.TabsItem(label='Reset'),
        ],
        size='lg',
        align='left'
    )
    
    out_dir = os.path.join(
        st.session_state.paths['project'], 'data_merged'
    )

    if tab == 'Select':
        with st.container(border=True):
            in_dir = st.session_state.paths['project']
            utildv.select_files(in_dir)

    elif tab == 'View':
        try:
            fname = os.path.join(
                out_dir, 'data_merged.csv'
            )
            df_data = pd.read_csv(fname)
            st.dataframe(df_data)
        except:
            st.warning('Could not read data file!')
        
    elif tab == 'Reset':
        st.info(f'Out folder name: {out_dir}')
        if st.button("Delete"):
            utilio.remove_dir(out_dir)

def plot_vars():
    """
    Panel for viewing dlmuse results
    """
    csv_plot = os.path.join(
        st.session_state.paths['project'], 'data_merged', 'data_merged.csv'
    )
    if os.path.exists(csv_plot):
        st.session_state.plot_data['df_data'] = utilpl.read_data(
            csv_plot
        )
    var_groups_data = ['demog', 'roi']
    var_groups_hue = ['cat_vars']
    pipeline = 'dlmuse'
    list_vars = st.session_state.plot_data['df_data'].columns.tolist()

    with st.sidebar:
        sac.divider(label='Viewing Options', align='center', color='gray')
        utilpl.user_add_plots(
            st.session_state.plot_params
        )
    utilpl.sidebar_flag_hide_setting()
    utilpl.sidebar_flag_hide_legend()
    utilpl.sidebar_flag_hide_mri()

    utilpl.panel_set_params_plot(st.session_state.plot_params, pipeline, list_vars)

    utilpl.panel_show_plots()

st.markdown(
    """
    ### Results Dashboard
    
    - Plot imaging variables and biomarkers derived from your dataset together with reference distributions
    """
)

tab = sac.tabs(
    items=[
        sac.TabsItem(label='Select Data Files'),
        sac.TabsItem(label='Plot Data'),
    ],
    size='lg',
    align='left'
)

if tab == 'Select Data Files':
    select_data_files()

elif tab == 'Plot Data':
    plot_vars()

# Show selections
utilses.disp_selections()

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()

