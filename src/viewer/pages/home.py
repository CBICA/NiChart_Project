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
            f'###### Explore NiChart {utilmisc.styled_text('(No Data Upload Required)')}', unsafe_allow_html=True
        )
        st.markdown(
            f'- Visualize distributions of imaging variables and biomarkers (pre-computed using NiChart reference data)', unsafe_allow_html=True
        )

        st.markdown(
            '''
            ###### Apply NiChart to Your Data
            
            - Select Your Pipeline
            
            - Upload Your Data
            
            - Run pipeline
            
            - View / Download Your Results
            ''', unsafe_allow_html=True
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
    
def user_select_var(sel_var_groups, plot_params, var_type, add_none = False):
    '''
    User panel to select a variable 
    Variables are grouped in categories
    '''
    df_groups = st.session_state.dicts['df_var_groups'].copy()
    df_groups = df_groups[df_groups.category.isin(sel_var_groups)]

    st.markdown(f'##### Variable: {var_type}')
    cols = st.columns([1,3])
    with cols[0]:
        
        list_group = df_groups.group.unique().tolist()
        try:
            curr_value = plot_params[f'{var_type}_group']
            curr_index = list_group.index(curr_value)
        except ValueError:
            curr_index = 0
            
        st.selectbox(
            "Variable Group",
            list_group,
            key = f'_{var_type}_group',
            index = curr_index
        )
        plot_params[f'{var_type}_group'] = st.session_state[f'_{var_type}_group']

    with cols[1]:

        sel_group = plot_params[f'{var_type}_group']
        if sel_group is None:
            return
        
        sel_atlas = df_groups[df_groups['group'] == sel_group]['atlas'].values[0]
        list_vars = df_groups[df_groups['group'] == sel_group]['values'].values[0]
        
        # Convert MUSE ROI variables from index to name
        if sel_atlas == 'muse':
            roi_dict = st.session_state.dicts['muse']['ind_to_name']
            list_vars = [roi_dict[k] for k in list_vars]

        if add_none:
            list_vars = ['None'] + list_vars

        try:
            curr_value = plot_params[var_type]
            curr_index = list_vars.index(curr_value)
        except ValueError:
            curr_index = 0
            
        st.selectbox(
            "Variable Name",
            list_vars,
            key = f'_{var_type}',
            index = curr_index
        )
        
        plot_params[var_type] = st.session_state[f'_{var_type}']

def user_select_var2(sel_var_groups, plot_params, var_type, add_none = False):
    '''
    User panel to select a variable 
    Variables are grouped in categories
    '''
    df_groups = st.session_state.dicts['df_var_groups'].copy()
    df_groups = df_groups[df_groups.category.isin(sel_var_groups)]

    print(df_groups)

    # Create nested var lists
    sac_items = []
    for tmpg in df_groups.group.unique().tolist():
        
        print(tmpg)
        print(tmpl)
        return

        tmpl = df_groups[df_groups['group'] == tmpg]['values'].values[0]
        tmp_item = sac.CasItem(tmpg, icon='app', children=tmpl)
        sac_items.append(tmp_item)

    print(sac_items)


    sel = sac.cascader(
        items = sac_items,
        label='Select x var', index=[0,1], multiple=False, search=True, clear=True
    )
        
    st.write(sel)


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
        #sac.TabsItem(label='Test'),
    ],
    size='lg',
    align='left'
)


if tab == 'Overview':
    view_overview()

if tab == 'Quick Start':
    view_quick_start()

if tab == 'Links':
    view_links()

if tab == 'Installation':
    view_installation()

#if tab == 'Test':
    #view_test()

# Show selections
utilses.disp_selections()
    
# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()
