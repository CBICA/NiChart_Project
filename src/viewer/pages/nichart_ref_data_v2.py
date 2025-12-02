import streamlit as st
import os
import pandas as pd
import yaml
from pathlib import Path
from graphviz import Digraph
from collections import defaultdict
import utils.utils_misc as utilmisc
import utils.utils_plots as utilpl
import utils.utils_pages as utilpg
import utils.utils_processes as utilprc
import utils.utils_session as utilses
from streamlit_image_select import image_select
import re
from utils.utils_logger import setup_logger

import streamlit_antd_components as sac

logger = setup_logger()
logger.debug('Page: NiChart Reference Data')

# Page config should be called for each page
utilpg.config_page()
utilpg.set_global_style()

# Set data type
st.session_state.workflow = 'ref_data'

if 'instantiated' not in st.session_state or not st.session_state.instantiated:
    utilses.init_session_state()

def view_overview():
    with st.container(border=True):
        st.markdown(
            '''
            
            - Welcome! This is where you can view neuroimaging chart values from the reference sample.
            
            '''
        )

def view_dlmuse() -> None:
    """
    Panel for viewing dlmuse results
    """
    list_res_type = ['Regional Volumes', 'Segmentation']
    sel_res_type = sac.tabs(
        list_res_type,
        size='lg',
        align='left'
    )   

    ## FIXME (list of rois from data file to init listbox selections)
    df = pd.read_csv(
            os.path.join(
                st.session_state.paths['resources'],
                'reference_data', 'centiles', 'dlmuse_centiles_CN.csv' 
            )
    )
    list_vars = ['Age', 'Sex'] + df.VarName.unique().tolist()


    if sel_res_type == 'Regional Volumes':
        var_groups_data = ['roi']
        pipeline = 'dlmuse'

        # Set centile selections
        st.session_state.plot_params['centile_values'] = st.session_state.plot_settings['centile_trace_types']

        with st.sidebar:
            sac.divider(label='Viewing Options', align='center', color='gray')
            utilpl.user_add_plots(
                st.session_state.plot_params
            )
            
        utilpl.sidebar_flag_hide_setting()
        utilpl.sidebar_flag_hide_legend()

        utilpl.panel_set_params_centile_plot(
            st.session_state.plot_params,
            var_groups_data,
            pipeline,
            list_vars
        )
        utilpl.panel_show_centile_plots()

        st.write()

    elif sel_res_type == 'Segmentation':
        ulay = st.session_state.ref_data["t1"]
        olay = st.session_state.ref_data["dlmuse"]        

        with st.sidebar:
            sac.divider(label='Viewing Options', align='center', color='gray')
        utilpl.sidebar_flag_hide_setting()

        utilmri.panel_set_params(
            st.session_state.plot_params,
            ['roi'],
            'muse',
            list_vars
        )

        utilmri.panel_view_seg(
            ulay, olay, st.session_state.plot_params
        )

def view_dlwmls() -> None:
    """
    Panel for viewing dlwmls segmentation
    """
    # Select result type        
    list_res_type = ['Segmentation']
    sel_res_type = st.pills(
        'Select result type',
        list_res_type,
        default = None,
        selection_mode = 'single',
        label_visibility = 'collapsed',
    )
    
    if sel_res_type == 'Segmentation':
        ulay = st.session_state.ref_data["fl"]
        olay = st.session_state.ref_data["dlwmls"]

        plot_params = st.session_state.plot_params
        
        utilmri.panel_view_seg(
            ulay, olay, 
        )

def view_dlmuse_biomarkers() -> None:
    """
    Panel for viewing biomarkers
    """
    st.info('Coming soon!')

def view_spare() -> None:
    """
    Panel for viewing biomarkers
    """
    st.info('Coming soon!')

def view_surrealgan() -> None:
    """
    Panel for viewing biomarkers
    """
    st.info('Coming soon!')

def show_description(pipeline) -> None:
    """
    Panel for viewing pipeline description
    """
    f_logo = os.path.join(
        st.session_state.paths['resources'], 'pipelines', pipeline, f'logo_{pipeline}.png'
    )
    fdoc = os.path.join(
        st.session_state.paths['resources'], 'pipelines', pipeline, f'overview_{pipeline}.md'
    )
    cols = st.columns([6, 1])
    with cols[0]:
        with open(fdoc, 'r') as f:
            st.markdown(f.read())
    with cols[1]:
        st.image(f_logo)

def data_overview():
    '''
    Description of NiChart data
    '''
    with st.container(border=True):
        st.markdown(
            ''' NiChart Reference Dataset is a large and diverse collection of MRI images from multiple studies. It was created as part of the ISTAGING project to develop a system for identifying imaging biomarkers of aging and neurodegenerative diseases. The dataset includes multi-modal MRI data, as well as carefully curated demographic, clinical, and cognitive variables from participants with a variety of health conditions. The reference dataset is a key component of NiChart for training machine learning models and for creating reference distributions of imaging measures and signatures, which can be used to compare NiChart values that are computed from the user data to normative or disease-related reference values.
            '''
        )
        st.image(
            os.path.join(
                st.session_state.paths['resources'], 'images', 'nichart_data.png'
            ),
            width=1200
        )

def pipeline_overview():
    '''
    Select a pipeline and show overview
    '''
    with st.container(border=True):

        pipelines = st.session_state.pipelines
        sitems = []
        colors = st.session_state.pipeline_colors
        for i, ptmp in enumerate(pipelines.Name.tolist()):
            sitems.append(
                sac.ButtonsItem(
                    label=ptmp, color = colors[i%len(colors)]
                )
            )
        
        sel_index = utilmisc.get_index_in_list(
            pipelines.Name.tolist(), st.session_state.sel_pipeline
        )
        sel_pipeline = sac.buttons(
            items=sitems,
            size='lg',
            radius='xl',
            align='left',
            index =  sel_index,
            key = '_sel_pipeline'
        )        
        label_matches = pipelines.loc[pipelines.Name == sel_pipeline, 'Label'].values
        if len(label_matches) == 0: # No selection
            return
        
        pname = label_matches[0]
        st.session_state.sel_pipeline = pname
        
        #sac.divider(label='Description', align='center', color='gray')
        
        show_description(pname)

def results_overview():
    '''
    Select a pipeline and show overview
    '''
    ## Set flag for hiding the settings
    #if '_flag_hide_settings' not in st.session_state:
        #st.session_state['_flag_hide_settings'] = st.session_state.plot_settings['flag_hide_settings']

    #def update_val():
        #st.session_state.plot_settings['flag_hide_settings'] = st.session_state['_flag_hide_settings']

    #with st.sidebar:
        #sac.divider(label='Plot Settings', align='center', color='gray')
        #st.checkbox(
            #'Hide Plot Settings',
            #key = '_flag_hide_settings',
            #on_change = update_val
        #)
    
    # Show results
    with st.container(border=True):
        if st.session_state.sel_pipeline == 'dlmuse':
            view_dlmuse()

        elif st.session_state.sel_pipeline == 'dlwmls':
            st.warning('Viewer not implemented for dlwmls')
            #view_dlwmls()

st.markdown("<h5 style='text-align:center; color:#3a3a88;'>NiChart Reference Distributions\n\n</h1>", unsafe_allow_html=True)

sel = sac.tabs([
    sac.TabsItem(label='Overview'),
    sac.TabsItem(label='Data'),
    sac.TabsItem(label='Pipelines'),
    sac.TabsItem(label='Results'),
    sac.TabsItem(label='Select Pipeline'),
    sac.TabsItem(label='Download Results'),
    sac.TabsItem(label='Go Back Home'),
], align='center',  size='xl', color='grape')

if sel == 'Overview':
    view_overview()
    
if sel == 'Data':
    data_overview()
    
if sel == 'Pipelines':
    pipeline_overview()

if sel == 'Results':
    results_overview()

if sel == 'Go Back Home':
    st.switch_page("pages/nichart_home.py")


# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()



