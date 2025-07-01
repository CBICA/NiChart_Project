import streamlit as st
import utils.utils_pages as utilpg
import utils.utils_plots as utilpl
import utils.utils_misc as utilmisc
import utils.utils_mriview as utilmri
import utils.utils_session as utilses
import pandas as pd
from streamlit_image_select import image_select
from stqdm import stqdm
import os
from utils.utils_logger import setup_logger

import streamlit_antd_components as sac

logger = setup_logger()
logger.debug('Page: Explore Nichart')

# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()
utilpg.set_global_style()

def view_synthseg() -> None:
    """
    Panel for viewing synthseg results
    """    
    # Select result type 
    st.info('Coming soon!')

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

    if sel_res_type == 'Regional Volumes':
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

    elif sel_res_type == 'Segmentation':
        ulay = st.session_state.ref_data["t1"]
        olay = st.session_state.ref_data["dlmuse"]        
        mri_params = st.session_state.mri_params

        utilmri.panel_set_params(
            mri_params, ['roi'], 'muse'
        )

        utilmri.panel_view_seg(
            ulay, olay, mri_params
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
        mri_params = st.session_state.mri_params

        utilmri.panel_set_params(
            mri_params, ['roi'], 'wmls'
        )

        utilmri.panel_view_seg(
            ulay, olay, mri_params
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
    with st.container(border=True):
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
        st.info('NiChart Data')

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
        pname = pipelines.loc[pipelines.Name == sel_pipeline, 'Label'].values[0]
        st.session_state.sel_pipeline = sel_pipeline
        show_description(pname)

def results_overview():
    '''
    Select a pipeline and show overview
    '''
    print(st.session_state.sel_pipeline)
    
    with st.container(border=True):
        if st.session_state.sel_pipeline == 'DLMUSE':
            view_dlmuse()

        elif st.session_state.sel_pipeline == 'DLWMLS':
            view_dlwmls()

#st.info(
st.markdown(
    """
    ### Explore Neuroimaging Chart
    """
)

tab = sac.tabs(
    items=[
        sac.TabsItem(label='Data'),
        sac.TabsItem(label='Pipelines'),
        sac.TabsItem(label='Results Preview'),
    ],
    size='lg',
    align='left'
)

# Select pipeline
if tab == 'Data':
    data_overview()
    
if tab == 'Pipelines':
    pipeline_overview()

if tab == 'Results Preview':
    results_overview()
    
if st.session_state.mode == 'debug':
    utilses.disp_session_state()
