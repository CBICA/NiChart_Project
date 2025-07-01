import streamlit as st
import os
import yaml
from pathlib import Path
from graphviz import Digraph
from collections import defaultdict
import utils.utils_pages as utilpg
import utils.utils_processes as utilprc
import utils.utils_session as utilses
from streamlit_image_select import image_select
import re
from utils.utils_logger import setup_logger

import streamlit_antd_components as sac

logger = setup_logger()
logger.debug('Page: Select Pipelines')

# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()
utilpg.set_global_style()

def view_doc(pipeline, dtype) -> None:
    """
    Panel for viewing input data for a pipeline
    """
    fdoc = os.path.join(
        st.session_state.paths['resources'],
        'pipelines',
        pipeline,
        f'{pipeline}_{dtype}.md'
    )
    if not os.path.exists(fdoc):
        st.warning('Could not find doc file!')
        return
    
    if dtype == 'input' or dtype == 'output':
        with open(fdoc, 'r') as f:
            markdown_content = f.read()
            st.markdown(markdown_content)
            
    if dtype == 'example':
        with open(fdoc, 'r') as f:
            markdown_content = f.read()
            st.code(markdown_content.strip(), language="text")

def sel_pipeline_from_list():
    '''
    Select a pipeline
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

        sel_pipeline = sac.buttons(
            items=sitems,
            size='lg',
            radius='xl',
            align='left'
        )
        pname = pipelines.loc[pipelines.Name == sel_pipeline, 'Label'].values[0]
    
        tab = sac.tabs(
            items=[
                sac.TabsItem(label='Input Data'),
                sac.TabsItem(label='Output'),
                sac.TabsItem(label='Example'),
            ],
            size='lg',
            align='left'
        )

        if tab == 'Input Data':
            view_doc(pname, 'input')

        if tab == 'Output':
            view_doc(pname, 'output')
        
        if tab == 'Example':
            view_doc(pname, 'example')
            
        sac.divider(label='', align='center', color='gray')
            
        if st.button('Select'):
            st.session_state.sel_pipeline = pname
            st.success(f'Pipeline selected: {pname}')

st.markdown(
    """
    ### Pipelines
    """
)

#tab = sac.tabs(
    #items=[
        #sac.TabsItem(label='List View'),
        #sac.TabsItem(label='Graph View'),
    #],
    #size='lg',
    #align='left'
#)

## List view
#if tab == 'List View':
    #sel_pipeline_from_list()

## Graph view
#if tab == 'Graph View':
    #st.info('Coming soon!')

sel_pipeline_from_list()

# Show selections
utilses.disp_selections()

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()



