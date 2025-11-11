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
utilpg.set_global_style()

if 'instantiated' not in st.session_state or not st.session_state.instantiated:
    utilses.init_session_state()


def extract_sections(in_doc):
    with open(in_doc, 'r') as f:
        lines = f.readlines()

    sections = {"Input": "", "Output": "", "Example": ""}
    current_section = None

    for line in lines:
        line = line.strip()
        if line.lower().startswith("## input"):
            current_section = "Input"
            continue
        elif line.lower().startswith("## output"):
            current_section = "Output"
            continue
        elif line.lower().startswith("## example"):
            current_section = "Example"
            continue

        #print(line)

        if current_section:
            sections[current_section] += line + "\n"

    return sections

def view_doc(pipeline, dtype) -> None:
    """
    Panel for viewing input data for a pipeline
    """
    fdoc = os.path.join(
        st.session_state.paths['resources'],
        'pipelines',
        pipeline,
        f'{pipeline}_doc.md'
    )
    if not os.path.exists(fdoc):
        st.warning('Could not find doc file!')
        return
    
    sections = extract_sections(fdoc)
        
    if dtype in ['Input', 'Output']:
        st.markdown(sections[dtype])

    if dtype in ['Example']:
        st.code(sections[dtype])
            

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
                sac.TabsItem(label='Input'),
                sac.TabsItem(label='Output'),
                sac.TabsItem(label='Example'),
            ],
            size='lg',
            align='left'
        )

        view_doc(pname, tab)
            
        sac.divider(label='', align='center', color='gray')
            
        if st.button('Select'):
            st.session_state.sel_pipeline = pname
            st.success(f'Pipeline selected: {pname}')

st.markdown(
    """
    ### Pipelines
    
    - Select a processing pipeline to apply on your data
    
    """
)

sel_pipeline_from_list()

# Show selections
utilses.disp_selections()

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()



