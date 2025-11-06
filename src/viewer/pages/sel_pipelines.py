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
            

def pipeline_selector_categories():
    # User selects their desired category, selection is filtered

    pipelines = st.session_state.pipelines
    sitems = []
    colors = st.session_state.pipeline_colors
    categories = st.session_state.pipeline_categories
    reqs = st.session_state.pipeline_requirements
        
    processing_pipelines = categories['image-processing']
    inference_pipelines = categories['inference']
    names = pipelines.Name.tolist()
    labels = pipelines.Label.tolist()
    shortcodes = pipelines.PipelineYaml.tolist()
    harmonized_shortcodes = pipelines.HarmonizedPipelineYaml.tolist()

    left, right = st.columns(2)
    only_show_harmonizable = st.checkbox("Only show pipelines whose output can be harmonized to the reference data.")
    with left:
        with st.container(border=True):
            st.markdown("### Feature-Extraction Pipelines")
            for i, ptmp in enumerate(names):
                if only_show_harmonizable:
                    if shortcodes[i] in processing_pipelines and harmonized_shortcodes[i]:
                        sitems.append(
                            sac.ButtonsItem(
                                label=ptmp, color = colors[i%len(colors)]
                            ))
                
                else:
                    if shortcodes[i] in processing_pipelines:
                        sitems.append(
                            sac.ButtonsItem(
                                label=ptmp, color = colors[i%len(colors)]
                            ))
                

            sel_pipeline = sac.buttons(
                items=sitems,
                size='lg',
                radius='xl',
                align='left'
            )
            pname = pipelines.loc[pipelines.Name == sel_pipeline, 'Label'].values[0]

    with right:
        with st.container(border=True):
            st.markdown("### Inference Pipelines")
            for i, ptmp in enumerate(names):
                if only_show_harmonizable:
                    if shortcodes[i] in inference_pipelines and harmonized_shortcodes[i]:
                        sitems.append(
                            sac.ButtonsItem(
                                label=ptmp, color = colors[i%len(colors)]
                            ))
                
                else:
                    if shortcodes[i] in inference_pipelines:
                        sitems.append(
                            sac.ButtonsItem(
                                label=ptmp, color = colors[i%len(colors)]
                            ))
                
            sel_pipeline = sac.buttons(
                items=sitems,
                size='lg',
                radius='xl',
                align='left'
            )
            pname = pipelines.loc[pipelines.Name == sel_pipeline, 'Label'].values[0]

    sac.divider(label='', align='center', color='gray')
            
    if st.button('Select'):
        st.session_state.sel_pipeline = pname
        st.session_state.pipeline_selected_explicitly = True
        st.success(f'Pipeline selected: {pname}')

    pass

def pipeline_selector_selectdatatypes():
    # User selects their data types, selection is filtered (others show up as gray/disabled)
    pass

def pipeline_selector_autodatatypes():
    # This one should auto-detect the user's available data 
    pass

def sel_pipeline_from_list():
    '''
    Select a pipeline
    '''
    with st.container(border=True):
        pipelines = st.session_state.pipelines
        sitems = []
        colors = st.session_state.pipeline_colors
        categories = st.session_state.pipeline_categories
        reqs = st.session_state.pipeline_requirements

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
            st.session_state.pipeline_selected_explicitly = True
            st.success(f'Pipeline selected: {pname}')

st.markdown(
    """
    ### Pipelines
    
    - Select a processing pipeline to apply on your data
    
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

st.markdown("# Category View")
pipeline_selector_categories()

st.markdown("# View based on user datatype selection")
pipeline_selector_selectdatatypes()

st.markdown("# Keyword View")

st.markdown("# Old Page Content")
sel_pipeline_from_list()

# Show selections
utilses.disp_selections()

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()



