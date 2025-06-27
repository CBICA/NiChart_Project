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

def view_input_data(method) -> None:
    """
    Panel for viewing input data for a pipeline
    """
    with st.container(border=True):
        fdoc = os.path.join(
            st.session_state.paths['resources'],
            'pipelines',
            method,
            'data_' + method + '.md'
        )
        with open(fdoc, 'r') as f:
            markdown_content = f.read()
        #st.markdown(markdown_content)
        
        parts = re.split(r"```", markdown_content)

        st.markdown(parts[0])
        st.code(parts[1].strip(), language="text")


def sel_pipeline_from_list():
    # Show a thumbnail image for each pipeline
    pdict = dict(
        zip(st.session_state.pipelines['Name'], st.session_state.pipelines['Label'])
    )
    pdir = os.path.join(st.session_state.paths['resources'], 'pipelines')
    logo_fnames = [
        os.path.join(pdir, pname, f'logo_{pname}.png') for pname in list(pdict.values())
    ]
    psel = image_select(
        "",
        images = logo_fnames,
        captions=list(pdict.keys()),
        index=-1,
        return_value="index",
        use_container_width = False
    )
    
    # Show description of the selected pipeline
    if psel < 0 :
        return
    
    sel_pipeline = list(pdict.values())[psel]
    if st.button('Select'):
        st.session_state.sel_pipeline = sel_pipeline
        st.success(f'Pipeline selected: {sel_pipeline}')
        view_input_data(sel_pipeline)

# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()

st.markdown(
    """
    ### Pipelines
    """
)

tab = sac.tabs(
    items=[
        sac.TabsItem(label='List View'),
        sac.TabsItem(label='Graph View'),
    ],
    size='lg',
    align='left'
)

if tab == 'List View':
    sel_pipeline_from_list()

elif tab == 'Graph View':
    st.info('Coming soon!')
    #sel_pipeline_from_graph()
    

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()



