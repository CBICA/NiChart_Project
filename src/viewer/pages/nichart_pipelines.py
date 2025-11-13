import streamlit as st
import os
import yaml
from pathlib import Path
from graphviz import Digraph
from collections import defaultdict
import utils.utils_misc as utilmisc
import utils.utils_pages as utilpg
import utils.utils_processes as utilprc
import utils.utils_session as utilses
import utils.utils_data_view as utildv
from utils.utils_styles import inject_global_css

from streamlit_image_select import image_select
import re
from utils.utils_logger import setup_logger

import streamlit_antd_components as sac

logger = setup_logger()
logger.debug("--- STARTING: Run Pipelines ---")

inject_global_css()

# Page config should be called for each page
utilpg.config_page()
utilpg.set_global_style()


if 'instantiated' not in st.session_state or not st.session_state.instantiated:
    utilses.init_session_state()

#################################
## Function definitions
def help_message(workflow):

    with st.popover("❓", width='content'):
        st.write(
            """
            **How to Use This Page**

            - Select a pipeline
            - Run the pipeline
            - View progress
            """
        )

def show_description(pipeline) -> None:
    """
    Panel for viewing pipeline description
    """
    with st.container(border=True, height=300):
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

def select_pipeline():
    '''
    Select a pipeline and show overview
    '''
    st.markdown("##### Select:")

    sac.divider(key='_p2_div1')

    pipelines = st.session_state.pipelines
    pnames = pipelines.Name.tolist()

    sel_opt = sac.chip(
        pnames,
        label='', index=0, align='left',
        size='md', radius='md', multiple=False, color='cyan',
        description='Select a pipeline'
    )

    show_description(sel_opt.lower())


def pipeline_menu():
    cols = st.columns([10,1,10])
    out_dir = os.path.join(
        st.session_state.paths['out_dir'], st.session_state['prj_name']
    )

    with cols[0]:
        select_pipeline()
    # with cols[2]:

#################################
## Main

workflow = st.session_state.workflow

with st.container(horizontal=True, horizontal_alignment="center"):
    st.markdown("<h4 style=color:#3a3a88;'>Select and Run Pipeline\n\n</h1>", unsafe_allow_html=True, width='content')
    help_message(workflow)

if st.session_state.workflow == 'ref_data':
    st.info('''
        You’ve selected the **Reference Data** workflow. This option doesn’t require pipeline selection.
        #- If you meant to analyze your data, please go back and choose a different workflow.
        - Otherwise, continue to the next step to explore the reference values.
        '''
    )

    sac.divider()
    
    sel_but = sac.chip(
        [
            sac.ChipItem(label = '', icon='arrow-right', disabled=False)
        ],
        label='', align='center', color='#aaeeaa', size='xl', return_index=True
    )
        
    if sel_but == 0:
        st.switch_page(f'pages/nichart_results.py')

else:
    pipeline_menu()
    sac.divider(key='_p0_div1')
    sel_but = sac.chip(
        [
            sac.ChipItem(label = '', icon='arrow-left', disabled=False),
            sac.ChipItem(label = '', icon='arrow-right', disabled=False)
        ],
        label='', align='center', color='#aaeeaa', size='xl', return_index=True
    )

    if sel_but == 0:
        st.switch_page("pages/nichart_data.py")

    if sel_but == 1:
        st.switch_page("pages/nichart_results.py")

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()



