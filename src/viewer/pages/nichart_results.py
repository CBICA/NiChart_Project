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
import utils.utils_io as utilio
import utils.utils_data_view as utildv
from utils.utils_styles import inject_global_css 

from streamlit_image_select import image_select
import re
from utils.utils_logger import setup_logger

import streamlit_antd_components as sac

logger = setup_logger()
logger.debug('Page: Select Pipelines')

inject_global_css()

# Page config should be called for each page
utilpg.config_page()
utilpg.set_global_style()


if 'instantiated' not in st.session_state or not st.session_state.instantiated:
    utilses.init_session_state()

def select_pipeline():
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
     
st.markdown("<h5 style='text-align:center; color:#3a3a88;'>Results\n\n</h1>", unsafe_allow_html=True)

select_pipeline()

sel_but = sac.chip(
    [
        sac.ChipItem(label = '', icon='arrow-left', disabled=False),
    ],
    label='', align='center', color='#aaeeaa', size='xl', return_index=True
)
    
if sel_but == 0:
    st.switch_page("pages/nichart_run_pipeline.py")

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()



