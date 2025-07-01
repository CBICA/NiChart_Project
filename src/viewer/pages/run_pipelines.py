import streamlit as st
import os
import yaml
from pathlib import Path
from graphviz import Digraph
from collections import defaultdict
import utils.utils_pages as utilpg
import utils.utils_processes as utilprc
import utils.utils_pipelines as utilpipe
import utils.utils_session as utilses
from utils.utils_logger import setup_logger

import streamlit_antd_components as sac

logger = setup_logger()
logger.debug('Page: Run Pipelines')

# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()
utilpg.set_global_style()

def panel_conf_pipeline():
    with st.container(border=True):
        st.markdown(
            """
            ##### Select parameters for each step
            """
        )
        for sel_step_name in st.session_state.processes['sel_steps']:
            sel_step = st.session_state.processes['steps'][sel_step_name]
            with st.expander(sel_step_name, expanded=False):
                
                parameter_values = {}

                for param in sel_step.get("parameters", []):
                    name = param["name"]
                    ptype = param["type"]
                    default = param.get("default")

                    if ptype == "bool":
                        value = st.checkbox(
                            name, value=default, key=f'_key_check_{sel_step_name}'
                        )
                    elif ptype == "int":
                        value = st.number_input(
                            name, value=default, step=1, key=f'_key_ni_{sel_step_name}'
                        )
                    elif ptype == "float":
                        value = st.number_input(
                            name, value=default, key=f'_key_ni_{sel_step_name}'
                        )
                    elif ptype == "str":
                        value = st.text_input(
                            name, value=default
                        )
                        
                    elif ptype == "select" and "options" in param:
                        value = st.selectbox(
                            name, param["options"], index=param["options"].index(default)
                        )
                    else:
                        st.warning(f"Unsupported type: {ptype}")
                        value = None

                    parameter_values[name] = value
                
                if st.button(
                    'Confirm',
                    key = f'_key_btn_confirm_{sel_step}'
                ):
                    # Show selected values
                    st.success("User-selected parameter values:")
                    st.json(parameter_values)                


def panel_verify_data():
    """
    Panel for verifying required pipeline data
    """
    sel_method = st.session_state.sel_pipeline
    sel_project = st.session_state.project
    in_dir = st.session_state.paths['project']
    
    st.success(f'Project Name: {sel_project}')
    st.success(f'Pipeline Name: {sel_method}')
    
    if st.button('Verify'):    
        flag_data = utilpipe.verify_data(sel_method)
        if flag_data:
            st.success('Input data verified!')
        else:
            st.error('Please check input data!')

def panel_run_pipeline():
    """
    Panel for running a pipeline
    """
    st.info('Coming soon!')

def panel_view_status():
    """
    Panel to view status of pipeline
    """
    st.info('Coming soon!')

st.markdown(
    """
    ### Run a pipeline
    """
)

tab = sac.tabs(
    items=[
        sac.TabsItem(label='Verify Input Data'),
        sac.TabsItem(label='Run Pipeline'),
        sac.TabsItem(label='View Status')
    ],
    size='lg',
    align='left'
)

if tab == 'Verify Input Data':
    panel_verify_data()
    
elif tab == 'Run Pipeline':
    panel_run_pipeline()

elif tab == 'View Status':
    panel_view_status()

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()

