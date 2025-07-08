import streamlit as st
from stqdm import stqdm
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
import utils.utils_stlogbox as stlogbox
import utils.utils_toolloader as tl

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
    # For now, just hardcode the mapping from sel_pipeline to a pipeline in resources/pipelines.
    sel_method = st.session_state.sel_pipeline
    st.success(f'Selected pipeline: {sel_method}')
    harmonize = st.checkbox("Harmonize to reference data? (Requires >= 3 scans)")
    ## TODO: Retrieve dynamically/match between front end and toolloader code
    ## This a nice and simple placeholder for now
    sel_pipeline_to_id = {
        'dlmuse': 'run_dlmuse',
        'spare-ad': 'run_spare_ad',
        'spare-ba': 'run_spare_ba',
        'bascores': 'run_bascores',
        'dlwmls': None,
        'spare-cvm': None,
        'surrealgan': None,
        'synthseg': None,
        'ccl-nmf': None,
        ## Add additional lines here ({sel_pipeline value} : {name of pipeline yaml} )
    }
    if harmonize:
        sel_pipeline_to_id = {
            'dlmuse': 'run_dlmuse_harmonized',
            'spare-ad': 'run_spare_ad_harmonized',
            'spare-ba': 'run_spare_ba_harmonized',
            'bascores': 'run_bascores',
            'dlwmls': None,
            'spare-cvm': None,
            'surrealgan': None,
            'synthseg': None,
            'ccl-nmf': None,
            ## Add additional lines here ({sel_pipeline value} : {name of pipeline yaml} )
        }

    pipeline_to_run = sel_pipeline_to_id.get(sel_method, None)
    if pipeline_to_run is None:
        st.error("The currently selected pipeline doesn't have an associated tool configuration. Please submit a bug report!")
        return
    
    if st.button("Run pipeline"):
        with st.container():
            st.subheader("Pipeline Logs")
            with st.expander("View all pipeline logs"):
                with st.container():
                    log_committed_box = st.empty()
            with st.expander("View current step live logs"):
                with st.container():
                    log_live_box = st.empty()
        pipeline_progress_bar = stqdm(total=2, desc="Submitting pipeline...", position=0)
        process_progress_bar = stqdm(total=2, desc="Waiting...", position=0)
        #pipeline_progress_bar_slot = st.empty()
        #process_progress_bar_slot = st.empty()

        log = stlogbox.StreamlitJobLogger(log_committed_box, log_live_box)

        result = tl.run_pipeline(
            pipeline_id=pipeline_to_run, ##TODO EDIT THIS
            global_vars={"STUDY": st.session_state.paths["project"]},
            pipeline_progress_bar=pipeline_progress_bar,
            process_progress_bar=process_progress_bar,
            log=log
        )

        st.success(f"Pipeline {pipeline_to_run} finished successfully.")


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

# Show selections
utilses.disp_selections()

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()

