import os
import shutil
import time
from typing import Any

import pandas as pd
import numpy as np
import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform
from scipy import ndimage
import utils.utils_misc as utilmisc
import utils.utils_user_select as utiluser
import utils.utils_io as utilio
import utils.utils_toolloader as utiltl
import utils.utils_stlogbox as stlogbox

import utils.utils_session as utilses
import gui.utils_plots as utilpl
import gui.utils_mriview as utilmri
import gui.utils_view as utilview
import pandas as pd
import gui.utils_widgets as utilwd

import traceback

import streamlit_antd_components as sac

import streamlit as st
from stqdm import stqdm

from utils.utils_logger import setup_logger
logger = setup_logger()

#################################
## Function definitions
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

    st.session_state.sel_pipeline = sel_opt

    show_description(sel_opt.lower())

def show_description(pipeline) -> None:
    """
    Panel for viewing pipeline description
    """
    pipeline_label = utiltl.get_pipeline_label_by_name(pipeline)
    with st.container(border=True, height=300):
        if pipeline == '':
            st.markdown("No description exists for this pipeline.")
            return
        if pipeline_label is None:
            st.markdown("Could not locate a label for this pipeline.")
            return
        f_logo = os.path.join(
            st.session_state.paths['resources'], 'pipelines', pipeline_label, f'logo_{pipeline_label}.png'
        )
        fdoc = os.path.join(
            st.session_state.paths['resources'], 'pipelines', pipeline_label, f'overview_{pipeline_label}.md'
        )
        cols = st.columns([6, 1])
        with cols[0]:
            with open(fdoc, 'r') as f:
                st.markdown(f.read())
        with cols[1]:
            st.image(f_logo)

def select_pipeline(enabled_pnames):
    '''
    Select a pipeline and show overview
    '''
    st.markdown("##### Select:")
    show_enabled_only = st.checkbox("Show only pipelines which match my available data", value=True)
    sac.divider(key='_p2_div1')

    pipelines = st.session_state.pipelines
    pnames = pipelines.Name.tolist()

    
    if show_enabled_only:
        if not enabled_pnames:
            st.error(f"It looks like your data doesn't meet the requirements for any pipelines. Please browse the pipeline listing using the checkbox above, then go back and upload some data!")
            return
        sel_opt = sac.chip(
            enabled_pnames,
            label='', index=0, align='left',
            size='md', radius='md', multiple=False, color='cyan',
            description='Select a pipeline'
        )
    else:
        sel_opt = sac.chip(
            pnames,
            label='', index=0, align='left',
            size='md', radius='md', multiple=False, color='cyan',
            description='Select a pipeline'
        )
        
    
    row = pipelines.loc[pipelines["Name"] == sel_opt, "Label"]
    sel_label = row.iloc[0] if not row.empty else ''
    show_description(sel_opt)
    st.session_state.sel_pipeline_name = sel_opt
    st.session_state.sel_pipeline_label = sel_label
    #st.info(f"DEBUG: sel_opt {sel_opt}")
    #st.info(f"DEBUG: sel_label {sel_label}")

    return sel_label

def pipeline_runner_menu(enabled_pnames, sel=False):
    st.markdown("##### Run:")
    sac.divider(key='_p2_div2')
    if not sel:
        st.info("Select a pipeline on the left, then look here to run it.")
        return
    sel_method = st.session_state.sel_pipeline_label
    sel_name = utiltl.get_pipeline_name_by_label(sel_method)
    st.success(f'Selected pipeline: {sel_name}')
    harmonize = False
    if 'subject_type' not in st.session_state or st.session_state.subject_type == 'multi':
        if utiltl.pipeline_is_harmonizable(sel_method):
            harmonize = st.checkbox("Harmonize to reference data? (Requires >= 30 scans)")
    st.session_state.do_harmonize = harmonize
    ## TODO: Retrieve dynamically/match between front end and toolloader code
    ## This a nice and simple placeholder for now
    if sel_name not in enabled_pnames:
        #st.info(f"DEBUG: Enabled pnames: {enabled_pnames}")
        #st.info(f"DEBUG: sel_name: {sel_name}")
        #st.info(f"DEBUG: sel_pipeline: {sel_method}")
        st.info("Your data doesn't meet the requirements for this pipeline. Correct the issues marked below to proceed.")
        utiltl.check_requirements_met_panel(sel_name)
        return
    pipeline_to_run = utiltl.get_pipeline_id_by_label(sel_method, harmonized=harmonize)

    if pipeline_to_run is None:
        st.error("The currently selected pipeline doesn't have an associated tool configuration. Please submit a bug report!")
        return
    skip_steps_when_possible = True
    skip_steps_when_possible = st.checkbox("Accelerate pipeline via caching? (Uncheck to force re-runs)", value=True)
    alert_placeholder = st.empty()
    if st.button("Run pipeline"):
        alert_placeholder.info(f"The pipeline {pipeline_to_run} is running. Please do not navigate away from this page.")
        pipeline_progress_bar = stqdm(total=2, desc="Submitting pipeline...", position=0)
        #process_progress_bar = stqdm(total=2, desc="Waiting...", position=0)
        process_progress_bar = None
        process_status_box = st.status("Submitting pipeline step...", expanded=True)
        #pipeline_progress_bar_slot = st.empty()
        #process_progress_bar_slot = st.empty()
        with st.container():
            st.subheader("Pipeline Logs")
            errbox = st.expander("Error messages", expanded=False)
            with st.expander("View all pipeline logs"):
                with st.container():
                    log_committed_box = st.empty()
            with st.expander("View current step live logs"):
                with st.container():
                    log_live_box = st.empty()


        log = stlogbox.StreamlitJobLogger(log_committed_box, log_live_box)
        execution_mode = 'local'
        if st.session_state.has_cloud_session:
            execution_mode = 'cloud'
        local_path_remapping = {}
        data_dir_locally = st.session_state.paths["out_dir"]
        data_dir_on_host = st.session_state.paths["host_out_dir"]
        if data_dir_on_host is not None:
            local_path_remapping[data_dir_locally] = data_dir_on_host
        try:
            result = utiltl.run_pipeline(
                pipeline_id=pipeline_to_run, ##TODO EDIT THIS
                global_vars={"STUDY": st.session_state.paths["project"]},
                execution_mode=execution_mode,
                pipeline_progress_bar=pipeline_progress_bar,
                process_progress_bar=process_progress_bar,
                process_status_box=process_status_box,
                log=log,
                metadata_location=os.path.join(st.session_state.paths["project"], "metadata.json"),
                reuse_cached_steps=skip_steps_when_possible,
                local_path_remapping=local_path_remapping
            )

            alert_placeholder.success(f"Pipeline {pipeline_to_run} finished successfully.")
        except Exception as err: 
            alert_placeholder.error(f"Pipeline {pipeline_to_run} failed with errors. Expand the log boxes for details.")
            process_status_box.update(state="error", label="Pipeline run failed.", expanded=False)
            errbox.error(traceback.format_exc())



    pass

def pipeline_menu():
    #cols = st.columns([10,1,10])
    cols = st.columns(2)
    out_dir = os.path.join(
        st.session_state.paths['out_dir'], st.session_state['prj_name']
    )

    pipelines = st.session_state.pipelines
    pnames = pipelines.Name.tolist()

    enabled_pnames = []
    disabled_pnames = []
    # Evaluate suitability for current data, filter accordingly
    for pname in pnames:
        result, blockers = utiltl.check_requirements_met_nopanel(pname, st.session_state.do_harmonize)
        if result:
            enabled_pnames.append(pname)
        else:
            disabled_pnames.append(pname)

    with cols[0]:
        sel = select_pipeline(enabled_pnames=enabled_pnames)
    with cols[1]:
        pipeline_runner_menu(enabled_pnames=enabled_pnames, sel=sel)


def panel_pipelines():

    workflow = st.session_state.workflow

    if workflow is None:
        st.info('Please select a Workflow!')
        return

    with st.container(horizontal=True, horizontal_alignment="center"):
        st.markdown("<h4 style=color:#3a3a88;'>Select and Run Pipeline\n\n</h1>", unsafe_allow_html=True, width='content')

    if st.session_state.workflow == 'ref_data':
        st.info('''
            You’ve selected the **Reference Data** workflow. This option doesn’t require pipeline selection.
            - If you meant to analyze your data, please go back and choose a different workflow.
            - Otherwise, continue to the next step to explore the reference values.
            '''
        )
    else:
        pipeline_menu()

