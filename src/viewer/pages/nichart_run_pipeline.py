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
import utils.utils_toolloader as utiltl
import utils.utils_stlogbox as stlogbox

from streamlit_image_select import image_select
from stqdm import stqdm
import re
from utils.utils_logger import setup_logger

import streamlit_antd_components as sac

logger = setup_logger()
logger.debug("--- STARTING: Run Pipelines ---")

inject_global_css()

# Page config should be called for each page
#utilpg.config_page()
utilpg.set_global_style()


if 'instantiated' not in st.session_state or not st.session_state.instantiated:
    utilses.init_session_state()


#################################
## Function definitions
def help_message(data_type):

    with st.popover("❓", width='content'):
        st.write(
            """
            **How to Use This Page**

            - Use the left panel to select a pipeline and view the description. (By default, we'll only show you pipelines for which your data meets prerequisites.)
            - Use the right panel to run the pipeline. You can select to harmonize the results to the reference data, if you have a sufficient number of data points.
            - If you select a pipeline which requires more data, the description of the missing files or fields will appear on the right panel.
            """
        )

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

def select_pipeline():
    '''
    Select a pipeline and show overview
    '''
    st.markdown("##### Select:")
    show_enabled_only = st.checkbox("Show only pipelines which match my available data", value=True)
    sac.divider(key='_p2_div1')

    pipelines = st.session_state.pipelines
    pnames = pipelines.Name.tolist()

    enabled_pnames = []
    disabled_pnames = []
    # Evaluate suitability for current data, filter accordingly
    for pname in pnames:
        result, blockers = utiltl.check_requirements_met_nopanel(pname)
        if result:
            enabled_pnames.append(pname)
        else:
            disabled_pnames.append(pname)
    
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
    st.session_state.sel_pipeline = sel_label
    return sel_label

def pipeline_runner_menu(sel=False):
    st.markdown("##### Run:")
    sac.divider(key='_p2_div2')
    if not sel:
        st.info("Select a pipeline on the left, then look here to run it.")
    sel_method = st.session_state.sel_pipeline
    st.success(f'Selected pipeline: {sel_method}')
    harmonize = False
    if 'subject_type' not in st.session_state or st.session_state.subject_type == 'multi':
        if utiltl.pipeline_is_harmonizable(sel_method):
            harmonize = st.checkbox("Harmonize to reference data? (Requires >= 30 scans)")
    st.session_state.do_harmonize = harmonize
    ## TODO: Retrieve dynamically/match between front end and toolloader code
    ## This a nice and simple placeholder for now
    
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

    pass

def pipeline_menu():
    #cols = st.columns([10,1,10])
    cols = st.columns(2)
    out_dir = os.path.join(
        st.session_state.paths['out_dir'], st.session_state['prj_name']
    )

    with cols[0]:
        sel = select_pipeline()
    with cols[1]:
        pipeline_runner_menu(sel)

#################################
## Main

data_type = st.session_state.data_type

with st.container(horizontal=True, horizontal_alignment="center"):
    st.markdown("<h4 style=color:#3a3a88;'>Select and Run Pipeline\n\n</h1>", unsafe_allow_html=True, width='content')
    help_message(data_type)

pipeline_menu()

sel_but = sac.chip(
    [
        sac.ChipItem(label = '', icon='arrow-left', disabled=False),
        sac.ChipItem(label = '', icon='arrow-right', disabled=False)
    ],
    label='', align='center', color='#aaeeaa', size='xl', return_index=True
)

if sel_but == 0:
    st.switch_page("pages/nichart_upload_data.py")

if sel_but == 1:
    st.switch_page("pages/nichart_download_results.py")

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()



