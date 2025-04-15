import streamlit as st
import os
import yaml
from pathlib import Path
from graphviz import Digraph
from collections import defaultdict
import utils.utils_pages as utilpg
import utils.utils_processes as utilprc
import utils.utils_session as utilses

def panel_sel_pipeline():
    with st.container(border=True):
        proc = st.session_state.processes

        st.markdown("##### Select Input Data:")
        sel_inputs = st.pills(
            "Select input:",
            proc['in_files'],
            selection_mode="multi",
            label_visibility="collapsed"
        )
        flag_input_all = st.checkbox(
            "Require all selected inputs",
            value=True,
            help=(
                "If checked, only include steps that depend on **all** selected inputs.\n\n"
                "If unchecked, include steps that depend on **any one** of the selected inputs."
            )
        )
        if len(sel_inputs) == 0:
            return
        
        sel_steps = utilprc.detect_reachable_steps(
            proc['data'], sel_inputs, flag_input_all
        )
        
        st.markdown("##### Select Pipeline Steps:")
        sel_steps = st.pills(
            "Select output:",
            sel_steps,
            selection_mode="multi",
            default = sel_steps,
            label_visibility="collapsed"
        )
        
        if len(sel_steps) > 0:
            
            flag_show_graph = st.checkbox(
                'Show Process Graph?',
                value=True
            )
            
            if flag_show_graph:
                graph = utilprc.build_proc_graph(
                    proc['data'], sel_steps, sel_inputs
                )
                st.graphviz_chart(graph.source)

            if st.button('Confirm'):
                st.success('Pipeline Selected!')

def panel_conf_pipeline():
    with st.container(border=True):
        st.markdown(
            """
            ### Work in prog ...
            """
        )

def panel_run_pipeline():
    with st.container(border=True):
        st.markdown(
            """
            ### Work in prog ...
            """
        )


# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()

st.markdown(
    """
    ### Select and run pipeline
    """
)

list_tasks = ["Select", "Configure", "Run"]
sel_task = st.pills(
    "Select Workflow Task", list_tasks, selection_mode="single", label_visibility="collapsed"
)
if sel_task == "Select":
    panel_sel_pipeline()
elif sel_task == "Configure":
    panel_conf_pipeline()
elif sel_task == "Run":
    panel_run_pipeline()

