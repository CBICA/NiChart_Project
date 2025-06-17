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
        
        # Read process data
        proc = st.session_state.processes

        # Let user select input files
        st.markdown("##### Select Input Data:")
        sel_inputs = st.pills(
            "Select proc input:",
            proc['in_files'],
            selection_mode="multi",
            label_visibility="collapsed",
        )

        flag_input_all = st.checkbox(
            "Require all selected inputs",
            help=(
                "If checked, only include steps that depend on **all** selected inputs.\n\n"
                "If unchecked, include steps that depend on **any one** of the selected inputs."
            ),
            value = True
        )
        if len(sel_inputs) == 0:
            return
        
        # Detect steps that can be run using the given input files
        reachable_steps = utilprc.detect_reachable_steps(
            proc['graph'], sel_inputs, flag_input_all
        )
        #list_sel_steps = utilprc.find_disconnected_pipelines(proc['steps'], sel_steps)

        # Sort steps (top down order in precess tree)
        reachable_steps = utilprc.topological_sort(
            proc['steps'], reachable_steps
        )
        
        # Let user select steps
        st.markdown("##### Select Pipeline Steps:")
        sel_steps = st.pills(
            "Select output:",
            reachable_steps,
            selection_mode="multi",
            label_visibility="collapsed",
            default = reachable_steps
        )
        
        if len(sel_steps) > 0:
            
            flag_show_graph = st.checkbox(
                'Show Process Graph?',
                value = True
            )

            graph, req_steps = utilprc.build_proc_graph(
                proc['steps'], sel_steps, sel_inputs
            )

            if flag_show_graph:
                st.graphviz_chart(graph.source)

            if st.button('Confirm'):
                sel_steps = utilprc.topological_sort(
                    proc['steps'], req_steps
                )
                st.session_state.processes['sel_inputs'] = sel_inputs
                st.session_state.processes['sel_steps'] = sel_steps
                st.success('Pipeline Selected!')
                st.markdown('##### Pipeline steps:')
                st.write(sel_steps)

def panel_conf_pipeline():
    with st.container(border=True):
        st.markdown(
            """
            ##### Select args for each step
            """
        )
        for sel_step in st.session_state.processes['sel_steps']:
            with st.expander(sel_step, expanded=True):
                for sel_arg in ['arg1', 'arg2', 'arg3']:
                    st.selectbox(
                        f'Select {sel_arg}',
                        [1,2,3],
                        0,
                        key = f'_key_selbox_{sel_step}_{sel_arg}',
                    )
                st.button(
                    'Confirm',
                    key = f'_key_btn_confirm_{sel_step}'
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

