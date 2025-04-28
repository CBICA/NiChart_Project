import streamlit as st
import os
import yaml
from pathlib import Path
from graphviz import Digraph
from collections import defaultdict
import utils.utils_pages as utilpg
import utils.utils_processes as utilprc
import utils.utils_session as utilses

def sel_pipeline_from_list():
    with st.container(border=True):
                
        # Read process data
        proc = st.session_state.processes

        # Let users select input files
        st.markdown("##### Filter tags:")
        sel_inputs = st.pills(
            "Select proc input:",
            proc['in_files'],
            selection_mode="multi",
            label_visibility="collapsed",
        )

        # Detect steps that can be run using the given input files
        reachable_steps = utilprc.detect_reachable_steps(
            proc['graph'], sel_inputs, False
        )

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


def sel_pipeline_from_graph():
    with st.container(border=True):
                
        # Read process data
        proc = st.session_state.processes

        # Let users select input files
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

st.info(
    """
    ### Pipelines
    - Choose from a variety of image processing and analysis pipelines.
    - Pipeline steps include image processing, feature extraction, and machine learning models
    - ***:red[List view:]*** Select pipelines by name, tag or output.
    - ***:red[Graph view:]*** Select pipelines based on their dependencies within the overall pipeline network.    
    - Once the processing is complete, go to the ***:red[Results]*** page for reports and visualizations.
    """
)

list_tasks = ["Select", "Configure", "Run"]
sel_task = st.pills(
    "Select Workflow Task", list_tasks, selection_mode="single", label_visibility="collapsed"
)
if sel_task == "Select":
    
    list_views = ["List View", "Graph View"]
    sel_view = st.pills(
        "Select View", list_views, selection_mode="single", label_visibility="collapsed"
    )
    if sel_view == "List View":
        sel_pipeline_from_list()
    elif sel_view == "Graph View":
        sel_pipeline_from_graph()

elif sel_task == "Configure":
    panel_conf_pipeline()
    
elif sel_task == "Run":
    panel_run_pipeline()



