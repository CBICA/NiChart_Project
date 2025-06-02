import streamlit as st
import os
import yaml
from pathlib import Path
from graphviz import Digraph
from collections import defaultdict
import utils.utils_pages as utilpg
import utils.utils_processes as utilprc
import utils.utils_session as utilses

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

st.markdown(
    """
    ### Configure and run a pipeline
    """
)

if st.session_state.sel_pipeline is None:
    st.warning('Please select a pipeline and upload your data first!')
else:
    st.success(f'Selected pipeline: {st.session_state.sel_pipeline}')
    list_tasks = ["Configure", "Run"]
    sel_task = st.pills(
        "Select Task",
        list_tasks,
        selection_mode="single",
        label_visibility="collapsed"
    )
    if sel_task == "Configure":
        panel_conf_pipeline()
        
    elif sel_task == "Run":
        panel_run_pipeline()



