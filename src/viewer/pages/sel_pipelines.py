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
                
        # Read pipelines
        pnames = st.session_state.pipelines.Name.tolist()

        # Let user select steps
        st.markdown("##### Select Pipeline:")
        sel_pipeline = st.pills(
            "Select pipeline:",
            pnames,
            selection_mode="single",
            label_visibility="collapsed",
            default = None
        )
        
    if sel_pipeline is None:
        return
    
    with st.container(border=True):
        st.markdown(f'Current selection: {sel_pipeline}')
        if st.button('Select'):
            st.session_state.sel_pipeline = sel_pipeline
            st.success(f'Pipeline selected {sel_pipeline}')
        
        
            
        

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
    ### Pipelines
    """
)

with st.expander('Select Pipeline', expanded=True):

    list_tasks = ["Simple", "Advanced"]
    sel_task = st.pills(
        "Select Workflow Task", list_tasks, selection_mode="single", label_visibility="collapsed"
    )
    if sel_task == "Simple":
        sel_pipeline_from_list()

    elif sel_task == "Advanced":
        st.warning('Not implemented yet!')
        #sel_pipeline_from_graph()
    



