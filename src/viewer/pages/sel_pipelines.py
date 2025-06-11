import streamlit as st
import os
import yaml
from pathlib import Path
from graphviz import Digraph
from collections import defaultdict
import utils.utils_pages as utilpg
import utils.utils_processes as utilprc
import utils.utils_session as utilses
from streamlit_image_select import image_select
import re

def view_input_data(method) -> None:
    """
    Panel for viewing input data for a pipeline
    """
    with st.container(border=True):
        fdoc = os.path.join(
            st.session_state.paths['resources'],
            'pipelines',
            method,
            'data_' + method + '.md'
        )
        with open(fdoc, 'r') as f:
            markdown_content = f.read()
        #st.markdown(markdown_content)
        
        parts = re.split(r"```", markdown_content)

        st.markdown(parts[0])
        st.code(parts[1].strip(), language="text")


def sel_pipeline_from_list():
    with st.container(border=True):
                
        # Show a thumbnail image for each pipeline
        pdict = dict(
            zip(st.session_state.pipelines['Name'], st.session_state.pipelines['Label'])
        )
        pdir = os.path.join(st.session_state.paths['resources'], 'pipelines')
        logo_fnames = [
            os.path.join(pdir, pname, f'logo_{pname}.png') for pname in list(pdict.values())
        ]
        psel = image_select(
            "",
            images = logo_fnames,
            captions=list(pdict.keys()),
            index=-1,
            return_value="index",
            use_container_width = False
        )
        
        # Show description of the selected pipeline
        if psel < 0 :
            return
        
        sel_pipeline = list(pdict.values())[psel]
        if st.button('Select'):
            st.session_state.sel_pipeline = sel_pipeline
            st.success(f'Pipeline selected {sel_pipeline}')
            view_input_data(sel_pipeline)

            

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

list_tasks = ["From List", "Advanced"]
sel_task = st.pills(
    "Select Workflow Task", list_tasks, selection_mode="single", label_visibility="collapsed"
)
if sel_task == "From List":
    sel_pipeline_from_list()

elif sel_task == "Advanced":
    st.warning('Not implemented yet!')
    #sel_pipeline_from_graph()
    



