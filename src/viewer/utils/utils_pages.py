import os
import shutil
from typing import Any

import jwt
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
import utils.utils_session as utilss
import utils.utils_check_files as utilcf

# from streamlit.web.server.websocket_headers import _get_websocket_headers

dict_main_menu = {
    'Home' : 'pages/Home.py',
    'Pipelines' : 'pages/Pipelines.py',
    'Config' : 'pages/Config.py',
}

dict_pipelines_std = {
    'Consolidate' : 'pages/study_consolidate.py',
    'QC' : 'pages/study_qc.py',
    'Devtools' : 'pages/study_devtools.py',
}

dict_pipelines_pooled = {
    'Prep' : 'pages/pooled_prep.py',
    'Harmonize' : 'pages/pooled_harmonize.py',
    'QC' : 'pages/pooled_qc.py',
}

def select_pipeline_std():
    '''
    Select pipeline from a list ant switch to pipeline page
    '''
    st.markdown('##### Single Study')
    sel_pipeline = st.pills(
        'Single Study',
        dict_pipelines_std.keys(),
        selection_mode='single',
        label_visibility="collapsed"
    )
    if sel_pipeline is None:
        return
    if sel_pipeline == st.session_state.sel_pipeline:
        return
    sel_page = dict_pipelines_std[sel_pipeline]
    st.session_state.sel_pipeline = sel_page
    st.switch_page(sel_page)

def select_pipeline_pooled():
    '''
    Select pipeline from a list ant switch to pipeline page
    '''
    st.markdown('##### Pooled Data')
    sel_pipeline = st.pills(
        'Pooled Data',
        dict_pipelines_pooled.keys(),
        selection_mode='single',
        label_visibility="collapsed"
    )
    if sel_pipeline is None:
        return
    if sel_pipeline == st.session_state.sel_pipeline:
        return
    sel_page = dict_pipelines_pooled[sel_pipeline]
    st.session_state.sel_pipeline = sel_page
    st.switch_page(sel_page)

def select_main_menu(sel_item):
    '''
    Select main menu page from a list ant switch to it
    '''
    sel_main_menu = st.pills(
        'Select Main Menu',
        dict_main_menu.keys(),
        selection_mode='single',
        default = sel_item,
        label_visibility="collapsed"
    )

    if sel_main_menu is None:
        return

    if sel_main_menu == sel_item:
        return

    sel_page = dict_main_menu[sel_main_menu]
    st.session_state.sel_main_menu = sel_page
    st.switch_page(sel_page)

def config_page() -> None:
    #st.session_state.nicon = Image.open("../resources/istaging1.png")
    st.set_page_config(
        page_title="ISTAGING",
        #   page_icon=st.session_state.nicon,
        layout="wide",
        # layout="centered",
        menu_items={
            "Get help": "https://istaging.com/",
            "Report a bug": "https://github.com/CBICA/I-Staging/issues/new?assignees=&labels=&projects=&template=bug_report.md&title=%5BBUG%5D+",
            "About": "https://istaging.com/",
        },
    )

