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
    'Debug' : 'pages/Debug.py',
}

def select_main_menu():
    '''
    Select main menu page from a list and switch to it
    '''
    with st.sidebar:
        sel_main_menu = st.pills(
            'Select Main Menu',
            dict_main_menu.keys(),
            selection_mode='single',
            default = st.session_state.sel_main_menu,
            label_visibility="collapsed"
        )

        if sel_main_menu is None:
            return

        if sel_main_menu == st.session_state.sel_main_menu:
            return

        st.session_state.sel_main_menu = sel_main_menu
        sel_page = dict_main_menu[sel_main_menu]
        st.switch_page(sel_page)


dict_pipelines = {
    'DLMUSE Biomarkers (T1)' : 'pages/p_DLMUSE_Biomarkers.py',
    'DLWMLS (FL)' : 'pages/p_DLWMLS.py',
    'DTI Biomarkers (DTI)' : 'pages/p_DTI.py',
    'rsfMRI Biomarkers (rsfMRI)' : 'pages/p_rsfMRI.py',
}

def select_pipeline():
    '''
    Select pipeline from a list and switch to pipeline page
    '''
    with st.sidebar:
        with st.container(border=True):
            #st.markdown('##### ')
            st.markdown('### Pipeline:')
            sel_pipeline = st.pills(
                'Pipelines',
                dict_pipelines.keys(),
                selection_mode='single',
                default = st.session_state.sel_pipeline,
                label_visibility="collapsed"
            )
            if sel_pipeline is None:
                return
            if sel_pipeline == st.session_state.sel_pipeline:
                return
            st.session_state.sel_pipeline = sel_pipeline
            sel_page = dict_pipelines[sel_pipeline]
            st.switch_page(sel_page)

dict_pipeline_steps = {
    'dlmuse_biomarkers': {
        'Overview' : 'pages/dlmuse_biomarkers_overview.py',
        'DLMUSE' : 'pages/process_DLMUSE.py',
        'ML Biomarkers' : 'pages/workflow_sMRI_MLScores.py',
        'Plotting' : 'pages/plot_sMRI_vars_study.py'
    },
}

def select_pipeline_step():
    '''
    Select pipeline step from a list and switch page
    '''
    sel_dict = dict_pipeline_steps['dlmuse_biomarkers']
    with st.sidebar:
        with st.container(border=True):
            st.markdown('### Pipeline step:')            
            sel_step = st.pills(
                'DLMUSE Biomarkers steps',
                sel_dict.keys(),
                selection_mode='single',
                default = st.session_state.sel_pipeline_step,                
                label_visibility="collapsed"
            )
            if sel_step is None:
                return
            if sel_step == st.session_state.sel_pipeline_step:
                return
            st.session_state.sel_pipeline_step = sel_step
            sel_page = sel_dict[sel_step]
            st.switch_page(sel_page)

def config_page() -> None:
    st.session_state.nicon = Image.open("../resources/nichart1.png")
    st.set_page_config(
        page_title="NiChart",
        page_icon=st.session_state.nicon,
        layout="wide",
        # layout="centered",
        menu_items={
            "Get help": "https://neuroimagingchart.com/",
            "Report a bug": "https://github.com/CBICA/NiChart_Project/issues/new?assignees=&labels=&projects=&template=bug_report.md&title=%5BBUG%5D+",
            "About": "https://neuroimagingchart.com/",
        },
    )

