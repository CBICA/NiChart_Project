import streamlit as st
import os
import yaml
from pathlib import Path
from graphviz import Digraph
from collections import defaultdict
import utils.utils_misc as utilmisc
import utils.utils_pages as utilpg
import utils.utils_processes as utilprc
import utils.utils_io as utilio
import utils.utils_session as utilses
from streamlit_image_select import image_select
import re
from utils.utils_logger import setup_logger

import streamlit_antd_components as sac

logger = setup_logger()
logger.debug('Page: Multi Subject Analysis')

# Page config should be called for each page
utilpg.config_page()
utilpg.set_global_style()

# Set data type
st.session_state.data_type = 'multi_subject'

if 'instantiated' not in st.session_state or not st.session_state.instantiated:
    utilses.init_session_state()

def view_overview():
    with st.container(border=True):
        st.markdown(
            '''
            
            - Welcome! This is where you can upload a study dataset with MRI scans of multiple subjects (DICOM or NIfTI format). Neuroimaging chart values for the sample will be calculated and displayed in comparison to reference distributions.
            
            - Image analysis pipelines require a set of files to be used as input, like the mri scans of the subjects (e.g. T1, FL) or a csv file with basic demographics for all subjects (Age, Sex, etc). For specifics of required data for each pipeline you can browse each pipeline in the pipelines tab and follow instructions in data upload tab.
            
            ''', unsafe_allow_html=True
        )

def upload_data():

    if st.button('Select Project'):
        out_dir = st.session_state.paths["out_dir"]
        sel_project = utilio.panel_select_project(out_dir, st.session_state.project)
        st.success(f'Project Name: {st.session_state.project}')
        
    with st.container(border=True):    
        utilio.panel_load_data()
        
     
    if st.button('View Project'):
        in_dir = st.session_state.paths['project']
        utildv.data_overview(in_dir)


    if st.button('Delete Project'):
        with st.container(border=True):
            st.success(f'Project Name: {st.session_state.project}')
            proj_dir = st.session_state.paths['project']
            if st.button("Delete this project", help="This will permanently delete all data in this project and invalidate all associated caching."):
                st.warning("Are you sure you want to delete all data associated with this project? This cannot be undone.")
                if st.button("Confirm deletion"):
                    shutil.rmtree(st.session_state.paths['project'])
                    st.success(f"Project {st.session_state.project} was successfully deleted.")
                    list_projects = utilio.get_subfolders(st.session_state.paths["out_dir"])
                    if len(list_projects) > 0:
                        sel_project = list_projects[0]
                        utilss.update_project(sel_project)

    # Show selections
    #utilses.disp_selections()
     

def select_pipeline():
    st.info('Work in progress!')

def view_results():
    st.info('Work in progress!')

def download_results():
    st.info('Work in progress!')

st.markdown("<h5 style='text-align:center; color:#3a3a88;'>Multi-Subject Analysis\n\n</h1>", unsafe_allow_html=True)


sel = sac.tabs([
    sac.TabsItem(label='Overview'),
    sac.TabsItem(label='Upload Data'),
    sac.TabsItem(label='Select Pipeline'),
    sac.TabsItem(label='View Results'),
    sac.TabsItem(label='Download Results'),
    sac.TabsItem(label='Go Back Home'),
], align='center',  size='xl', color='grape')

if sel == 'Overview':
    view_overview()
    
if sel == 'Upload Data':
    upload_data()

if sel == 'Select Pipeline':
    select_pipeline()
    
if sel == 'View Results':
    view_results()

if sel == 'Download Results':
    download_results()

if sel == 'Go Back Home':
    st.switch_page("pages/nichart_home.py")


# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()



