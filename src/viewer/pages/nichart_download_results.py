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
import utils.utils_io as utilio
import utils.utils_data_view as utildv
from utils.utils_styles import inject_global_css 

from streamlit_image_select import image_select
import re
from utils.utils_logger import setup_logger

import streamlit_antd_components as sac

logger = setup_logger()
logger.debug('Page: Select Pipelines')

inject_global_css()

# Page config should be called for each page
utilpg.config_page()
utilpg.set_global_style()


if 'instantiated' not in st.session_state or not st.session_state.instantiated:
    utilses.init_session_state()

def panel_download():
    '''
    Panel to download results
    '''
    prj_dir = st.session_state.paths['prj_dir']
    list_dirs = utilio.get_subfolders(prj_dir)
    for folder_name in ['downloads', 'user_upload']:
        if folder_name in list_dirs:
            list_dirs.remove(folder_name)
    
    if len(list_dirs) == 0:
        return
    
    sel_opt = sac.checkbox(
        list_dirs,
        label='Select a folder:', align='center', 
        color='#aaeeaa', size='xl',
        check_all='Select all'
    )

    if sel_opt is None:
        return

    with st.container(horizontal=True, horizontal_alignment="center"):
        out_dir = os.path.join(prj_dir, 'downloads')
        os.makedirs(out_dir, exist_ok=True)
        out_zip = os.path.join(out_dir, 'nichart_results.zip')

        if st.button('Prepare Data'):
            utilio.zip_folders(prj_dir, sel_opt, out_zip)
            with open(out_zip, "rb") as f:
                file_download = f.read()            
            st.toast('Created zip file with selected folders')

            flag_download = os.path.exists(out_zip)
            st.download_button(f"Download", file_download, 'nichart_results.zip')
            os.remove(out_zip)
            st.toast('File downloaded')
    
     
st.markdown("<h5 style='text-align:center; color:#3a3a88;'>Download Results\n\n</h1>", unsafe_allow_html=True)

panel_download()

sel_but = sac.chip(
    [
        sac.ChipItem(label = '', icon='arrow-left', disabled=False),
        sac.ChipItem(label = '', icon='arrow-right', disabled=False)
    ],
    label='', align='center', color='#aaeeaa', size='xl', return_index=True
)
    
if sel_but == 0:
    st.switch_page("pages/nichart_run_pipeline.py")

if sel_but == 1:
    st.switch_page("pages/nichart_view_results.py")

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()



