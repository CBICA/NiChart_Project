import os
import shutil
import time
from typing import Any

import pandas as pd
import numpy as np
import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform
from scipy import ndimage
import utils.utils_misc as utilmisc
import utils.utils_user_select as utiluser
import utils.utils_io as utilio

import utils.utils_session as utilses
import gui.utils_plots as utilpl
import gui.utils_mriview as utilmri
import gui.utils_view as utilview
import pandas as pd

import streamlit_antd_components as sac

import streamlit as st
from stqdm import stqdm

from utils.utils_logger import setup_logger
logger = setup_logger()

def view_dlmuse_volumes(layout):
    """
    View dlmuse volumes
    """
    ## FIXME (list of rois from data file to init listbox selections)
    fname=os.path.join(
        st.session_state.paths['resources'], 'reference_data', 'centiles', 'dlmuse_centiles_CN.csv'
    ) 
    df = pd.read_csv(fname)
    list_vars = ['Age', 'Sex'] + df.VarName.unique().tolist()

    var_groups_data = ['roi']
    pipeline = 'dlmuse'

    # Set centile selections
    st.session_state.plot_params['centile_values'] = st.session_state.plot_settings['centile_trace_types']

    with layout:
        sac.divider(label='Plot Controls', align='center', color='gray')

        utilpl.user_add_plots(
            st.session_state.plot_params
        )

        sac.divider(label='Plot Settings', align='center', color='gray')

        utilpl.panel_set_params_centile_plot(
            st.session_state.plot_params, var_groups_data, pipeline, list_vars
        )

    utilpl.panel_show_plots()

    st.write()


def view_dlmuse_segmentation(layout):
    """
    View dlmuse segmentations
    """
    fname=os.path.join(
        st.session_state.paths['resources'], 'reference_data', 'centiles', 'dlmuse_centiles_CN.csv'
    ) 
    df = pd.read_csv(fname)
    list_vars = ['Age', 'Sex'] + df.VarName.unique().tolist()
    
    ulay = st.session_state.ref_data["t1"]
    olay = st.session_state.ref_data["dlmuse"]        

    with layout:
        sac.divider(label='Viewing Options', align='center', color='gray')
        
    # Set params
    utilmri.panel_set_params(st.session_state.plot_params, ['roi'], 'muse', list_vars)

    # Show figures
    utilmri.panel_view_seg(ulay, olay, st.session_state.plot_params)

def select_task(layout):
    with layout:
        sel_task = st.selectbox(
            'Task:',
            ['Download Results', 'View Results'],
            index=0
        )
    return sel_task


def select_main_data(layout):
    with layout:
        sel_mdata = st.selectbox(
            'Data:',
            ['None', 'Current Project', 'Sample Study 1', 'Sample Study 2'],
            index=0
        )
    return sel_mdata

def select_ref_data(layout):
    with layout:
        sel_rdata = st.selectbox(
            'Reference data:',
            ['None', 'CN', 'CN Females', 'CN Males'],
            index=0
        )
    return sel_rdata

def select_pipeline(layout):
    with layout:
        sel_pipe = st.selectbox(
            'Pipeline:',
            ['dlmuse', 'dlwmls'],
            index=0
        )
    return sel_pipe

def select_dtype(layout):
    with layout:
        sel_dtype = st.selectbox(
            'Result type:',
            ['ROI Volumes', 'Segmentation'],
            index=0
        )
    return sel_dtype

def panel_download():
    '''
    Panel to download results
    '''
    with st.container(horizontal=True, horizontal_alignment="center"):

        st.markdown(f"###### 📁 Project Folder:   `{st.session_state.prj_name}`", width='content')
    
        prj_dir = st.session_state.paths['prj_dir']
        list_dirs = utilio.get_subfolders(prj_dir)
        for folder_name in ['downloads', 'user_upload']:
            if folder_name in list_dirs:
                list_dirs.remove(folder_name)
        
        if len(list_dirs) == 0:
            return
        
        sel_opt = sac.checkbox(
            list_dirs,
            label='Folder(s) to download:', align='center', 
            color='#aaeeaa', size='xl',
            check_all='Select all'
        )

        if sel_opt is None or len(sel_opt)==0:
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

def panel_user_data(layout):

    logger.debug('    Function: panel_ref_data')

    with layout:
        sac.divider(label='Data Files', align='center', color='gray')

    sel_task = select_task(layout)
    if sel_task == 'Download Results':
        panel_download()
    elif sel_task == 'View Results':
        sel_pipe = select_pipeline(layout)
        if sel_pipe == 'dlmuse': 
            sel_dtype = select_dtype(layout)
            if sel_dtype == 'ROI Volumes':
                view_dlmuse_volumes(layout)
            elif sel_dtype == 'Segmentation':
                view_dlmuse_segmentation(layout)
        
def panel_ref_data(layout):

    logger.debug('    Function: panel_ref_data')

    with layout:
        sac.divider(label='Data Files', align='center', color='gray')
        
    sel_mdata = select_main_data(layout)

    sel_rdata = select_ref_data(layout)
    if sel_rdata != 'None':
        fname = os.path.join(
            st.session_state.paths['centiles'],
            'dlmuse_centiles_CN.csv'
        )
        st.session_state.plot_data['df_cent'] = utilio.read_csv(fname)

    sel_pipe = select_pipeline(layout)
    if sel_pipe == 'dlmuse':
        view_dlmuse_volumes(layout)

    elif sel_pipe == 'dlwmls':
        st.warning('Viewer not implemented for dlwmls')


