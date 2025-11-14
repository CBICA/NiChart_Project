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

def view_dlmuse() -> None:
    """
    Panel for viewing dlmuse results
    """
    with st.sidebar:
        sel_task = st.selectbox(
            'Data type:',
            ['ROI Volumes', 'Segmentation'],
            index=0
        )

    ## FIXME (list of rois from data file to init listbox selections)
    df = pd.read_csv(
            os.path.join(
                st.session_state.paths['resources'],
                'reference_data', 'centiles', 'dlmuse_centiles_CN.csv' 
            )
    )
    list_vars = ['Age', 'Sex'] + df.VarName.unique().tolist()


    if sel_task == 'ROI Volumes':
        var_groups_data = ['roi']
        pipeline = 'dlmuse'

        # Set centile selections
        st.session_state.plot_params['centile_values'] = st.session_state.plot_settings['centile_trace_types']

        with st.sidebar:
            sac.divider(label='Viewing Options', align='center', color='gray')
            utilpl.user_add_plots(
                st.session_state.plot_params
            )
            
        utilpl.sidebar_flag_hide_setting()
        utilpl.sidebar_flag_hide_legend()

        utilpl.panel_set_params_centile_plot(
            st.session_state.plot_params,
            var_groups_data,
            pipeline,
            list_vars
        )
        utilpl.panel_show_centile_plots()

        st.write()

    elif sel_task == 'Segmentation':
        ulay = st.session_state.ref_data["t1"]
        olay = st.session_state.ref_data["dlmuse"]        

        with st.sidebar:
            sac.divider(label='Viewing Options', align='center', color='gray')
        utilpl.sidebar_flag_hide_setting()

        utilmri.panel_set_params(
            st.session_state.plot_params,
            ['roi'],
            'muse',
            list_vars
        )

        utilmri.panel_view_seg(
            ulay, olay, st.session_state.plot_params
        )

def panel_user_data_view():
    # Show results
    with st.sidebar:
        sel_pipe = st.selectbox(
            'Pipeline',
            ['dlmuse', 'dlwmls'],
            index=0
        )
    
    if sel_pipe == 'dlmuse':
        view_dlmuse()

    elif sel_pipe == 'dlwmls':
        st.warning('Viewer not implemented for dlwmls')
        #view_dlwmls()


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


    with layout:
        sel_tab = sac.tabs(
            items=[
                sac.TabsItem(label='Download'),
                sac.TabsItem(label='View'),
            ],
            size='sm',
            align='left'
        )
    
    if sel_tab == 'Download':
        panel_download()

    elif sel_tab == 'View':
        panel_user_data_view()
        
def panel_ref_data():
    #st.info(
    panel_user_data_view()
        
    # Show selections
    utilses.disp_selections()


