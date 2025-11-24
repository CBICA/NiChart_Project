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
import gui.utils_widgets as utilwd

import streamlit_antd_components as sac

import streamlit as st
from stqdm import stqdm

from utils.utils_logger import setup_logger
logger = setup_logger()

def safe_index(lst, value, default=None):
    try:
        return lst.index(value)
    except ValueError:
        return default

def select_from_list(layout, list_opts, var_name, hdr):
    '''
    Generic selection box 
    For a variable (var_name) initiated with the given list (list_opts)
    Variable is saved in session_state (used as the key for the select box)
    '''
    sel_ind = safe_index(list_opts, st.session_state.get(var_name))
    with layout:
        sel_opt = st.selectbox(hdr, list_opts, key=var_name, index=sel_ind)
    return st.session_state[var_name]

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

        #utilpl.panel_set_params_centile_plot(
            #st.session_state.plot_params, var_groups_data, pipeline, list_vars
        #)
        utilpl.set_plot_params()

    utilpl.panel_show_plots()

    st.write()


def view_segmentation(layout, pipeline):
    """
    View segmentations
    """
    img_views = ["axial", "coronal", "sagittal"]

    with layout:
        sac.divider(label='Data', align='center', color='gray')

    if pipeline == 'dlmuse':

        fname = os.path.join(
            st.session_state.paths['project'], 'dlmuse', 'dlmuse_vol.csv'
        )
        df = pd.read_csv(fname)

        # Rename columns if dict for data exists
        df = df.rename(columns = st.session_state.dicts['muse']['ind_to_name'])

        list_vars = df.columns.unique().tolist()
        list_mrids = df.MRID.sort_values().tolist()
        
        with layout:
            sel_mrid = utilwd.my_selectbox(list_mrids, 'res_sel_mrid', 'Subject')

        if sel_mrid is None:
            return

        #######################
        ## Set olay ulay images
        #ulay = st.session_state.ref_data["t1"]
        #olay = st.session_state.ref_data["dlmuse"]
        ulay = os.path.join(
            st.session_state.paths['project'], 't1', f'{sel_mrid}_T1.nii.gz'
        )
        olay = os.path.join(
            st.session_state.paths['project'], 'dlmuse', f'{sel_mrid}_T1_DLMUSE.nii.gz'
        )
        if not os.path.exists(ulay):
            st.warning(f'Underlay image not found: {ulay}')
            return
        if not os.path.exists(olay):
            st.warning(f'Overlay image not found {olay}')
            return
        #######################

        # Select ROI
        with layout:
            utilwd.select_muse_roi(list_vars)

        # Select plot parameters
        with layout:
            sac.divider(label='Plot Options', align='center', color='gray')

            plot_params = st.session_state.plot_params

            plot_params['list_orient'] = utilwd.my_multiselect(
                img_views, '_sel_orient', 'View Planes'
            )

            if len(plot_params['list_orient']) == 0:
                return

            plot_params['is_show_overlay'] = st.checkbox("Show overlay", True, disabled=False)

            plot_params['crop_to_mask'] = st.checkbox("Crop to mask", True, disabled=False)



        # Show figures
        utilmri.panel_view_seg(ulay, olay, plot_params)

    elif pipeline == 'dlwmls':
        st.info('Not available yet ...')

def view_img_vars(layout, pipeline):
    """
    View image variables
    """    
    if pipeline == 'dlmuse':
        view_dlmuse_volumes(layout)
        
    elif pipeline == 'dlwmls':
        st.info('Not available yet ...')

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

def panel_results(layout):
    logger.debug('    Function: panel_results')

    with layout:
        sac.divider(label='General Options', align='center', color='gray')

    with layout:
        sel_task = utilwd.my_selectbox(['Download', 'View'], 'res_sel_task', hdr='Task')

    if sel_task == 'Download':
        panel_download()

    elif sel_task == 'View':

        with layout:
            sel_rtype = utilwd.my_selectbox(['Quantitative', 'Image'], 'res_sel_rtype', 'Result Type')

        if sel_rtype == 'Image':
            with layout:
                sel_pipe = utilwd.my_selectbox(['dlmuse', 'dlwmls'], 'res_sel_pipe', 'Pipeline')
            view_segmentation(layout, sel_pipe)

        elif sel_rtype == 'Quantitative':
            with layout:
                sel_pipe = utilwd.my_selectbox(['dlmuse', 'dlwmls'], 'res_sel_pipe', 'Pipeline')
            view_img_vars(layout, sel_pipe)


