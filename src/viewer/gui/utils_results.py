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

def set_plot_params(df):
    """
    Panel for selecting plotting parameters
    """
    list_vars = df.columns.unique().tolist()

    sac.divider(label='Plotting Parameters', align='center', color='indigo', size='lg')
    
    tab = sac.tabs(
        items=[
            sac.TabsItem(label='Variables'),
            sac.TabsItem(label='Filters'),
            sac.TabsItem(label='Trends'),
            sac.TabsItem(label='Centiles'),
            sac.TabsItem(label='Plot Settings'),
        ],
        size='sm',
        align='left'
    )
    
    
    #### Variables
    if tab == 'Variables':
        sel_xvar = utilwd.select_var_twolevels(
            'plot_params', 'xvargroup', 'xvar',
            'Variable X', list_vars, ['demog', 'roi'],
        )
        
        sel_yvar = utilwd.select_var_twolevels(
            'plot_params', 'yvargroup', 'yvar',
            'Variable Y', list_vars, ['roi'],
        )

        sel_hvar = utilwd.select_var_twolevels(
            'plot_params', 'hvargroup', 'hvar',
            'Grouping Variable', list_vars, ['demog'],
        )

    #### Filters
    if tab == 'Filters':

        # Let user select sex var
        sel_sex = utilwd.my_multiselect('plot_params', 'filter_sex', ['F','M'], 'Sex')

        # Let user pick an age range
        sel_age_range = utilwd.my_slider(
            'plot_params', 'filter_age', 'Age Range', 0, 110
        )

    #### Trends
    if tab == 'Trends':
        sac.divider(label='Trends', align='center', color='indigo', size='lg')
        utilwd.select_trend()
    
    #### Centiles
    if tab == 'Centiles':
        sac.divider(label='Centiles', align='center', color='indigo', size='lg')
        utilwd.select_centiles()

    #### Plot Settings
    if tab == 'Plot Settings':
        utilwd.select_plot_settings()

def set_centileplot_params():
    """
    Panel for selecting centile plot parameters
    """
    sac.divider(label='Plotting Parameters', align='center', color='indigo', size='lg')
    
    tab = sac.tabs(
        items=[
            sac.TabsItem(label='Centiles'),
            sac.TabsItem(label='Variables'),
            sac.TabsItem(label='Filters'),
            sac.TabsItem(label='Plot Settings'),
        ],
        size='sm',
        align='left'
    )

    #### Centiles
    if tab == 'Centiles':
        utilwd.select_centiles()
    
    #### Variables
    if tab == 'Variables':
        sel_xvar = utilwd.select_var_twolevels(
            'plot_params', 'xvargroup', 'xvar',
            'Variable X', ['Age'], ['demog'],
        )
        
        sel_yvar = utilwd.select_var_twolevels(
            'plot_params', 'yvargroup', 'yvar',
            'Variable Y', None, ['roi']
        )
    #### Filters
    if tab == 'Filters':
        # Let user pick an age range
        sel_age_range = utilwd.my_slider(
            'plot_params', 'filter_age', 'Age Range', 0, 110
        )
        
    #### Plot Settings
    if tab == 'Plot Settings':
        utilwd.select_plot_settings()        


def set_plot_controls():
    sac.divider(label='Plot Controls', align='center', color='indigo', size='lg')
    with st.container(horizontal=True, horizontal_alignment="center"):
        if st.button('Add Plot'):
            st.session_state.plots = utilpl.add_plot(
                st.session_state.plots, st.session_state.plot_params
            )
            #st.write(st.session_state.plots)
            #st.write(st.session_state.plot_params)
        if st.button('Delete Selected'):
            st.session_state.plots = utilpl.delete_sel_plots(
                st.session_state.plots
            )

        if st.button('Delete All'):
            st.session_state.plots = utilpl.delete_all_plots()


def plot_centiles(layout):
    """
    View centile values (reference data)
    """
    plot_params = st.session_state.plot_params
    
    with layout:
        set_centileplot_params()
        
    with layout:
        set_plot_controls()
        
    utilpl.panel_show_plots()

def plot_imgvars(layout):
    """
    View img variables (user data)
    """
    fname = os.path.join(
        st.session_state.paths['project'], 'plots', 'data_all.csv'
    )
    df = pd.read_csv(fname)

    ## FIXME rename rois
    df = df.rename(
        columns = st.session_state.dicts['muse']['ind_to_name']
    )
    
    if 'grouping_var' not in df:
        df["grouping_var"] = "Data"
    
    st.session_state.plot_data['df_data'] = df.copy()

    var_groups_data = ['roi']
    pipeline = 'dlmuse'

    with layout:
        set_plot_params(df)

    with layout:
        set_plot_controls()
            
    utilpl.panel_show_plots()

def view_segmentation(layout):
    """
    View segmentations
    """
    pipeline = st.session_state.general_params['sel_pipeline']

    with layout:
        sac.divider(label='Data', align='center', color='grape', size = 'xl')

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
            sel_mrid = utilwd.my_selectbox(
                'mriplot_params', 'sel_mrid', list_mrids, 'Subject'
            )

        if sel_mrid is None or sel_mrid == 'Select an option…':
            return

        #######################
        ## FIXME
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
        ## FIXME
        with layout:
            sel_roi = utilwd.select_muse_roi(list_vars)
        if sel_roi is None or sel_roi == 'Select an option…':
            return
        st.session_state.mriplot_params['sel_roi'] = sel_roi

        # Select plot parameters
        with layout:
            utilwd.select_mriplot_settings()
            
        utilmri.panel_view_seg(ulay, olay, st.session_state.mriplot_params)

    elif pipeline == 'dlwmls':
        st.info('Not available yet ...')

def view_img_vars(layout):
    """
    View image variables
    """    
    pipeline = st.session_state.general_params['sel_pipeline']
    if pipeline == 'dlmuse':
        if st.session_state.workflow == 'ref_data':
            plot_centiles(layout)
        else:
            plot_imgvars(layout)
      
    elif pipeline == 'dlwmls':
        st.info('Not available yet ...')

def prepare_data_for_download(prj_dir, sel_opt, out_zip):
    utilio.zip_folders(prj_dir, sel_opt, out_zip)
    with open(out_zip, "rb") as f:
        file_download = f.read()
    st.toast('Created zip file with selected folders')
    os.remove(out_zip)
    return file_download

def panel_download():
    '''
    Panel to download results
    '''
    if st.session_state.workflow == 'ref_data':
        st.info('Reference data download is not available at this time.')
        return
    
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

            st.download_button(
                label = f"Download",
                data = prepare_data_for_download(prj_dir, sel_opt, out_zip),
                file_name = 'nichart_results.zip',
                on_click = 'ignore'
            )


def panel_results():
    logger.debug('    Function: panel_results')

    if st.session_state.workflow is None:
        st.info('Please select a Workflow!')
        return

    # Set plotting parameters layout
    if st.session_state.layout_plots == 'Main':
        layout = st.container(border=False)
    else:
        layout = st.sidebar


    with layout:
        with st.container(horizontal=True, horizontal_alignment="left"):
            st.markdown("##### Settings ", width='content')
            with st.popover("❓", width='content'):
                st.write(
                    """
                    **Data Viewer Settings Help**
                    - Select options to view results from a specific pipeline
                    """
                )


    with layout:
        sac.divider(label='General Options', align='center', color='indigo', size='lg')

    with layout:
        sel_task = utilwd.my_selectbox(
            'general_params', 'sel_task', ['Download', 'View'], hdr='Task'
        )

    if sel_task == 'Download':
        panel_download()

    elif sel_task == 'View':
        with layout:
            sel_rtype = utilwd.my_selectbox(
                'general_params', 'sel_rtype',
                ['Numeric', 'Image'], 'Data Type'
            )

        if sel_rtype == 'Image':
            with layout:
                sel_pipe = utilwd.my_selectbox(
                    'general_params', 'sel_pipeline',
                    ['dlmuse', 'dlwmls'], 'Pipeline'
                )
            if sel_pipe is None or sel_pipe == 'Select an option...':
                return
            view_segmentation(layout)

        elif sel_rtype == 'Numeric':
            with layout:
                sel_pipe = utilwd.my_selectbox(
                    'general_params', 'sel_pipeline',
                    ['dlmuse', 'dlwmls'], 'Pipeline'
                )
            if sel_pipe is None or sel_pipe == 'Select an option...':
                return
            view_img_vars(layout)


