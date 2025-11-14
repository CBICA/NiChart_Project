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
    list_res_type = ['Regional Volumes', 'Segmentation']
    sel_res_type = sac.tabs(
        list_res_type,
        size='lg',
        align='left'
    )   

    ## FIXME (list of rois from data file to init listbox selections)
    df = pd.read_csv(
            os.path.join(
                st.session_state.paths['resources'],
                'reference_data', 'centiles', 'dlmuse_centiles_CN.csv' 
            )
    )
    list_vars = ['Age', 'Sex'] + df.VarName.unique().tolist()


    if sel_res_type == 'Regional Volumes':
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

    elif sel_res_type == 'Segmentation':
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

def panel_view_ref():
    #st.info(
    results_overview()
        
    # Show selections
    utilses.disp_selections()

def panel_view_user():
    #st.info(
    results_overview()
        
    # Show selections
    utilses.disp_selections()

def results_overview():
    # Show results
    with st.container(border=True):
        if st.session_state.sel_pipeline == 'dlmuse':
            view_dlmuse()

        elif st.session_state.sel_pipeline == 'dlwmls':
            st.warning('Viewer not implemented for dlwmls')
            #view_dlwmls()

