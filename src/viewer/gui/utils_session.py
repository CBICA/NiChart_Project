import os
from typing import Any
import pandas as pd
import streamlit as st
import gui.utils_cmaps as utilcmap
import gui.utils_dataprep as utildataprep

def init_plots():
    '''
    Set initial plot parameters in session state
    '''
    plots = {
        'index': 0,
        'df_plots': pd.DataFrame(columns=['flag_sel']),
        'settings': {},
        'params': {},
        'plots': {},
        'data': {},
        'auto': {},
        'selected': {}
    }

    # Set plot data params
    pref = os.path.join(st.session_state.paths['centiles'], 'nichart_centiles_dlmuse')
    ctype = 'CN'
    dtmp = {
        "cent_type": ctype,
        "cent_path": pref,
        "csv_cent": None,
        "df_cent": None,
        "csv_user":  os.path.join(st.session_state.paths['curr_data'], 'plots', 'plot_data.csv'),
        "df_user": None,
    }
    plots['data'].update(dtmp)

    dtmp = {
        'mrid': None,
        'xlab': None,
        'ylab': None,
        'mri_ulay': None,
        'mri_olay': None,
        'xcoor': None,
        'ycoor': None
    }
    plots['selected'].update(dtmp)

    dtmp = {
        'num_per_row': 4,
        'list_labels': ['DL_MUSE_Volume_GM', 'DL_MUSE_Volume_WM', 'DL_MUSE_Volume_Ventricles', 'DL_MUSE_Volume_Hippocampus_R'],
    }
    plots['auto'].update(dtmp)

    dtmp = {
        "plot_type": "scatter",
        "xlab": 'Age',
        "xvar": 'Age',
        "xmin": None,
        "xmax": None,
        "ylab": 'DL_MUSE_Volume_GM',
        "yvar": 'DL_MUSE_Volume_601',
        "ymin": None,
        "ymax": None,
        "hlab": None,
        "hvar": None,
        "hvals": None,
        "flab": None,
        "fvar": None,
        "fvals": None,
        "cent_values": ['centile_25', 'centile_50', 'centile_75'],
        "list_roi_indices": None,
        'filter_sex': ['F', 'M'],
        'filter_age': [40, 95],
        "trend": None,
        "flag_show_conf": False,
        "lowess_s": 0.7,
        'flag_sel': True,
        'flag_norm_icv': False,
        'flag_norm_cent': False,
    }
    plots['params'].update(dtmp)

    dtmp = {
        'index': 0,
        'mri_layout_horizontal': True,
        'mri_orient': ["axial", "coronal", "sagittal"],
        'mri_ulay': None,
        'mri_olay': None,
        'mri_flag_olay': True,
        'mri_flag_crop': True,
        'mri_flag_show': True,
        'sel_mrid': None,
        'sel_roi': None,
        'sel_task': None,
        'sel_rtype': None,
        'sel_pipeline': 'dlmuse',   #--- IGNORE --- --- IGNORE ---
        'map_minmax': [2.0, 5.0],
        'curr': -1,
        'active': None,
        "res_type": None,       # Quantitative or Image
        "trend_types": ["Linear", "Smooth LOWESS Curve"],
        "linfit_trace_types": ["lin_fit", "conf_95%"],
        "cent_trace_types": [
            "centile_5", "centile_25", "centile_50", "centile_75", "centile_95",
        ],
        "distplot_trace_types": ["histogram", "density", "rug"],
        'flag_resize': False,
        "flag_show_legend": False,
        "min_per_row": 1,
        "max_per_row": 5,
        "num_per_row": 2,
        "margin": 20,
        "h_init": 500,
        "h_coeff": 1.0,
        "h_coeff_max": 2.0,
        "h_coeff_min": 0.6,
        "h_coeff_step": 0.2,
        "distplot_binnum": 100,
        "cmaps": utilcmap.cmaps_init,
        "alphas": utilcmap.alphas_init,
        "w_cent": 6,
        "w_fit": 6,
        "min_age": 20,
        "max_age": 100,
        "flag_xline": True,
        "flag_yline": True,
    }
    plots['settings'].update(dtmp)

    return plots

def init_ref_plots():
    '''
    Set initial reference plot parameters in session state
    '''
    # Set initial plot parameters for reference plots
    ss=st.session_state
    plots = init_plots()

    # Set centile types
    plots['settings']['cent_types'] = ["None", "CN", "CN-Males", "CN-Females"]

    # Save in session state
    st.session_state['ref_plots'] = plots.copy()

    # Load centile data
    utildataprep.load_centile_data('ref_plots')

def init_user_plots():
    '''
    Set initial user plot parameters in session state
    '''
    # Set initial plot parameters for user plots
    ss=st.session_state
    plots = init_plots()

    # Set centile types
    plots['settings']['cent_types'] = ["CN", "CN-Males", "CN-Females"]

    # Save in session state
    st.session_state['user_plots'] = plots.copy()

    # Load centile data
    utildataprep.load_centile_data('user_plots')

def update_session_state():
    '''
    Update session state with plotting variables
    '''

    # Initiate Session State Values
    if "plotvars_instantiated" not in st.session_state:
        init_ref_plots()
        init_user_plots()
        
        # Set viewer to reference data by default
        st.session_state.flag_user_data = False

        st.session_state.plotvars_instantiated = True
