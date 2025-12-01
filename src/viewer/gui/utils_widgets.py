import os
import shutil
from typing import Any, Optional

import pandas as pd
import numpy as np
import streamlit as st
import utils.utils_session as utilses
import utils.utils_misc as utilmisc
import utils.utils_mriview as utilmri

import plotly.graph_objs as go
import plotly.figure_factory as ff
import utils.utils_traces as utiltr

import streamlit_antd_components as sac

################################
## Wrappers for standard widgets
def my_slider(var_group, var_name, hdr, min_val=0, max_val=10, step=1):
    def update_slider():
        st.session_state[var_group][var_name] = st.session_state[f'_{var_name}']
        
    return st.slider(
        hdr,
        min_value=min_val,
        max_value=max_val,
        step=step, 
        key=f'_{var_name}',
        value = st.session_state[var_group][var_name],
        on_change=update_slider        
    )

def my_checkbox(var_group, var_name, hdr):
    def update_checkbox():
        st.session_state[var_group][var_name] = st.session_state[f'_{var_name}']
    return st.checkbox(
        hdr,
        key=f'_{var_name}',
        value = st.session_state[var_group][var_name],
        on_change=update_checkbox
    )

def my_multiselecttmp(list_opts, var_name, hdr='selection box', label_visibility='visible'):
    '''
    Wrapper for multiselect
    '''
    sel_vals = st.session_state.get(var_name)
    sel_opt = st.multiselect(
        hdr, list_opts, key=f'_{var_name}',
        default=sel_vals, label_visibility=label_visibility
    )
    st.session_state[var_name] = sel_opt
    return sel_opt

def my_multiselect(var_group, var_name, list_opts, hdr='selection box', label_visibility='visible'):
    '''
    Wrapper for multiselect
    '''
    sel_opt = st.multiselect(
        hdr, list_opts, key=f'_{var_name}',
        default=st.session_state[var_group][var_name],
        label_visibility=label_visibility
    )
    st.session_state[var_group][var_name] = sel_opt
    return sel_opt

def safe_index(lst, value, default=0):
    try:
        return lst.index(value)
    except ValueError:
        return default

def my_selectbox(var_group, var_name, list_opts, hdr='selection box', label_visibility='visible'):
    '''
    Wrapper for selectbox
    '''
    options = ["Select an option…"] + list_opts
    sel_ind = safe_index(options, st.session_state[var_group][var_name])

    sel_opt = st.selectbox(
        hdr, options, key=f'_{var_name}', index=sel_ind,
        label_visibility=label_visibility, disabled=(len(list_opts) == 0)
    )
    st.session_state[var_group][var_name] = sel_opt
    return sel_opt

################################
## Custom widgets

def selectbox_twolevels(var_group, var_name1, var_name2, df_vars, list_vars = None, dicts_rename = None):
    '''
    Selectbox with two levels to select a variable grouped in categories
    Also renames variables (e.g. roi indices to names)
    Returns the selected variable
    '''
    roi_dict = {}
    for i, row in df_vars.iterrows():
        tmp_group = row['group']
        tmp_list = row['values']
        tmp_atlas = row['atlas']

        # Convert ROI variables from index to name
        if tmp_atlas is not None:
            tmp_list = [dicts_rename[tmp_atlas][k] for k in tmp_list]

        # Select vars that are included in the data
        if list_vars is not None:
            tmp_list = [x for x in tmp_list if x in list_vars]

        roi_dict[tmp_group] = tmp_list

    with st.container(horizontal=True, horizontal_alignment="left"):
        sel_opt1 = my_selectbox(
            var_group, var_name1, list(roi_dict), label_visibility='collapsed'
        )
        if sel_opt1 is None or sel_opt1 == 'Select an option…':
            list_opts = []
        else:
            list_opts = roi_dict[sel_opt1]
        sel_opt2 = my_selectbox(
            var_group, var_name2, list_opts, label_visibility='collapsed'
        )

        st.session_state[var_group][var_name1] = sel_opt1
        st.session_state[var_group][var_name2] = sel_opt2

    return sel_opt2

def select_muse_roi(list_vars):
    """
    Panel to set mriview parameters
    """
    df_vars = st.session_state.dicts['df_var_groups']
    
    # Select roi
    st.write('ROI Name')
    sel_var = selectbox_twolevels(
        'plot_params',
        'xvargroup',
        'xvar',
        df_vars[df_vars.category.isin(['roi'])],
        list_vars,
        dicts_rename = {
            'muse': st.session_state.dicts['muse']['ind_to_name']
        }
    )
    st.session_state['sel_roi'] = sel_var
    return sel_var

def select_var_twolevels(var_group, var_name1, var_name2, hdr, list_cat, list_vars = None):
    """
    Panel to select a variable using a two level selection
    First level is the var category provided with data dict
    """
    df_vars = st.session_state.dicts['df_var_groups']
    sel_cats = df_vars[df_vars.category.isin(list_cat)]

    
    # Select roi
    st.write(hdr)
    sel_var = selectbox_twolevels(
        var_group, var_name1, var_name2,
        sel_cats, list_vars,
        dicts_rename = {
            'muse': st.session_state.dicts['muse']['ind_to_name']
        }
    )
    return sel_var


################################
## Specific selections
def select_trend():
    '''
    Panel to select trend
    '''
    list_trends = st.session_state.plot_settings["trend_types"]
    sel_trend = my_selectbox(
        'plot_params', 'trend', list_trends, 'Trend'
    )

    if sel_trend is None:
        return

    if sel_trend == 'Select an option…':
        return

    if sel_trend == 'Linear':
        show_conf = my_checkbox('plot_params', 'show_conf', "Add confidence interval")

    elif sel_trend == 'Smooth LOWESS Curve':
        sel_lowess_s = my_slider('plot_params', 'lowess_s', 'Smoothness', 0.4, 1.0, 0.1)

def select_centiles():
    '''
    User panel to select centile values
    '''
    plot_params = st.session_state.plot_params
    
    list_types = ['CN', 'CN-Females', 'CN-Males', 'CN-ICVNorm']
    list_values = ['centile_5', 'centile_25', 'centile_50', 'centile_75', 'centile_95']

    sel_cent_type = my_selectbox('plot_params', 'centile_type', list_types, 'Centile Type')
    
    if sel_cent_type is None:
        return

    if sel_cent_type == 'Select an option…':
        return

    # Read centile dataframe
    pipeline = st.session_state.general_params['sel_pipeline']
    csv_cent = os.path.join(
        st.session_state.paths['centiles'],
        f'{pipeline}_centiles_{plot_params['centile_type']}.csv'
    )
    if csv_cent != st.session_state.plot_data['csv_cent']:
        try:
            df_cent = pd.read_csv(csv_cent)
        except:
            st.toast('Could not read centile data!')
            return

        st.session_state.plot_data['csv_cent'] = csv_cent
        st.session_state.plot_data['df_cent'] = df_cent
        st.toast('Loaded centile data!')

    sel_cent_vals = my_multiselect('plot_params', 'centile_values', list_values, 'Centile Values')

    if plot_params['centile_values'] is not None:
        plot_params['traces'] = plot_params['traces'] + plot_params['centile_values']
        
def select_plot_settings():
    '''
    Panel to select plot settings
    '''
    plot_settings = st.session_state.plot_settings
    
    flag_auto = my_checkbox('plot_settings', 'flag_auto', 'Auto resize')
    
    if flag_auto:
        hdr='Max. number of plots per row'
    else:
        hdr='Number of plots per row'
    
    sel_num = my_slider( 
        'plot_settings', 'num_per_row', hdr,
        st.session_state.plot_settings["min_per_row"],
        st.session_state.plot_settings["max_per_row"],
        1
    )

    sel_h = my_slider( 
        'plot_settings', 'h_coeff', "Plot Height",
        st.session_state.plot_settings["h_coeff_min"],
        st.session_state.plot_settings["h_coeff_max"],
        st.session_state.plot_settings["h_coeff_step"]
    )

    # Checkbox to show/hide plot legend
    sel_f = my_checkbox('plot_settings', 'flag_hide_legend', "Hide legend")

def select_mriplot_settings():
    '''
    Panel to select mriplot settings
    '''
    img_views = ["axial", "coronal", "sagittal"]

    sac.divider(label='Plot Options', align='center', color='indigo', size='lg')

    sel_orient = my_multiselect(
        'mriplot_params', 'sel_orient', img_views, 'View Planes'
    )

    if sel_orient is None or len(sel_orient) == 0:
        return

    flag_overlay = my_checkbox('mriplot_params', 'flag_overlay', "Show overlay")

    flag_crop = my_checkbox('mriplot_params', 'flag_crop', "Crop to mask")


