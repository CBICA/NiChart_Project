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

def select_sex_var():
    '''
    Set sex var values
    '''
    sel_vals = st.pills(
        label='Select Sex',
        options=['F', 'M'],
        key = '_sex_var',
        default = st.session_state.plot_params['filter_sex'],
        selection_mode = 'multi',
    )
    st.session_state.plot_params['filter_sex'] = sel_vals

    st.slider(
        'Select Age Range:',
        min_value = st.session_state.plot_settings['min_age'],
        max_value = st.session_state.plot_settings['max_age'],
        key = '_age_range',
        on_change = update_age_range
    )

## Wrappers for standard widgets
def my_slider(var_name, hdr, min_val, max_val):
    # Initialize once
    if var_name not in st.session_state:
        st.session_state[var_name] = min_val
    return st.slider(
        hdr,
        min_value=min_val,
        max_value=max_val,
        key=var_name
    )

def my_checkbox(var_name, hdr):
    # Initialize once
    if var_name not in st.session_state:
        st.session_state[var_name] = min_val
    return st.slider(
        hdr,
        min_value=min_val,
        max_value=max_val,
        key=var_name
    )
    #'''
    #Wrapper for checkbox
    #'''
    #sel_val = st.session_state.get(var_name)
    #sel_opt = st.checkbox(
        #hdr, value=sel_val, key=f'_{var_name}'
    #)
    #st.session_state[var_name] = sel_opt
    #return sel_opt

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

def my_multiselect(list_opts, var_name, hdr='selection box', label_visibility='visible'):
    '''
    Wrapper for multiselect
    '''
    sel_opt = st.multiselect(
        hdr, list_opts, key=f'_{var_name}',
        default=st.session_state.get(var_name), label_visibility=label_visibility
    )
    st.session_state[var_name] = sel_opt
    return sel_opt

def safe_index(lst, value, default=0):
    try:
        return lst.index(value)
    except ValueError:
        return default

def my_selectbox(list_opts, var_name, hdr='selection box', label_visibility='visible'):
    '''
    Wrapper for selectbox
    '''
    options = ["Select an option…"] + list_opts
    sel_ind = safe_index(options, st.session_state.get(var_name))
    sel_opt = st.selectbox(
        hdr, options, key=f'_{var_name}', index=sel_ind,
        label_visibility=label_visibility, disabled=(len(list_opts) == 0)
    )
    st.session_state[var_name] = sel_opt
    return sel_opt

def selectbox_twolevels(
    df_vars, list_vars,
    var_name1, var_name2,
    dicts_rename = None
):
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
        tmp_list = [x for x in tmp_list if x in list_vars]

        roi_dict[tmp_group] = tmp_list
        
    with st.container(horizontal=True, horizontal_alignment="left"):
        sel_opt1 = my_selectbox(list(roi_dict), var_name1, label_visibility='collapsed')
        if sel_opt1 is None or sel_opt1 == 'Select an option…':
            list_opts = []
        else:
            list_opts = roi_dict[sel_opt1]
        sel_opt2 = my_selectbox(list_opts, var_name2, label_visibility='collapsed')

        st.session_state[var_name1] = sel_opt1
        st.session_state[var_name2] = sel_opt2

    return sel_opt2

## Specific Widgets
def select_muse_roi(list_vars):
    """
    Panel to set mriview parameters
    """
    df_vars = st.session_state.dicts['df_var_groups']
    
    # Select roi
    st.write('ROI Name')
    sel_var = selectbox_twolevels(
        df_vars[df_vars.category.isin(['roi'])],
        list_vars,
        'res_sel_roi_cat',
        'res_sel_roi_name',
        dicts_rename = {
            'muse': st.session_state.dicts['muse']['ind_to_name']
        }
    )
    st.session_state['sel_roi'] = sel_var
    return sel_var

def select_var_twolevels(hdr, list_vars, list_cat, var_name1, var_name2):
    """
    Panel to select a variable using a two level selection
    First level is the var category provided with data dict
    """
    df_vars = st.session_state.dicts['df_var_groups']
    sel_cats = df_vars[df_vars.category.isin(list_cat)]
    
    print('aaa')
    print(list_cat)
    print(df_vars.category)
    
    # Select roi
    st.write(hdr)
    sel_var = selectbox_twolevels(
        sel_cats, list_vars,
        var_name1, var_name2,
        dicts_rename = {
            'muse': st.session_state.dicts['muse']['ind_to_name']
        }
    )
    return sel_var

