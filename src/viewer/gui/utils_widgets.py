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

## Generic functions
def my_multiselect(list_opts, var_name, hdr='selection box', label_visibility='visible'):
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
    flag_add_none = False,
    dicts_rename = None
):
    '''
    Selectbox with two levels to select a variable grouped in categories
    Also renames variables (e.g. roi indices to names)
    Returns the selected variable
    '''
    roi_dict = {}
    ind_count = 0
    for i, row in df_vars.iterrows():
        tmp_group = row['group']
        tmp_list = row['values']
        tmp_atlas = row['atlas']

        # Convert ROI variables from index to name
        if tmp_atlas is not None:
            tmp_list = [dicts_rename[tmp_atlas][k] for k in tmp_list]

        # Select vars that are included in the data
        tmp_list = [x for x in tmp_list if x in list_vars]

        # Add a None item in var list
        if flag_add_none:
            tmp_list = ['None'] + tmp_list
    
        roi_dict[tmp_group] = tmp_list

    with st.container(horizontal=True, horizontal_alignment="left"):
        sel1 = my_selectbox(list(roi_dict), var_name1, label_visibility='collapsed')
        if sel1 is None or sel1 == 'Select an option…':
            list_opts = []
        else:
            list_opts = roi_dict[sel1]
        sel2 = my_selectbox(list_opts, var_name2, label_visibility='collapsed')

    return sel2

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
        'res_sel_roi_group',
        'res_sel_roi_name',
        flag_add_none = False,
        dicts_rename = {
            'muse': st.session_state.dicts['muse']['ind_to_name']
        }
    )
    st.session_state['sel_roi'] = sel_var
