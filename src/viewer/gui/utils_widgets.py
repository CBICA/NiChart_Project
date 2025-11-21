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

def safe_index(lst, value, default=None):
    try:
        return lst.index(value)
    except ValueError:
        return default

def my_selectbox(list_opts, var_name, hdr):
    '''
    Generic selectbox
    Standard selectbox does not initialize value to the key variable by default
    This is a wrapper to resolve this issue
    '''
    sel_ind = safe_index(list_opts, st.session_state.get(var_name))
    sel_opt = st.selectbox(hdr, list_opts, key=var_name, index=sel_ind)
    return st.session_state[var_name]

def selectbox_twolevel(
    df_vars, list_vars,
    vname1, vname2,
    flag_add_none = False, dicts_rename = None):
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
        sel1 = my_selectbox(list(roi_dict), vname1, 'Group:')
        if sel1 is None:
            return
        sel2 = my_selectbox(roi_dict[sel1], vname2, 'Var:')

    return sel2
