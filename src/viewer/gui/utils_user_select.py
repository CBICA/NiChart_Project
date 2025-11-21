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

def select_from_list(list_opts, var_name, hdr):
    '''
    Generic selection box 
    For a variable (var_name) initiated with the given list (list_opts)
    Variable is saved in session_state (used as the key for the select box)
    '''
    sel_ind = safe_index(list_opts, st.session_state.get(var_name))
    sel_opt = st.selectbox(hdr, list_opts, key=var_name, index=sel_ind)
    return st.session_state[var_name]

def select_var_from_group(df_vars, list_vars, flag_add_none = False, dicts_rename = None):
    '''
    Panel for user to select a variable grouped in categories
    '''
    roi_dict = {}
    ind_count = 0
    for i, row in df_vars.iterrows():
        tmp_group = row['group']
        tmp_list = row['values']
        tmp_atlas = row['atlas']

        st.write(f'A1: {tmp_group}')
        #st.write(f'A2: {tmp_list}')
        #st.write(f'A3: {tmp_atlas}')

        # Convert ROI variables from index to name
        if tmp_atlas is not None:
            tmp_list = [dicts_rename[tmp_atlas][k] for k in tmp_list]

        # Select vars that are included in the data
        tmp_list = [x for x in tmp_list if x in list_vars]

        # Add a None item in var list
        if flag_add_none:
            tmp_list = ['None'] + tmp_list
    
        roi_dict[tmp_group] = tmp_list

    
    #st.write(roi_dict.keys().tolist())
    
    with st.container(horizontal=True, horizontal_alignment="left"):
        sel_group = select_from_list(list(roi_dict), '_sel_group', 'Group:')
        if sel_group is None:
            return
        sel_var = select_from_list(roi_dict[sel_group], '_sel_var', 'Var:')

