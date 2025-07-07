import os
import shutil
from typing import Any, Optional

import pandas as pd
import numpy as np
import streamlit as st
import utils.utils_io as utilio
import utils.utils_session as utilses
import utils.utils_misc as utilmisc
import utils.utils_mriview as utilmri

import plotly.graph_objs as go
import plotly.figure_factory as ff
import utils.utils_traces as utiltr

import streamlit_antd_components as sac

def select_var_from_group(
    label,
    df_vars,
    init_group,
    init_var,
    flag_add_none = False,
    dicts_rename = None
):

    # print(flag_add_none )
    # return [init_group, init_var]

    '''
    Panel for user to select a variable grouped in categories
    Variable groups are given in df_vars
    If a variable is an ROI index a dictionary for renaming should be given in dicts_rename
    '''
    # Create nested var lists
    sac_items = []
    init_ind = None

    ind_count = 0
    for i, row in df_vars.iterrows():
        tmp_group = row['group']
        tmp_list = row['values']
        tmp_atlas = row['atlas']

        # Convert ROI variables from index to name
        if tmp_atlas is not None:
            tmp_list = [dicts_rename[tmp_atlas][k] for k in tmp_list]

        # Add a None item in var list
        if flag_add_none:
            tmp_list = ['None'] + tmp_list

        tmp_item = sac.CasItem(tmp_group, icon='app', children=tmp_list)
        sac_items.append(tmp_item)

        # Detect index of selected items
        # !!! CasItem keeps a linear (flattened) index of nested items
        # !!! For each group, the index is moved to:
        #     curr_index + #items in group + 1
        if init_group == tmp_group:
            if init_var in tmp_list:
                init_ind = [ind_count, ind_count + 1 + tmp_list.index(init_var)]
        ind_count = ind_count + 1 + len(tmp_list)

    # Show var selector
    sel_var = sac.cascader(
        items = sac_items,
        label=label,
        index=init_ind,
        return_index=False,
        multiple=False,
        search=True,
        clear=True,
        key=f'_sel_{label}'
    )
    #st.success(f'Selected: {sel_var}')

    return sel_var


def select_var_from_group2(
    df_groups,
    sel_groups,
    var_type,
    init_sel,
    add_none = False,
    dict_muse = None
):
    '''
    Select a variable grouped in categories
    '''
    # Select groups
    df_sel = df_groups[df_groups.category.isin(sel_groups)].reset_index()

    # Create nested var lists
    sac_items = []
    init_ind = None

    ind_count = 0
    for i, row in df_sel.iterrows():
        tmp_group = row['group']
        tmp_list = row['values']       
        tmp_atlas = row['atlas']
        
        # Convert MUSE ROI variables from index to name
        if tmp_atlas == 'muse':
            tmp_list = [dict_muse[k] for k in tmp_list]

        if add_none:
            tmp_list = ['None'] + tmp_list
        
        tmp_item = sac.CasItem(tmp_group, icon='app', children=tmp_list)
        sac_items.append(tmp_item)

        # Detect index of selected items
        # !!! CasItem keeps a linear (flattened) index of nested items
        # !!! For each group, the index is moved to:
        #     curr_index + #items in group + 1
        if init_sel[0] == tmp_group:
            if init_sel[1] in tmp_list:
                init_ind = [ind_count, ind_count + 1 + tmp_list.index(init_sel[1])]
        ind_count = ind_count + 1 + len(tmp_list)

    # Show var selector
    sel_var = sac.cascader(
        items = sac_items,
        label=f'Variable: {var_type}',
        index=init_ind,
        return_index = False,
        multiple=False,
        search=True,
        clear=True,
        key=f'_sel_{var_type}'
    )
    #st.success(f'Selected: {sel_var}')
    return sel_var
