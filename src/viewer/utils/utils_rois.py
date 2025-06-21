from typing import Any

import pandas as pd
import streamlit as st
import os

# from stqdm import stqdm


@st.cache_data  # type:ignore
def get_list_rois(sel_var: str, roi_dict: dict, derived_dict: dict) -> Any:
    """
    Get a list of ROI indices for the selected var
    """
    if sel_var is None:
        return None

    # Convert ROI name to index
    if sel_var in roi_dict.keys():
        sel_var = roi_dict[sel_var]

    sel_var = int(sel_var)

    # Get list of derived ROIs
    if sel_var in derived_dict.keys():
        list_rois = derived_dict[sel_var]
    else:
        if str.isnumeric(sel_var):
            list_rois = [sel_var]
        else:
            list_rois = []

    return list_rois


@st.cache_data  # type:ignore
def get_roi_names(csv_rois: str) -> Any:
    """
    Get a list of ROI names
    """
    # Read list
    df = pd.read_csv(csv_rois)
    return df.Name.tolist()


def muse_derived_to_dict(in_list: list) -> Any:
    """
    Create a dictionary from derived roi list
    """
    # Read list
    df = pd.read_csv(in_list, header=None)


    dict_derived = {
        row[0]: [int(x) for x in row[2:] if pd.notna(x)] for _, row in df.iterrows()
    }

    ## Create dict of roi indices and derived indices
    #dict_derived = {}
    #for i, tmp_ind in enumerate(df[0].values):
        #df_tmp = df[df[] == tmp_ind].drop([0, 1], axis=1)
        #sel_vals = df_tmp.T.dropna().astype(int).values.flatten()
        #dict_derived[str(tmp_ind)] = list(sel_vals)

    return dict_derived

def muse_derived_to_df(in_list: list) -> Any:
    """
    Create a df from derived roi list
    """
    # Read list
    df = pd.read_csv(in_list, header=None)

    # Rename first two columns
    df = df.rename(columns={0: 'Index', 1: 'Name'})

    # Create a new column 'List' with the remaining columns as a list
    df['List'] = df.iloc[:, 2:].apply(lambda row: row.dropna().astype(str).tolist(), axis=1)

    # Keep only the desired columns
    df = df[['Index', 'Name', 'List']]
    df.Index = df.Index.astype(str)

    return df

def muse_roi_groups_to_df(in_list: list) -> Any:
    """
    Create a df from roi groups list
    """
    # Read list
    df = pd.read_csv(in_list, header=None)

    # Rename first two columns
    df = df.rename(columns={0: 'Name'})

    # Create a new column 'List' with the remaining columns as a list
    df['List'] = df.iloc[:, 1:].apply(lambda row: row.dropna().astype(int).tolist(), axis=1)

    # Keep only the desired columns
    df = df[['Name', 'List']]

    return df


def muse_get_derived(sel_roi: str, in_list: list) -> Any:
    """
    Create a list of derived roi indices for the selected roi
    """

    # Read list
    df = pd.read_csv(in_list, header=None)

    # Keep only selected ROI
    df = df[df[0].astype(str) == sel_roi]

    if df.shape[0] == 0:
        return []

    # Get list of derived rois
    sel_vals = df.drop([0, 1], axis=1).T.dropna().astype(int).values.flatten()

    return sel_vals

def read_muse_dicts():
    '''
    Function to read muse dictionaries and save in session state
    '''
    f_muse = os.path.join(
        st.session_state.paths['resources'], 'dicts', 'muse', 'muse_dict.csv'
    )
    f_muse_derived = os.path.join(
        st.session_state.paths['resources'], 'dicts', 'muse', 'muse_mapping_derived.csv'
    )

    # Read muse roi list to dictionaries (ind->name, name->ind)
    df_muse = pd.read_csv(f_muse)

    # Remove duplicate entries
    d1 = dict(zip(df_muse["Index"].astype(str), df_muse["Name"].astype(str)))
    d2 = dict(zip(df_muse["Name"].astype(str), df_muse["Index"].astype(str)))

    # Read derived roi lists to dict
    d3 = muse_derived_to_dict(f_muse_derived)

    out_dicts = {
        'ind_to_name' : d1,
        'name_to_ind' : d2,
        'derived' : d3
    }

    return out_dicts


    muse['dict_roi'] = dict1
    muse['dict_roi_inv'] = dict2
    muse['dict_derived'] = dict3
    muse['df_derived'] = df_derived
    muse['df_groups'] = df_groups

