from typing import Any

import pandas as pd
import streamlit as st

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


def muse_derived_to_dict(list_derived: list) -> Any:
    """
    Create a dictionary from derived roi list
    """
    # Read list
    df = pd.read_csv(list_derived, header=None)


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


def muse_get_derived(sel_roi: str, list_derived: list) -> Any:
    """
    Create a list of derived roi indices for the selected roi
    """

    # Read list
    df = pd.read_csv(list_derived, header=None)

    # Keep only selected ROI
    df = df[df[0].astype(str) == sel_roi]

    if df.shape[0] == 0:
        return []

    # Get list of derived rois
    sel_vals = df.drop([0, 1], axis=1).T.dropna().astype(int).values.flatten()

    return sel_vals
