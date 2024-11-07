from typing import Any

import pandas as pd
import streamlit as st
from stqdm import stqdm


@st.cache_data  # type:ignore
def get_roi_names(csv_rois: str) -> Any:
    """
    Get a list of ROI names
    """
    # Read list
    df = pd.read_csv(csv_rois)
    return df.Name.tolist()


def derived_list_to_dict(list_sel_rois: list, list_derived: list) -> Any:
    """
    Create a dictionary from derived roi list
    """

    # Read list
    df_sel = pd.read_csv(list_sel_rois)
    df = pd.read_csv(list_derived, header=None)

    # Keep only selected ROIs
    df = df[df[0].isin(df_sel.Index)]

    # Create dict of roi names and indices
    dict_roi = dict(zip(df[1], df[0]))

    # Create dict of roi indices and derived indices
    dict_derived = {}
    for i, tmp_ind in stqdm(
        enumerate(df[0].values),
        desc="Creating derived roi indices ...",
        total=len(df[0].values),
    ):
        df_tmp = df[df[0] == tmp_ind].drop([0, 1], axis=1)
        sel_vals = df_tmp.T.dropna().astype(int).values.flatten()
        dict_derived[tmp_ind] = sel_vals

    return dict_roi, dict_derived


def get_derived_rois(sel_roi: str, list_derived: list) -> Any:
    """
    Create a list of derived roi indices for the selected roi
    """

    # Read list
    df = pd.read_csv(list_derived, header=None)
    
    print(f'aaa {sel_roi}')
    print(f'aaa {df}')

    # Keep only selected ROI
    df = df[df[0].astype(str) == sel_roi]

    if df.shape[0] == 0:
        return []

    # Get list of derived rois
    sel_vals = df.drop([0, 1], axis=1).T.dropna().astype(int).values.flatten()

    return sel_vals
