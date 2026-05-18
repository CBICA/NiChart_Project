from typing import Any

import pandas as pd
import streamlit as st
import os
import gui.utils_varnames as utilvars

def muse_derived_get_indices(in_roi: str, roi_dict: dict, derived_dict: dict) -> Any:
    """
    Get a list of ROI indices for the selected muse roi name or index
    Single item for single roi, or multiple for a derived roi
    """
    if in_roi is None:
        return []

    # Convert ROI name to index
    if in_roi in roi_dict.keys():
        in_roi = roi_dict[in_roi]

    # Convert to int
    in_roi = int(in_roi)

    # Get list of derived ROIs
    if in_roi in derived_dict.keys():
        list_rois = derived_dict[in_roi]
    else:
        list_rois = [in_roi]
    return list_rois

def muse_derived_to_dict(in_list: list) -> Any:
    """
    Create a dictionary from derived roi list
    """
    # Read list
    df = pd.read_csv(in_list, header=None)

    dict_derived = {
        row[0]: [int(x) for x in row[2:] if pd.notna(x)] for _, row in df.iterrows()
    }
    return dict_derived

def muse_read_dicts():
    '''
    Function to read muse dictionaries and save in session state
    '''
    f_muse = os.path.join(
        st.session_state.paths['resources'], 'atlases', 'muse', 'muse_dict.csv'
    )
    f_muse_derived = os.path.join(
        st.session_state.paths['resources'], 'atlases', 'muse', 'muse_mapping_derived.csv'
    )

    # Read muse roi list to dictionaries (ind->name, name->ind)
    df_muse = pd.read_csv(f_muse)
    map_muse = utilvars.VarMapper(df_muse) 

    # Read derived roi lists to dict
    map_muse_derived = muse_derived_to_dict(f_muse_derived)

    muse = {
        'map' : map_muse,
        'derived' : map_muse_derived
    }
    st.session_state.dicts['muse'] = muse

def muse_get_roi_indices(sel_roi):
    '''
    Detect indices for a selected ROI
    '''
    if sel_roi is None:
        return None
    
    df_derived = st.session_state.dicts['muse']['derived']
    list_roi_indices = df_derived[sel_roi]

    return list_roi_indices
