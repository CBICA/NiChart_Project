import os
import shutil
import time
from typing import Any

import pandas as pd
import streamlit as st
import utils.utils_doc as utildoc
import utils.utils_io as utilio
import utils.utils_session as utilss
import utils.utils_rois as utilroi
import utils.utils_nifti as utilni
from stqdm import stqdm
import utils.utils_st as utilst


def select_muse_roi():

    # Select roi
    list_roi = ['GM', 'WM', 'VN']
    sel_roi = st.selectbox(
        "Select ROI",
        list_roi,
        None,
        help="Select an ROI from the list"
    )
    if sel_roi is not None:
        st.session_state.plot_params['yvar'] = sel_roi

    ## Select ROI name 
    #list_roi_names = df_rois.Name.sort_values().tolist()
    #sel_roi = st.selectbox(
        #"ROI", list_roi_names, key="selbox_rois", index=None, disabled=False
    #)        
    
    # Get indice for the selected roi
    df_sel  = df_rois[df_rois.Name == sel_roi]
    if 'List' in df_sel:
        list_roi_indices = df_sel.List.values[0]
    else:
        list_roi_indices = [df_sel.Index.values[0]]
