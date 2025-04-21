import os
import shutil
from typing import Any

import numpy as np
import streamlit as st
import utils.utils_io as utilio
import utils.utils_session as utilses

COL_LEFT = 5
COL_RIGHT_EMPTY = 0.01
COL_RIGHT_BUTTON = 1

def get_next_option(list_options, sel_opt):
    '''
    From a list of options get the next one to selected option
    '''
    sel_ind = list_options.index(sel_opt)
    new_ind = (sel_ind + 1) % len(list_options)
    next_opt = list_options[new_ind]
    return next_opt

def user_input_select(
    label: Any,
    key: Any,
    selections: Any,
    init_val: Any,
    helpmsg: str,
    flag_disabled: bool,
) -> Any:
    """
    Single selection box to read user selection
    """
    tmpcol = st.columns((COL_LEFT, COL_RIGHT_EMPTY))
    with tmpcol[0]:
        out_sel = st.selectbox(
            label,
            selections,
            index=init_val,
            key=key,
            help=helpmsg,
            disabled=flag_disabled,
        )
    return out_sel

def user_input_multiselect(
    label: str,
    key: Any,
    options: list,
    init_val: str,
    help_msg: str,
    flag_disabled: bool,
) -> Any:
    """
    Single text field to read a text input from the user
    """
    tmpcol = st.columns((COL_LEFT, COL_RIGHT_EMPTY))
    with tmpcol[0]:
        out_sel = st.multiselect(
            label, options, init_val, key=key, help=help_msg, disabled=flag_disabled
        )
        return out_sel
    
def show_img3D(
    img: np.ndarray,
    scroll_axis: Any,
    sel_axis_bounds: Any,
    img_name: str,
    size_auto: bool,
) -> None:
    """
    Display a 3D img
    """

    # Create a slider to select the slice index
    slice_index = st.slider(
        f"{img_name}",
        0,
        sel_axis_bounds[1] - 1,
        value=sel_axis_bounds[2],
        key=f"slider_{img_name}",
    )

    # Extract the slice and display it
    if size_auto:
        if scroll_axis == 0:
            st.image(img[slice_index, :, :], use_container_width=True)
        elif scroll_axis == 1:
            st.image(img[:, slice_index, :], use_container_width=True)
        else:
            st.image(img[:, :, slice_index], use_container_width=True)
    else:
        w_img = (
            st.session_state.mriview_const["w_init"]
            * st.session_state.mriview_var["w_coeff"]
        )
        if scroll_axis == 0:
            # st.image(img[slice_index, :, :], use_container_width=True)
            st.image(img[slice_index, :, :], width=w_img)
        elif scroll_axis == 1:
            st.image(img[:, slice_index, :], width=w_img)
        else:
            st.image(img[:, :, slice_index], width=w_img)
