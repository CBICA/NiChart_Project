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
