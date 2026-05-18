from typing import Any, Optional

import streamlit as st

from utils.utils_logger import setup_logger
logger = setup_logger()

def get_nested_value(obj, path):
    '''
    Retrieves a value from a nested dictionary given a path like 'a.b.c'
    '''
    keys = path.split('.')
    for key in keys:
        obj = obj[key]
    return obj

def set_nested_value(obj, path, value):
    '''
    Sets a value in a nested dictionary given a path like 'a.b.c'
    '''
    keys = path.split('.')
    for key in keys[:-1]:
        # Create the dictionary level if it doesn't exist
        obj = obj.setdefault(key, {})
    obj[keys[-1]] = value

## Wrappers for standard widgets
def my_slider(var_name, label, min_val=0, max_val=10, step=1, width = 300):
    '''
    Wrapper for slider
    '''
    my_key = '_' + var_name.replace('.', '_')
    if my_key not in st.session_state:
        st.session_state[my_key] = get_nested_value(st.session_state, var_name)

    def update_slider():
        set_nested_value(st.session_state, var_name, st.session_state[my_key])

    st.slider(
        label,
        min_value=min_val,
        max_value=max_val,
        step=step,
        key=my_key,
        # value = get_nested_value(st.session_state, var_name),
        width = width,
        on_change=update_slider
    )

def my_checkbox(var_name, label):
    '''
    Wrapper for checkbox
    '''
    my_key = '_' + var_name.replace('.', '_')
    if my_key not in st.session_state:
        st.session_state[my_key] = get_nested_value(st.session_state, var_name)

    def update_checkbox():
        set_nested_value(st.session_state, var_name, st.session_state[my_key])

    st.checkbox(
        label,
        key=my_key,
        on_change=update_checkbox
    )

def my_multiselect(
        var_name, list_opts, label='selection box', label_visibility='visible', width = 300
    ):
    '''
    Wrapper for multiselect
    '''
    my_key = '_' + var_name.replace('.', '_')
    if my_key not in st.session_state:
        st.session_state[my_key] = get_nested_value(st.session_state, var_name)

    def update_multiselect():
        set_nested_value(st.session_state, var_name, st.session_state[my_key])

    st.multiselect(
        label,
        list_opts,
        key=my_key,
        label_visibility=label_visibility,
        placeholder='Select options...',
        width = width,
        on_change=update_multiselect
    )

def my_pills(var_name, list_opts, label='pills', label_visibility='visible'):
    '''
    Wrapper for pills
    '''
    my_key = '_' + var_name.replace('.', '_')
    if my_key not in st.session_state:
        st.session_state[my_key] = get_nested_value(st.session_state, var_name)

    def update_pills():
        set_nested_value(st.session_state, var_name, st.session_state[my_key])

    st.pills(
        label,
        list_opts,
        key=my_key,
        selection_mode = 'multi',
        label_visibility=label_visibility,
        on_change=update_pills
    )

def safe_index(lst, value, default=0):
    try:
        return lst.index(value)
    except ValueError:
        return default

def my_selectbox(
        var_name, list_opts, label='selection box', label_visibility='visible', width = 300
    ):
    '''
    Wrapper for selectbox
    '''
    my_key = '_' + var_name.replace('.', '_')

    logger.debug(f"Initializing selectbox for {var_name} with key {my_key}")
    logger.debug(f"  {get_nested_value(st.session_state, var_name)}")

    if my_key not in st.session_state:
        st.session_state[my_key] = get_nested_value(st.session_state, var_name)

    def update_selectbox():
        set_nested_value(st.session_state, var_name, st.session_state[my_key])

    curr_val = get_nested_value(st.session_state, var_name)
    sel_ind = safe_index(list_opts, curr_val)  # Get index of current value
    logger.debug(f"Selected index for {var_name}: {sel_ind} {list_opts[sel_ind]}")
    st.selectbox(
        label,
        list_opts,
        # index=sel_ind,
        key=my_key,
        label_visibility=label_visibility,
        placeholder='Select an option...',
        width = width,
        on_change=update_selectbox
    )

def my_number_input(var_name, label, step=1.0, label_visibility='visible'):
    '''
    Wrapper for number_input
    '''
    my_key = '_' + var_name.replace('.', '_')
    if my_key not in st.session_state:
        st.session_state[my_key] = get_nested_value(st.session_state, var_name)

    def update_number_input():
        set_nested_value(st.session_state, var_name, st.session_state[my_key])

    st.number_input(
        label,
        # value=st.session_state[my_key],
        step=step,
        key=my_key,
        label_visibility=label_visibility,
        on_change=update_number_input
    )

