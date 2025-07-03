import os
from typing import Any, Optional
import pandas as pd
import numpy as np
import streamlit as st

###################################################################
# Misc utils

def styled_text(text):
    return f'<span style="color:teal; font-weight:600; background-color: #f5f5fa; padding: 4px 4px; border-radius: 3px;">{text}</span>'


def add_items_to_list(my_list: list, items_to_add: list) -> list:
    """Adds multiple items to a list, avoiding duplicates.

    Args:
      my_list: The list to add items to.
      items_to_add: A list of items to add.

    Returns:
      The modified list.
    """
    for item in items_to_add:
        if item not in my_list:
            my_list.append(item)
    return my_list

def remove_items_from_list(my_list: list, items_to_remove: list) -> list:
    """Removes multiple items from a list.

    Args:
      my_list: The list to remove items from.
      items_to_remove: A list of items to remove.

    Returns:
      The modified list.
    """
    out_list = []
    for item in my_list:
        if item not in items_to_remove:
            out_list.append(item)
    return out_list

def get_index_in_list(in_list: list, in_item: str) -> Optional[int]:
    """
    Returns the index of the item in list, or None if item not found
    """
    if in_item not in in_list:
        return None
    else:
        return list(in_list).index(in_item)
    
def get_roi_indices(sel_roi, atlas):
    '''
    Detect indices for a selected ROI
    '''
    if sel_roi is None:
        return None
    
    # Detect indices
    if atlas == 'muse':
        df_derived = st.session_state.rois['muse']['df_derived']
        
        list_roi_indices = df_derived[df_derived.Name == sel_roi].List.values[0]
        
        print(list_roi_indices)
        return list_roi_indices

    elif atlas == 'wmls':
        list_roi_indices = [1]
        return list_roi_indices

    return None    
