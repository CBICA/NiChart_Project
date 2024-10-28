import os
import tkinter as tk
from tkinter import filedialog
from typing import Any

import utils.utils_st as utilst
import numpy as np
import streamlit as st
import utils.utils_io as utilio
import glob

def detect_image_path(img_dir, mrid, img_suff):
    '''
    Detect image path using image folder, mrid and suffix
    Image search covers two different folder structures:
    - {img_dir}/{mrid}*{img_suff}
    - {img_dir}/{mrid}/{mrid}*{img_suff}
    '''
    print('Searching for image' + f'{img_dir}  {mrid}  {img_suff}')

    pattern = os.path.join(img_dir, f"{mrid}*{img_suff}")
    tmp_sel = glob.glob(pattern, recursive=False)
    if len(tmp_sel) > 0:
        return tmp_sel[0]

    pattern = os.path.join(img_dir, mrid, f"{mrid}*{img_suff}")
    tmp_sel = glob.glob(pattern, recursive=False)
    if len(tmp_sel) > 0:
        return tmp_sel[0]

    return None

def check_images():
    '''
    Checks if underlay and overlay images exists
    '''

    # Check underlay image
    sel_img = detect_image_path(
        st.session_state.paths["T1"],
        st.session_state.sel_mrid,
        st.session_state.suff_t1img
    )
    
    print(f"aaa {sel_img}")
    
    if sel_img is None:
        return False
    else:
        st.session_state.paths["sel_img"] = sel_img

    # Check overlay image
    sel_img = detect_image_path(
        st.session_state.paths["DLMUSE"],
        st.session_state.sel_mrid,
        st.session_state.suff_seg
    )
    if sel_img is None:
        return False
    else:
        st.session_state.paths["sel_seg"] = sel_img
        return True

def get_image_paths():
    '''
    Reads image path and suffix info from the user
    '''
    if st.session_state.app_type == "CLOUD":
        st.warning('Sorry, there are no images to show! Uploading images for viewing purposes is not implemented in the cloud version!')
    else:
        st.warning("I'm having trouble locating the underlay image. Please select path and suffix!")

        # Select image dir
        utilst.util_select_folder(
            'selected_t1_folder',
            'Underlay image folder',
            st.session_state.paths['T1'],
            st.session_state.paths['last_in_dir'],
            False,
        )

        # Select suffix
        suff_t1img = utilst.user_input_text(
            "Underlay image suffix",
            st.session_state.suff_t1img,
            "Enter the suffix for the T1 images",
            False
        )
        st.session_state.suff_t1img = suff_t1img

        # Select image dir
        utilst.util_select_folder(
            'selected_dlmuse_folder',
            'Overlay image folder',
            st.session_state.paths['DLMUSE'],
            st.session_state.paths['last_in_dir'],
            False,
        )

        # Select suffix
        suff_seg = utilst.user_input_text(
            "Overlay image suffix",
            st.session_state.suff_seg,
            "Enter the suffix for the DLMUSE images",
            False
        )
        st.session_state.suff_seg = suff_seg

        if st.button("Check image paths!"):
            if check_images():
                st.rerun()
            else:
                st.warning('Image not found!')

