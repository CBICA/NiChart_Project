import glob
import os
import time
from typing import Optional

import streamlit as st
import utils.utils_st as utilst


def detect_image_path(img_dir: str, mrid: str, img_suff: str) -> Optional[str]:
    """
    Detect image path using image folder, mrid and suffix
    Image search covers three different folder structures:
    - {img_dir}/{mrid}{img_suff}
    - {img_dir}/{mrid}*{img_suff}
    - {img_dir}/{mrid}/{mrid}*{img_suff}
    """
    print("Searching for image " + f"{img_dir}  {mrid}  {img_suff}")

    fimg = os.path.join(img_dir, f"{mrid}{img_suff}")
    if os.path.exists(fimg):
        return fimg

    pattern = os.path.join(img_dir, f"{mrid}*{img_suff}")
    tmp_sel = glob.glob(pattern, recursive=False)
    if len(tmp_sel) > 0:
        return tmp_sel[0]

    pattern = os.path.join(img_dir, mrid, f"{mrid}*{img_suff}")
    tmp_sel = glob.glob(pattern, recursive=False)
    if len(tmp_sel) > 0:
        return tmp_sel[0]

    return None


def check_image_underlay() -> bool:
    """
    Checks if underlay image exists
    """
    sel_img = detect_image_path(
        st.session_state.paths["T1"],
        st.session_state.sel_mrid,
        st.session_state.suff_t1img,
    )
    if sel_img is None:
        return False
    else:
        st.session_state.paths["sel_img"] = sel_img
        return True


def check_image_overlay() -> bool:
    sel_img = detect_image_path(
        st.session_state.paths["dlmuse"],
        st.session_state.sel_mrid,
        st.session_state.suff_seg,
    )
    if sel_img is None:
        return False
    else:
        st.session_state.paths["sel_seg"] = sel_img
        return True


@st.dialog("Get input data")  # type:ignore
def update_ulay_image_path() -> None:
    """
    Reads image path and suffix info from the user
    """
    if st.session_state.app_type == "cloud":
        st.warning(
            "Sorry, uploading images for viewing purposes is not implemented in the cloud version!"
        )
        return

    # Select image dir
    utilst.util_select_folder(
        "selected_t1_folder",
        "Underlay image folder",
        st.session_state.paths["T1"],
        st.session_state.paths["file_search_dir"],
        False,
    )

    # Select suffix
    suff_t1img = utilst.user_input_textfield(
        "Underlay image suffix",
        st.session_state.suff_t1img,
        "Enter the suffix for the T1 images",
        False,
    )
    st.session_state.suff_t1img = suff_t1img

    if st.button("Check underlay image"):
        if check_image_underlay():
            st.success(f'Image found! {st.session_state.paths["sel_img"]}')
            time.sleep(1)
            st.rerun()
        else:
            st.warning("Image not found!")


@st.dialog("Get input data")  # type:ignore
def update_olay_image_path() -> None:
    """
    Reads image path and suffix info from the user
    """
    if st.session_state.app_type == "cloud":
        st.warning(
            "Sorry, uploading images for viewing purposes is not implemented in the cloud version!"
        )
        return

    # Select image dir
    utilst.util_select_folder(
        "selected_dlmuse_folder",
        "Overlay image folder",
        st.session_state.paths["dlmuse"],
        st.session_state.paths["file_search_dir"],
        False,
    )

    # Select suffix
    suff_seg = utilst.user_input_textfield(
        "Overlay image suffix",
        st.session_state.suff_seg,
        "Enter the suffix for the DLMUSE images",
        False,
    )
    st.session_state.suff_seg = suff_seg

    if st.button("Check overlay image"):
        if check_image_overlay():
            st.success(f'Overlay image found! {st.session_state.paths["sel_seg"]}')
            time.sleep(1)
            st.rerun()
        else:
            st.warning("Image not found!")
