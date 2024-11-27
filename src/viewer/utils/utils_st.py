import os
from typing import Any

import numpy as np
import streamlit as st
import utils.utils_io as utilio
import utils.utils_session as utilses

# from wfork_streamlit_profiler import Profiler
# import pyinstrument

COL_LEFT = 3
COL_RIGHT_EMPTY = 0.01
COL_RIGHT_BUTTON = 1


def user_input_textfield(
    label: str, init_val: str, help_msg: str, flag_disabled: bool
) -> Any:
    """
    Single text field to read a text input from the user
    """
    tmpcol = st.columns((COL_LEFT, COL_RIGHT_EMPTY))
    with tmpcol[0]:
        user_sel = st.text_input(
            label, value=init_val, help=help_msg, disabled=flag_disabled
        )
        return user_sel


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
        user_sel = st.selectbox(
            label,
            selections,
            index=init_val,
            key=key,
            help=helpmsg,
            disabled=flag_disabled,
        )
    return user_sel


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
        user_sel = st.multiselect(
            label, options, init_val, key=key, help=help_msg, disabled=flag_disabled
        )
        return user_sel


def user_input_filename(
    label_btn: Any,
    key_st: Any,
    label_txt: str,
    path_last: str,
    init_path_curr: str,
    help_msg: str,
) -> Any:
    """
    Text field next to a button to read an input file path
    """
    out_file = None
    path_curr = init_path_curr
    path_dir = path_last
    tmpcol = st.columns((COL_LEFT, COL_RIGHT_BUTTON), vertical_alignment="bottom")
    with tmpcol[1]:
        if st.button(label_btn, key=f"key_btn_{key_st}"):
            # path_curr, path_dir = utilio.browse_file(path_dir)
            out_file = utilio.browse_file(path_dir)
            if out_file is not None and os.path.exists(out_file):
                path_curr = os.path.abspath(out_file)

    with tmpcol[0]:
        out_sel = st.text_input(
            label_txt,
            key=f"key_txt_{key_st}",
            value=path_curr,
            help=help_msg,
        )
        if os.path.exists(out_sel):
            path_curr = out_sel
    return path_curr, path_dir


def user_input_foldername(
    label_btn: Any,
    key_st: Any,
    label_txt: str,
    path_last: str,
    path_curr: str,
    help_msg: str,
) -> Any:
    """
    Text field in left and button in right to read an input folder path
    """
    out_str = None
    tmpcol = st.columns((COL_LEFT, COL_RIGHT_BUTTON), vertical_alignment="bottom")

    # Button to select folder
    with tmpcol[1]:
        if st.button(label_btn, key=f"btn_{key_st}"):
            if os.path.exists(path_curr):
                out_str = utilio.browse_folder(path_curr)
            else:
                out_str = utilio.browse_folder(path_last)

    if out_str is not None and os.path.exists(out_str):
        out_str = os.path.abspath(out_str)
        path_curr = os.path.abspath(out_str)

    # Text field to select folder
    with tmpcol[0]:
        if os.path.exists(path_curr):
            out_str = st.text_input(
                label_txt, key=f"txt2_{key_st}", value=path_curr, help=help_msg
            )
        else:
            out_str = st.text_input(
                label_txt,
                key=f"txt2_{key_st}",
                value="",
                help=help_msg,
            )

    if os.path.exists(out_str):
        out_str = os.path.abspath(out_str)
        path_curr = out_str

    return out_str


def show_img3D(
    img: np.ndarray, scroll_axis: Any, sel_axis_bounds: Any, img_name: str
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
    w_img = (
        st.session_state.mriview_const["w_init"]
        * st.session_state.mriview_var["w_coeff"]
    )
    if scroll_axis == 0:
        # st.image(img[slice_index, :, :], use_column_width=True)
        st.image(img[slice_index, :, :], width=w_img)
    elif scroll_axis == 1:
        st.image(img[:, slice_index, :], width=w_img)
    else:
        st.image(img[:, :, slice_index], width=w_img)


def util_panel_workingdir(app_type: str) -> None:
    """
    Panel to set results folder name
    """
    curr_dir = st.session_state.paths["dset"]

    # Read dataset name (used to create a folder where all results will be saved)
    helpmsg = (
        "Each study's results are organized in a dedicated folder named after the study"
    )
    st.session_state.dset = user_input_textfield(
        "Study name", st.session_state.dset, helpmsg, False
    )

    if app_type == "desktop":
        # Read output folder from the user
        helpmsg = "Results will be saved to the output folder.\n\nChoose the path by typing it into the text field or using the file browser to browse and select it"
        dir_out = user_input_foldername(
            "Select folder",
            "btn_sel_dir_out",
            "Output folder",
            st.session_state.paths["file_search_dir"],
            st.session_state.paths["dir_out"],
            helpmsg,
        )

        if dir_out != "":
            st.session_state.paths["dir_out"] = dir_out

    # Create results folder
    if st.session_state.dset != "" and st.session_state.paths["dir_out"] != "":
        st.session_state.paths["dset"] = os.path.join(
            st.session_state.paths["dir_out"], st.session_state.dset
        )

    # Check if results folder name changed
    if curr_dir != st.session_state.paths["dset"]:
        # Create output folder
        if not os.path.exists(st.session_state.paths["dset"]):
            os.makedirs(st.session_state.paths["dset"])

        # Update paths for output subfolders
        utilses.update_default_paths()
        utilses.reset_flags()


def copy_uploaded_to_dir() -> None:
    # Copies files to local storage
    if len(st.session_state["uploaded_input"]) > 0:
        utilio.copy_and_unzip_uploaded_files(
            st.session_state["uploaded_input"], st.session_state.paths["target_path"]
        )


def util_upload_folder(
    dir_out: str, title_txt: str, flag_disabled: bool, help_txt: str
) -> None:
    """
    Upload user data to target folder
    Input data may be a folder, multiple files, or a zip file (unzip the zip file if so)
    """
    # Set target path
    if not flag_disabled:
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        st.session_state.paths["target_path"] = dir_out

    # Upload data
    st.file_uploader(
        title_txt,
        key="uploaded_input",
        accept_multiple_files=True,
        on_change=copy_uploaded_to_dir,
        disabled=flag_disabled,
        help=help_txt,
    )


def util_upload_file(
    out_file: str,
    title_txt: str,
    key_uploader: str,
    flag_disabled: bool,
    label_visibility: str,
) -> bool:
    """
    Upload user data to target folder
    Input data is a single file
    """
    # Set target path
    if not flag_disabled:
        if not os.path.exists(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file))

    # Upload input
    in_file = st.file_uploader(
        title_txt,
        key=key_uploader,
        accept_multiple_files=False,
        disabled=flag_disabled,
        label_visibility=label_visibility,
    )

    # Copy to target
    if not os.path.exists(out_file):
        utilio.copy_uploaded_file(in_file, out_file)
        return True

    return False


def util_select_folder(
    key_selector: str,
    title_txt: str,
    dir_out: str,
    file_search_dir: str,
    flag_disabled: bool,
) -> None:
    """
    Select user input folder and link to target folder
    """
    # Check if out folder already exists
    curr_dir = ""
    if os.path.exists(dir_out):
        fcount = utilio.get_file_count(dir_out)
        if fcount > 0:
            curr_dir = dir_out

    # Upload data
    helpmsg = "Select input folder.\n\nChoose the path by typing it into the text field or using the file browser to browse and select it"
    sel_dir = user_input_foldername(
        "Select folder",
        f"btn_{key_selector}",
        title_txt,
        file_search_dir,
        curr_dir,
        helpmsg,
    )

    if sel_dir is not None:
        if not os.path.exists(dir_out) and os.path.exists(sel_dir):
            # Create parent dir of output dir
            if not os.path.exists(os.path.dirname(dir_out)):
                os.makedirs(os.path.dirname(dir_out))
            # Link user input dicoms
            os.symlink(sel_dir, dir_out)


def util_select_file(
    key_selector: str,
    title_txt: str,
    out_file: str,
    file_search_dir: str,
) -> bool:
    """
    Select user input file and copy to target file
    """
    # Check if out file already exists
    curr_file = ""
    if os.path.exists(out_file):
        curr_file = out_file

    # Select file
    helpmsg = "Select input file.\n\nChoose the path by typing it into the text field or using the file browser to browse and select it"
    sel_file, file_search_dir = user_input_filename(
        "Select file",
        f"btn_{key_selector}",
        title_txt,
        file_search_dir,
        curr_file,
        helpmsg,
    )

    if not os.path.exists(out_file) and os.path.exists(sel_file):
        # Create parent dir of out file
        if not os.path.exists(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file))
        # Link user input dicoms
        os.system(f"cp {sel_file} {out_file}")
        return True

    return False


def add_debug_panel() -> None:
    """
    Displays vars used in dev phase
    """
    st.sidebar.divider()
    st.sidebar.write("*** Debugging Flags ***")
    if st.sidebar.checkbox("Switch to cloud?"):
        st.session_state.app_type = "cloud"
    else:
        st.session_state.app_type = "desktop"

    list_vars = ["", "All", "plots", "plot_var", "rois", "paths"]
    # list_vars = st.session_state.keys()
    sel_var = st.sidebar.selectbox("View session state vars", list_vars, index=0)
    if sel_var != "":
        with st.expander("DEBUG: Session state", expanded=True):
            if sel_var == "All":
                st.write(st.session_state)
            else:
                st.write(st.session_state[sel_var])
