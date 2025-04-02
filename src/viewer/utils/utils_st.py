import os
import shutil
from typing import Any

import numpy as np
import streamlit as st
import utils.utils_io as utilio
import utils.utils_session as utilses

# from wfork_streamlit_profiler import Profiler
# import pyinstrument

COL_LEFT = 5
COL_RIGHT_EMPTY = 0.01
COL_RIGHT_BUTTON = 1


def util_select_expname(out_dir: str, exp_curr: str) -> Any:
    """
    Set/select experiment name
    """
    # Read dataset name (used to create a folder where all results will be saved)
    sel_opt = st.radio(
        "Options:",
        ["Select Existing", "Create New"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if sel_opt == "Select Existing":
        list_exp = utilio.get_subfolders(out_dir)
        exp_sel = st.selectbox(
            "Select",
            list_exp,
            None,
            label_visibility="collapsed",
            placeholder="Select experiment name",
        )

    if sel_opt == "Create New":
        exp_sel = st.text_input(
            "Experiment name:",
            None,
            label_visibility="collapsed",
            placeholder="Type experiment name",
        )
        if exp_sel is not None:
            dir_tmp = os.path.join(out_dir, exp_sel)
            if not os.path.exists(dir_tmp):
                os.makedirs(dir_tmp)
    return exp_sel


def util_panel_experiment() -> None:
    """
    Panel for selecting experiment
    """
    with st.container(border=True):
        if st.session_state.flags["experiment"]:
            st.success(
                f"Experiment directory: {st.session_state.paths['experiment']}",
                icon=":material/thumb_up:",
            )
            if st.button("Switch"):
                st.session_state.flags["experiment"] = False
                st.session_state.experiment = ""
                st.rerun()

        else:
            out_dir = st.session_state.paths["out_dir"]
            exp_curr = st.session_state.experiment
            exp_sel = util_select_expname(out_dir, exp_curr)

            if exp_sel is not None and exp_sel != exp_curr:
                st.session_state.experiment = exp_sel
                st.session_state.paths["experiment"] = os.path.join(
                    st.session_state.paths["out_dir"], exp_sel
                )
                st.session_state.flags["experiment"] = True
                # Update paths when selected experiment changes
                utilses.update_default_paths()
                utilses.reset_flags()
                st.rerun()


def user_input_textfield(
    label: str, init_val: str, help_msg: str, flag_disabled: bool
) -> Any:
    """
    Single text field to read a text input from the user
    """
    tmpcol = st.columns((COL_LEFT, COL_RIGHT_EMPTY))
    with tmpcol[0]:
        out_text = st.text_input(
            label, value=init_val, help=help_msg, disabled=flag_disabled
        )
        return out_text


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


def user_input_filename(
    label_btn: Any,
    key_st: Any,
    label_txt: str,
    search_dir: str,
    init_path: str,
    help_msg: str,
) -> Any:
    """
    Text field next to a button to read an input file path
    """
    out_path = init_path
    tmpcol = st.columns((COL_LEFT, COL_RIGHT_BUTTON), vertical_alignment="bottom")

    # Button to select file
    with tmpcol[1]:
        if st.button(label_btn, key=f"key_btn_{key_st}", use_container_width=True):
            tmp_sel = utilio.browse_file(search_dir)
            if tmp_sel is not None and os.path.exists(tmp_sel):
                out_path = os.path.abspath(tmp_sel)

    # Text field to select file
    with tmpcol[0]:
        tmp_sel = st.text_input(
            label_txt,
            key=f"key_txt_{key_st}",
            value=out_path,
            help=help_msg,
        )
        if os.path.exists(tmp_sel):
            out_path = tmp_sel

    return out_path


def user_input_foldername(
    label_btn: Any,
    key_st: Any,
    label_txt: str,
    search_dir: str,
    init_path: str,
    help_msg: str,
) -> Any:
    """
    Text field in left and button in right to read an input folder path
    """
    out_path = init_path
    tmpcol = st.columns((COL_LEFT, COL_RIGHT_BUTTON), vertical_alignment="bottom")

    # Button to select folder
    with tmpcol[1]:
        if st.button(label_btn, key=f"btn_{key_st}", use_container_width=True):
            tmp_sel = utilio.browse_folder(search_dir)
            if tmp_sel is not None and os.path.exists(tmp_sel):
                out_path = os.path.abspath(tmp_sel)

    # Text field to select folder
    with tmpcol[0]:
        tmp_sel = st.text_input(
            label_txt, key=f"txt2_{key_st}", value=out_path, help=help_msg
        )
        if os.path.exists(tmp_sel):
            out_path = os.path.abspath(tmp_sel)

    return out_path


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
            st.image(img[slice_index, :, :], use_column_width=True)
        elif scroll_axis == 1:
            st.image(img[:, slice_index, :], use_column_width=True)
        else:
            st.image(img[:, :, slice_index], use_column_width=True)
    else:
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


def util_help_dialog(s_title: str, s_text: str) -> None:
    @st.dialog(s_title)  # type:ignore
    def help_working_dir():
        st.markdown(s_text)

    col1, col2 = st.columns([0.5, 0.1])
    with col2:
        if st.button(
            "Get help 🤔", key="key_btn_help_" + s_title, use_container_width=True
        ):
            help_working_dir()


def util_workingdir_get_help() -> None:
    @st.dialog("Working Directory")  # type:ignore
    def help_working_dir():
        st.markdown(
            """
            - A NiChart pipeline executes a series of steps, with input/output files organized in a predefined folder structure.

            - Results for an **experiment** (a new analysis on a new dataset) are kept in a dedicated **working directory**.

            - Set an **"output path"** (desktop app only) and an **"experiment name"** to define the **working directory** for your analysis. You only need to set the working directory once.

            - The **experiment name** can be any identifier that describes your analysis or data; it does not need to match the input study or data folder name.

            - You can initiate a NiChart pipeline by selecting the **working directory** from a previously completed experiment.
            """
        )
        st.warning(
            """
            On the cloud app, uploaded data and results of experiments are deleted in regular intervals!

            Accordingly, the data for a previous experiment may not be available.
            """
        )

    col1, col2 = st.columns([0.5, 0.1])
    with col2:
        if st.button(
            "Get help 🤔", key="key_btn_help_working_dir", use_container_width=True
        ):
            help_working_dir()


def copy_uploaded_to_dir() -> None:
    # Copies files to local storage
    if len(st.session_state["uploaded_input"]) > 0:
        utilio.copy_and_unzip_uploaded_files(
            st.session_state["uploaded_input"], st.session_state.paths["target_path"]
        )


def util_upload_folder(
    out_dir: str, title_txt: str, flag_disabled: bool, help_txt: str
) -> None:
    """
    Upload user data to target folder
    Input data may be a folder, multiple files, or a zip file (unzip the zip file if so)
    """
    # Set target path
    if not flag_disabled:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        st.session_state.paths["target_path"] = out_dir

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
    # Upload input
    in_file = st.file_uploader(
        title_txt,
        key=key_uploader,
        accept_multiple_files=False,
        disabled=flag_disabled,
        label_visibility=label_visibility,
    )
    if in_file is None:
        return False

    # Create parent dir of out file
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))
    # Remove existing output file
    if os.path.exists(out_file):
        os.remove(out_file)
    # Copy selected input file to destination
    utilio.copy_uploaded_file(in_file, out_file)
    return True


def util_select_folder(
    key_selector: str,
    title_txt: str,
    out_dir: str,
    file_search_dir: str,
    flag_disabled: bool,
) -> None:
    """
    Select user input folder and link to target folder
    """
    # Check if out folder already exists
    curr_dir = ""
    if os.path.exists(out_dir):
        fcount = utilio.get_file_count(out_dir)
        if fcount > 0:
            curr_dir = out_dir

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

    if sel_dir is not None and os.path.exists(sel_dir):
        # Remove existing output folder
        if os.path.exists(out_dir) and out_dir != sel_dir:
            if os.path.islink(out_dir):
                os.unlink(out_dir)
            else:
                shutil.rmtree(out_dir)

        # Create parent dir of output dir
        if not os.path.exists(os.path.dirname(out_dir)):
            os.makedirs(os.path.dirname(out_dir))

        # Link user input dicoms
        if not os.path.exists(out_dir):
            os.symlink(sel_dir, out_dir)


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
    sel_file = user_input_filename(
        "Select file",
        f"btn_{key_selector}",
        title_txt,
        file_search_dir,
        curr_file,
        helpmsg,
    )

    if out_file != sel_file:
        if os.path.exists(sel_file):
            # Remove existing output file
            if os.path.exists(out_file):
                os.remove(out_file)
            # Create parent dir of out file
            if not os.path.exists(os.path.dirname(out_file)):
                os.makedirs(os.path.dirname(out_file))
            # Copy selected input file to destination
            os.system(f"cp {sel_file} {out_file}")
            return True

    return False


def util_select_dir(curr_dir: str, key_txt: str) -> Any:
    """
    Panel to select a folder from the file system
    """
    tab1, tab2 = st.tabs(["Enter Path", "Browse Path"])
    with tab1:
        sel_dir = st.text_input(
            "",
            key=f"_key_sel_{key_txt}",
            value=None,
            label_visibility="collapsed",
        )
        if sel_dir is not None:
            sel_dir = os.path.abspath(sel_dir)

    with tab2:
        if st.button(
            "Browse",
            key=f"_key_btn_{key_txt}",
        ):
            sel_dir = utilio.browse_folder(curr_dir)

    if sel_dir is not None:
        sel_dir = os.path.abspath(sel_dir)
        if not os.path.exists(sel_dir):
            try:
                os.makedirs(sel_dir)
                st.info(f"Created directory: {sel_dir}")
            except:
                st.error(f"Could not create directory: {sel_dir}")
    return sel_dir


def add_debug_panel() -> None:
    """
    Displays vars used in dev phase
    """
    if not st.session_state.forced_cloud:
        st.sidebar.divider()
        st.sidebar.write("*** Debugging Flags ***")
        is_cloud_mode = st.session_state.app_type == "cloud"
        if st.sidebar.checkbox("Switch to cloud?", value=is_cloud_mode):
            st.session_state.app_type = "cloud"
        else:
            st.session_state.app_type = "desktop"

        list_vars = [
            "",
            "All",
            "plots",
            "plot_var",
            "rois",
            "paths",
            "flags",
            "checkbox",
        ]
        # list_vars = st.session_state.keys()
        sel_var = st.sidebar.selectbox("View session state vars", list_vars, index=0)
        if sel_var != "":
            with st.expander("DEBUG: Session state", expanded=True):
                if sel_var == "All":
                    st.write(st.session_state)
                else:
                    st.write(st.session_state[sel_var])
