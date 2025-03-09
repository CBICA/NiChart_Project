import os
import shutil
from typing import Any

import numpy as np
import streamlit as st
import utils.utils_io as utilio
import utils.utils_session as utilses
import utils.utils_doc as utildoc

import time


COL_LEFT = 5
COL_RIGHT_EMPTY = 0.01
COL_RIGHT_BUTTON = 1

def util_help_dialog(s_title: str, s_text: str) -> None:
    """
    Display help dialog box
    """
    @st.dialog(s_title)  # type:ignore
    def help_working_dir():
        st.markdown(s_text)

    col1, col2 = st.columns([0.5, 0.1])
    with col2:
        if st.button(
            "Get help 🤔", key="key_btn_help_" + s_title, use_container_width=True
        ):
            help_working_dir()

def util_select_expname(dir_out: str, exp_curr:str) -> str:
    """
    Set/select experiment name
    """
    # Read dataset name (used to create a folder where all results will be saved)
    sel_opt = st.radio(
        'Options:',
        ['Select Existing', 'Create New'],
        horizontal = True,
        label_visibility = 'collapsed'
    )

    if sel_opt == 'Select Existing':
        list_exp = utilio.get_subfolders(dir_out)
        exp_sel = st.selectbox(
            "Select",
            list_exp,
            None,
            label_visibility = 'collapsed',
            placeholder='Select experiment name'
        )

    if sel_opt == 'Create New':
        exp_sel = st.text_input(
            "Experiment name:",
            None,
            label_visibility = 'collapsed',
            placeholder='Type experiment name'
        )
        if exp_sel is not None:
            dir_tmp = os.path.join(dir_out, exp_sel)
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
            if st.button('Reset', key = 'reset_exp'):
                st.session_state.flags["experiment"] = False
                st.session_state.experiment = ''
                st.rerun()

        else:
            dir_out = st.session_state.paths["dir_out"]
            exp_curr = st.session_state.experiment
            exp_sel = util_select_expname(dir_out, exp_curr)

            if exp_sel is not None and exp_sel != exp_curr:
                st.session_state.experiment = exp_sel
                st.session_state.paths["experiment"] = os.path.join(
                    st.session_state.paths["dir_out"], exp_sel
                )
                st.session_state.flags["experiment"] = True
                # Update paths when selected experiment changes
                utilses.update_default_paths()
                utilses.reset_flags()
                st.rerun()

        util_help_dialog(utildoc.title_exp, utildoc.def_exp)

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

    if sel_dir is not None and os.path.exists(sel_dir):
        # Remove existing output folder
        if os.path.exists(dir_out) and dir_out != sel_dir:
            if os.path.islink(dir_out):
                os.unlink(dir_out)
            else:
                shutil.rmtree(dir_out)

        # Create parent dir of output dir
        if not os.path.exists(os.path.dirname(dir_out)):
            os.makedirs(os.path.dirname(dir_out))

        # Link user input dicoms
        if not os.path.exists(dir_out):
            os.symlink(sel_dir, dir_out)

def util_select_dir(curr_dir, key_txt) -> None:
    """
    Panel to select a folder from the file system
    """
    sel_dir = None
    sel_opt = st.radio(
        'Options:',
        ['Browse Path', 'Enter Path'],
        horizontal = True,
        label_visibility = 'collapsed'
    )
    if sel_opt == 'Browse Path':
        if st.button(
            'Browse',
            key=f'_key_btn_{key_txt}',
        ):
            sel_dir = utilio.browse_folder(curr_dir)

    elif sel_opt == 'Enter Path':
        sel_dir = st.text_input(
            '',
            key=f'_key_sel_{key_txt}',
            value=curr_dir,
            label_visibility='collapsed',
        )
        if sel_dir is not None:
            sel_dir = os.path.abspath(sel_dir)

    if sel_dir is not None:
        sel_dir = os.path.abspath(sel_dir)
        if not os.path.exists(sel_dir):
            try:
                os.makedirs(sel_dir)
                st.info(f'Created directory: {sel_dir}')
            except:
                st.error(f'Could not create directory: {sel_dir}')
    return sel_dir

def util_panel_input_multi(dtype: str, status:bool) -> None:
    """
    Panel for selecting multiple input files or folder(s)
    """
    with st.container(border=True):
        # Check init status
        if not status:
            st.warning('Please check previous step!')
            return

        # Check if data exists
        dout = st.session_state.paths[dtype]
        if st.session_state.flags[dtype]:
            st.success(
                f"Data is ready: {dout}",
                icon=":material/thumb_up:",
            )
            # Delete folder if user wants to reload
            if st.button('Reset'):
                try:
                    if os.path.islink(dout):
                        os.unlink(dout)
                    else:
                        shutil.rmtree(dout)
                    st.session_state.flags[dtype] = False
                    st.success(f'Removed dir: {dout}')
                    time.sleep(4)
                except:
                    st.error(f'Could not delete folder: {dout}')
                st.rerun()

        else:
            if st.session_state.app_type == "cloud":
                # Upload data
                util_upload_folder(
                    st.session_state.paths["dicoms"],
                    "Input files or folders",
                    False,
                    "Raw dicom files can be uploaded as a folder, multiple files, or a single zip file",
                )

            else:  # st.session_state.app_type == 'desktop'
                # Get user input
                sel_dir = util_select_dir(dout, 'sel_folder')
                if sel_dir is None:
                    return

                # Link it to out folder
                if not os.path.exists(dout):
                    try:
                        os.symlink(sel_dir, dout)
                    except:
                        st.error(f'Could not link user input to destination folder: {dout}')

            # Check out files
            fcount = utilio.get_file_count(st.session_state.paths[dtype])
            if fcount > 0:
                st.session_state.flags[dtype] = True
                p_dicom = st.session_state.paths[dtype]
                st.success(
                    f" Uploaded data: ({p_dicom}, {fcount} files)",
                    icon=":material/thumb_up:",
                )
                time.sleep(4)

                st.rerun()
            util_help_dialog(utildoc.title_dicoms, utildoc.def_dicoms)


def util_panel_download(dtype:str, status:bool) -> None:
    """
    Panel for downloading results
    """
    with st.container(border=True):
        # Check init status
        if not status:
            st.warning('Please check previous step!')
            return

        # Zip results and download
        out_zip = bytes()
        out_dir = st.session_state.paths["download"]
        in_dir = st.session_state.paths[dtype]
        if not os.path.exists(in_dir):
            st.error('Input data missing!')
            return
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        try:
            f_tmp = os.path.join(out_dir, dtype)
            out_zip = utilio.zip_folder(in_dir, f_tmp)
            st.download_button(
                f'Download results: {dtype}',
                out_zip,
                file_name=f"{st.session_state.experiment}_{dtype}.zip",
            )
        except:
            st.error('Could not download data!')

