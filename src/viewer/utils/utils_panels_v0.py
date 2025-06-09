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


def util_sel_task(out_dir: str, task_curr: str) -> Any:
    """
    Set/select task name
    """
    # Read existing task folders
    list_tasks = utilio.get_subfolders(out_dir)
    st.write("Select Existing")
    sel_opt = st.pills(
        "Options:",
        options = list_tasks,
        default = st.session_state.navig['task'],
        label_visibility="collapsed",
    )

    st.write("Create New")
    task_sel = st.text_input(
        "Experiment name:",
        None,
        label_visibility="collapsed",
        placeholder="Type task name",
    )
    if st.button('Create task'):
        if task_sel is not None:
            dir_tmp = os.path.join(out_dir, task_sel)
            if not os.path.exists(dir_tmp):
                os.makedirs(dir_tmp)

            sel_opt = task_sel

    return sel_opt

def util_panel_task() -> None:
    """
    Panel for selecting task name
    """
    with st.container(border=True):
        if st.session_state.flags["task"]:
            st.success(
                f"Task name: {st.session_state.navig['task']}",
                icon=":material/thumb_up:",
            )
            if st.button("Reset", key="reset_exp"):
                st.session_state.flags["task"] = False
                st.session_state.navig['task'] = None
                #st.rerun()

        else:
            out_dir = st.session_state.paths["out_dir"]
            task_curr = st.session_state.navig['task']
            task_sel = util_sel_task(out_dir, task_curr)

            if task_sel is not None and task_sel != task_curr:
                st.session_state.navig['task'] = task_sel
                st.session_state.paths["task"] = os.path.join(
                    st.session_state.paths["out_dir"], task_sel
                )
                st.session_state.flags["task"] = True
                # Update paths when selected task changes
                utilss.update_default_paths()
                utilss.reset_flags()
                #st.rerun()

        util_help_dialog(utildoc.title_exp, utildoc.def_exp)


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
    out_dir: str,
    file_search_dir: str,
    flag_disabled: bool,
) -> None:
    """
    Select user input folder and link to target folder
    """
    # Check if out folder already exists
    out_dir = ""
    if os.path.exists(out_dir):
        fcount = utilio.get_file_count(out_dir)
        if fcount > 0:
            out_dir = out_dir

    # Upload data
    helpmsg = "Select input folder.\n\nChoose the path by typing it into the text field or using the file browser to browse and select it"
    sel_dir = user_input_foldername(
        "Select folder",
        f"btn_{key_selector}",
        title_txt,
        file_search_dir,
        out_dir,
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


def util_select_dir(out_dir: str, key_txt: str) -> Any:
    """
    Panel to select a folder from the file system
    """
    sel_dir = None
    sel_opt = st.radio(
        "Options:",
        ["Browse Path", "Enter Path"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if sel_opt == "Browse Path":
        if st.button(
            "Browse",
            key=f"_key_btn_{key_txt}",
        ):
            sel_dir = utilio.browse_folder(out_dir)

    elif sel_opt == "Enter Path":
        sel_dir = st.text_input(
            "",
            key=f"_key_sel_{key_txt}",
            value=out_dir,
            label_visibility="collapsed",
        )
        if sel_dir is not None:
            sel_dir = os.path.abspath(sel_dir)

    if sel_dir is not None:
        sel_dir = os.path.abspath(sel_dir)
        if not os.path.exists(sel_dir):
            try:
                os.makedirs(sel_dir)
                st.info(f"Created directory: {sel_dir}")
            except:
                st.error(f"Could not create directory: {sel_dir}")
    return sel_dir

def util_panel_input_multi(dtype: str) -> None:
    """
    Panel for selecting multiple input files or folder(s)
    """
    with st.container(border=True):
        # Check if data exists
        out_dir = os.path.join(
            st.session_state.paths['task'], dtype
        )
        if os.path.exists(out_dir):
            st.success(
                f"Out dir exists: {out_dir}",
                icon=":material/thumb_up:",
            )
            # Delete folder if user wants to reload
            if st.button("Reset", f"key_btn_reset_{dtype}"):
                try:
                    if os.path.islink(out_dir):
                        os.unlink(out_dir)
                    else:
                        shutil.rmtree(out_dir)
                    st.success(f"Removed dir: {out_dir}")
                    time.sleep(4)
                except:
                    st.error(f"Could not delete folder: {out_dir}")
                #st.rerun()

        else:
            if st.session_state.app_type == "cloud":
                # Upload data
                util_upload_folder(
                    out_dir,
                    "Input files or folders",
                    False,
                    "Input files can be uploaded as a folder, multiple files, or a single zip file",
                )

            else:  # st.session_state.app_type == 'desktop'
                # Get user input
                sel_dir = util_select_dir(out_dir, "sel_folder")
                if sel_dir is None:
                    return

                # Link it to out folder
                if not os.path.exists(out_dir):
                    try:
                        os.symlink(sel_dir, out_dir)
                    except:
                        st.error(
                            f"Could not link user input to destination folder: {out_dir}"
                        )

            # Check out files
            fcount = utilio.get_file_count(out_dir)
            if fcount > 0:
                st.success(
                    f" Uploaded data: ({fcount} files)",
                    icon=":material/thumb_up:",
                )
                time.sleep(4)

                #st.rerun()



#def util_panel_input_multi(dtype: str, status: bool) -> None:
    #"""
    #Panel for selecting multiple input files or folder(s)
    #"""
    #with st.container(border=True):
        ## Check init status
        #if not status:
            #st.warning("Please check previous step!")
            #return

        ## Check if data exists
        #out_dir = st.session_state.paths[dtype]
        #if st.session_state.flags[dtype]:
            #st.success(
                #f"Data is ready: {out_dir}",
                #icon=":material/thumb_up:",
            #)
            ## Delete folder if user wants to reload
            #if st.button("Reset", f"key_btn_reset_{dtype}"):
                #try:
                    #if os.path.islink(out_dir):
                        #os.unlink(out_dir)
                    #else:
                        #shutil.rmtree(out_dir)
                    #st.session_state.flags[dtype] = False
                    #st.success(f"Removed dir: {out_dir}")
                    #time.sleep(4)
                #except:
                    #st.error(f"Could not delete folder: {out_dir}")
                #st.rerun()

        #else:
            ## Create parent dir for out data
            #dbase = os.path.dirname(out_dir)
            #print(dbase)
            #print('aaa')
            #if not os.path.exists(dbase):
                #os.makedirs(dbase)

            #if st.session_state.app_type == "cloud":
                ## Upload data
                #util_upload_folder(
                    #st.session_state.paths["dicoms"],
                    #"Input files or folders",
                    #False,
                    #"Input files can be uploaded as a folder, multiple files, or a single zip file",
                #)

            #else:  # st.session_state.app_type == 'desktop'
                ## Get user input
                #sel_dir = util_select_dir(out_dir, "sel_folder")
                #if sel_dir is None:
                    #return

                ## Link it to out folder
                #if not os.path.exists(out_dir):
                    #try:
                        #os.symlink(sel_dir, out_dir)
                    #except:
                        #st.error(
                            #f"Could not link user input to destination folder: {out_dir}"
                        #)

            ## Check out files
            #fcount = utilio.get_file_count(st.session_state.paths[dtype])
            #if fcount > 0:
                #st.session_state.flags[dtype] = True
                #p_dicom = st.session_state.paths[dtype]
                #st.success(
                    #f" Uploaded data: ({p_dicom}, {fcount} files)",
                    #icon=":material/thumb_up:",
                #)
                #time.sleep(4)

                #st.rerun()
            #util_help_dialog(utildoc.title_dicoms, utildoc.def_dicoms)


def util_select_file(out_dir: str, key_txt: str) -> Any:
    """
    Select a file
    """
    sel_file = None
    sel_opt = st.radio(
        "Options:",
        ["Browse File", "Enter File Path"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if sel_opt == "Browse File":
        if st.button("Browse", key=f"_key_btn_{key_txt}"):
            sel_file = utilio.browse_file(out_dir)  # FIXME

    elif sel_opt == "Enter Path":
        sel_file = st.text_input(
            "",
            key=f"_key_sel_{key_txt}",
            value=out_dir,
            label_visibility="collapsed",
        )
    if sel_file is None:
        return

    sel_file = os.path.abspath(sel_file)
    if not os.path.exists(sel_file):
        return

    return sel_file


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


def util_panel_input_single(dtype: str, status: bool) -> None:
    """
    Panel for selecting single input file
    """
    with st.container(border=True):
        # Check init status
        if not status:
            st.warning("Please check previous step!")
            return

        # Check if data exists
        fout = st.session_state.paths[dtype]
        if st.session_state.flags[dtype]:
            st.success(
                f"Data is ready: {fout}",
                icon=":material/thumb_up:",
            )
            # Delete data if user wants to reload
            if st.button("Reset", f"key_btn_reset_{dtype}"):
                try:
                    shutil.rmtree(fout)
                    st.session_state.flags[dtype] = False
                    st.success(f"Removed file: {fout}")
                    time.sleep(4)
                except:
                    st.error(f"Could not delete file: {fout}")
                #st.rerun()

        else:
            # Create parent dir for out data
            dbase = os.path.dirname(fout)
            if not os.path.exists(dbase):
                os.makedirs(dbase)

            if st.session_state.app_type == "cloud":
                # Upload data
                util_upload_file(
                    st.session_state.paths[dtype],
                    "input data",
                    "upload_data_file",
                    False,
                    "visible",
                )

            else:  # st.session_state.app_type == 'desktop'
                # Get user input
                sel_file = util_select_file(
                    # st.session_state.paths['init'], 'sel_file', key=f'key_self_{dtype}'
                    st.session_state.paths["init"],
                    "sel_file",
                )
                if sel_file is None:
                    return

                # Copy file
                try:
                    shutil.copy2(sel_file, fout)
                except:
                    st.error(f"Could not copy input file to destination: {fout}")

            # Check out files
            if os.path.exists(fout):
                st.session_state.flags[dtype] = True
                st.success(
                    f" Uploaded data: {fout}",
                    icon=":material/thumb_up:",
                )
                time.sleep(4)

                #st.rerun()
            util_help_dialog(utildoc.title_dicoms, utildoc.def_dicoms)


def util_panel_download(dtype: str, status: bool) -> None:
    """
    Panel for downloading results
    """
    with st.container(border=True):
        # Check init status
        if not status:
            st.warning("Please check previous step!")
            return

        # Zip results and download
        out_zip = bytes()
        out_dir = st.session_state.paths["download"]
        in_dir = st.session_state.paths[dtype]
        if not os.path.exists(in_dir):
            st.error("Input data missing!")
            return
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        try:
            f_tmp = os.path.join(out_dir, dtype)
            out_zip = utilio.zip_folder(in_dir, f_tmp)
            st.download_button(
                f"Download results: {dtype}",
                out_zip,
                file_name=f"{st.session_state.navig['task']}_{dtype}.zip",
            )
        except:
            st.error("Could not download data!")

def panel_view_mri(config) -> None:
    """
    Panel for viewing mri images
    """
    with st.container(border=True):

        # Select MRID
        sel_mrid = st.selectbox(
            "MRID",
            config['list_mrid'],
            key="_key_selbox_mrid",
            index=None,
        )
        if sel_mrid is None:
            return

        # Select target ROI
        sel_var = st.selectbox(
            "ROI",
            config['list_roi'],
            key="_key_selbox_rois",
            index=None
        )
        if sel_var is None:
            return

        # Select orientation(s)
        list_orient = st.multiselect(
            "Select viewing planes:", utilni.img_views, utilni.img_views, disabled=False
        )        
        if list_orient is None or len(list_orient) == 0:
            st.warning("Please select the viewing plane!")
            return
        
        # View hide overlay
        is_show_overlay = st.checkbox("Show overlay", True, disabled=False)

        # Crop to mask area
        crop_to_mask = st.checkbox("Crop to mask", True, disabled=False)
        
        # Get indices for the selected var
        list_rois = utilroi.get_list_rois(
            sel_var,
            st.session_state.rois["roi_dict_inv"],
            st.session_state.rois["roi_dict_derived"],
        )
        
        img_ulay = utilio.get_image_path(
            config['dir_ulay'], sel_mrid, ["nii.gz", ".nii"]
        )

        img_olay = utilio.get_image_path(
            config['dir_olay'], sel_mrid, ["nii.gz", ".nii"]
        )

        print('aaa')
        print(config['dir_ulay'])
        print(sel_mrid)
        print('aaa')

        with st.spinner("Wait for it..."):
            # Process image and mask to prepare final 3d matrix to display
            img, mask, img_masked = utilni.prep_image_and_olay(
                img_ulay,
                img_olay,
                list_rois,
                crop_to_mask,
            )

            # Detect mask bounds and center in each view
            mask_bounds = utilni.detect_mask_bounds(mask)

            # Show images
            blocks = st.columns(len(list_orient))
            for i, tmp_orient in stqdm(
                enumerate(list_orient),
                desc="Showing images ...",
                total=len(list_orient),
            ):
                with blocks[i]:
                    ind_view = utilni.img_views.index(tmp_orient)
                    size_auto = True
                    if is_show_overlay is False:
                        utilst.show_img3D(
                            img,
                            ind_view,
                            mask_bounds[ind_view, :],
                            tmp_orient,
                            size_auto,
                        )
                    else:
                        utilst.show_img3D(
                            img_masked,
                            ind_view,
                            mask_bounds[ind_view, :],
                            tmp_orient,
                            size_auto,
                        )


def panel_select_single(list_options, def_val, msg_txt, key) -> None:
    if msg_txt != '':
        st.markdown(f'##### {msg_txt}')
    
    sel_items = st.pills(
        'Select Item',
        list_options,
        default = def_val,
        selection_mode = 'single',
        label_visibility = 'collapsed',
        key = f'_key_pills_{key}'
    )
    
    return sel_items

def panel_select_multi(list_options, msg_txt, key) -> None:
    if msg_txt != '':
        st.markdown(f'##### {msg_txt}')    
    
    list_sel = []
    if st.checkbox('Select/Deselect All', key = f'_key_check_{key}'):
        list_sel = list_options
        
    sel_items = st.pills(
        'Select Item',
        list_options,
        default = list_sel,
        selection_mode = 'multi',
        label_visibility="collapsed",
        key = f'_key_pills_{key}'
    )
    
    return sel_items
