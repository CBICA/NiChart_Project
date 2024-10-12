import os
import tkinter as tk
from tkinter import filedialog
from typing import Any

import numpy as np
import streamlit as st


def display_folder_contents(folder_path: str, parent_folder: str = "") -> None:
    """Displays the contents of a folder in a Streamlit panel with a tree structure.

    Args:
        folder_path (str): The path to the folder.
        parent_folder (str): The parent folder's name (optional).
    """

    st.title("Folder Contents")

    # Check if the folder exists
    if not os.path.exists(folder_path):
        st.error(f"Folder '{folder_path}' does not exist.")
        return

    # Get a list of files and directories in the folder
    contents = os.listdir(folder_path)

    # Create a container for the folder contents
    container = st.container()

    # Display the parent folder name
    if parent_folder:
        container.markdown(f"**{parent_folder}**")

    # Iterate over the contents and display them
    for item in contents:
        item_path = os.path.join(folder_path, item)

        # Check if the item is a file or a directory
        if os.path.isfile(item_path):
            # Display the file name with indentation based on the parent folder
            file_name = os.path.basename(item_path)
            file_url = f"download/{file_name}"  # Adjust the download URL as needed
            container.markdown(
                f"{'  ' * len(parent_folder.split('/'))}[Download]({file_url}) {file_name}"
            )
        else:
            # Display the directory name with indentation and a link to explore it
            directory_name = os.path.basename(item_path)
            container.markdown(
                f"{'  ' * len(parent_folder.split('/'))}[Explore]({directory_name}) {directory_name}"
            )

            # Recursively display the contents of the subdirectory
            display_folder_contents(item_path, parent_folder=directory_name)


def display_folder(in_dir: str) -> None:
    """
    Displays the contents of a folder in a Streamlit panel.
    """

    st.title("Folder Contents")

    # Check if the folder exists
    if not os.path.exists(in_dir):
        st.error(f"Folder '{in_dir}' does not exist.")
        return

    # Get a list of files and directories in the folder
    contents = os.listdir(in_dir)

    # Create a container for the folder contents
    container = st.container()

    # Iterate over the contents and display them
    for item in contents:
        item_path = os.path.join(in_dir, item)

        # Check if the item is a file or a directory
        if os.path.isfile(item_path):
            # Display the file name with a link to download it
            file_name = os.path.basename(item_path)
            file_url = f"download/{file_name}"  # Adjust the download URL as needed
            container.write(f"[Download]({file_url}) {file_name}")
        else:
            # Display the directory name with a link to explore it
            directory_name = os.path.basename(item_path)
            container.write(f"[Explore]({directory_name}) {directory_name}")


def browse_file(path_init: str) -> Any:
    """
    File selector
    Returns the file name selected by the user and the parent folder
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    out_path = filedialog.askopenfilename(initialdir=path_init)
    path_out = os.path.dirname(out_path)
    root.destroy()
    return out_path, path_out


def browse_folder(path_init: str) -> str:
    """
    Folder selector
    Returns the folder name selected by the user
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    out_path = filedialog.askdirectory(initialdir=path_init)
    root.destroy()
    return out_path


def user_input_text(label: str, init_val: str, help_msg: str) -> Any:
    """
    St text field to read a text input from the user
    """
    tmpcol = st.columns((1, 8))
    with tmpcol[0]:
        user_sel = st.text_input(label, value=init_val, help=help_msg)
        return user_sel


def user_input_file(
    label_btn: Any,
    key_btn: Any,
    label_txt: str,
    init_path_dir: str,
    init_path_curr: str,
    help_msg: str,
) -> Any:
    """
    St button + text field to read an input file path from the user
    """
    path_curr = init_path_curr
    path_dir = init_path_dir
    tmpcol = st.columns((8, 1))
    with tmpcol[1]:
        if st.button(label_btn, key=key_btn):
            path_curr, path_dir = browse_file(path_dir)

    with tmpcol[0]:
        tmp_sel = st.text_input(label_txt, value=path_curr, help=help_msg)
        if os.path.exists(tmp_sel):
            path_curr = tmp_sel
    return path_curr, path_dir


def user_input_folder(
    label_btn: Any,
    key_btn: Any,
    label_txt: str,
    init_path_dir: str,
    init_path_curr: str,
    help_msg: str,
) -> str:
    """
    St button + text field to read an input directory path from the user
    """
    path_curr = init_path_curr
    path_dir = init_path_dir
    tmpcol = st.columns((8, 1))
    with tmpcol[1]:
        if st.button(label_btn, key=key_btn):
            if path_curr == "":
                path_curr = browse_folder(path_dir)
            else:
                path_curr = browse_folder(path_curr)

    with tmpcol[0]:
        path_curr = st.text_input(label_txt, value=path_curr, help=help_msg)
    return path_curr


def user_input_select(label: Any, selections: Any, key: Any, helpmsg: str) -> Any:
    """
    St selection box to selet a text from the user
    """
    tmpcol = st.columns((1, 8))
    with tmpcol[0]:
        user_sel = st.selectbox(label, selections, key=key, help=helpmsg)
    return user_sel


def show_img3D(
    img: np.ndarray, scroll_axis: Any, sel_axis_bounds: Any, img_name: str
) -> None:
    """
    Displays a 3D img
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
    if scroll_axis == 0:
        st.image(img[slice_index, :, :], use_column_width=True)
    elif scroll_axis == 1:
        st.image(img[:, slice_index, :], use_column_width=True)
    else:
        st.image(img[:, :, slice_index], use_column_width=True)


def update_default_paths() -> None:
    """
    Update default paths if the working dir changed
    """
    for d_tmp in st.session_state.dict_paths.keys():
        st.session_state.paths[d_tmp] = os.path.join(
            st.session_state.paths["dset"],
            st.session_state.dict_paths[d_tmp][0],
            st.session_state.dict_paths[d_tmp][1],
        )
        print(f"setting {st.session_state.paths[d_tmp]}")

    st.session_state.paths["csv_seg"] = os.path.join(
        st.session_state.paths["dset"], "DLMUSE", "DLMUSE_Volumes.csv"
    )

    st.session_state.paths["csv_mlscores"] = os.path.join(
        st.session_state.paths["dset"],
        "MLScores",
        f"{st.session_state.dset_name}_DLMUSE+MLScores.csv",
    )

    st.session_state.paths["csv_demog"] = os.path.join(
        st.session_state.paths["dset"], "Lists", "Demog.csv"
    )


def util_panel_workingdir(app_type: str) -> None:

    # Panel for selecting the working dir
    with st.expander(":outbox_tray: Working Dir", expanded=False):

        curr_dir = st.session_state.paths["dset"]

        if app_type == "DESKTOP":
            # Read output folder from the user
            helpmsg = "Results will be saved to the output folder.\n\nChoose the path by typing it into the text field or using the file browser to browse and select it"
            st.session_state.paths["out"] = user_input_folder(
                "Select folder",
                "btn_sel_out_dir",
                "Output folder",
                st.session_state.paths["last_sel"],
                st.session_state.paths["out"],
                helpmsg,
            )

        # Read dataset name (used to create a folder where all results will be saved)
        helpmsg = "Each dataset's results are organized in a dedicated folder named after the dataset"
        st.session_state.dset_name = user_input_text(
            "Dataset name", st.session_state.dset_name, helpmsg
        )

        if st.session_state.dset_name != "":
            st.session_state.paths["dset"] = os.path.join(
                st.session_state.paths["out"], st.session_state.dset_name
            )

        # Dataset output folder name changed
        if curr_dir != st.session_state.paths["dset"]:

            # Create output folder
            if not os.path.exists(st.session_state.paths["dset"]):
                os.makedirs(st.session_state.paths["dset"])

            # Update paths for output subfolders
            update_default_paths()

            st.success(
                f"All results will be saved to: {st.session_state.paths['dset']}",
                icon=":material/thumb_up:",
            )


