import os
import shutil
import tkinter as tk
import zipfile
from tkinter import filedialog
from typing import Any, BinaryIO, List, Optional
import streamlit as st
import time

import pandas as pd

# https://stackoverflow.com/questions/64719918/how-to-write-streamlit-uploadedfile-to-temporary-in_dir-with-original-filenam
# https://gist.github.com/benlansdell/44000c264d1b373c77497c0ea73f0ef2
# https://stackoverflow.com/questions/65612750/how-can-i-specify-the-exact-folder-in-streamlit-for-the-uploaded-file-to-be-save


def browse_file(path_init: str) -> Any:
    """
    File selector
    Returns the file name selected by the user and the parent folder
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    out_file = filedialog.askopenfilename(initialdir=path_init)
    root.destroy()
    if len(out_file) == 0:
        return None
    return out_file

def browse_folder(path_init: str) -> Any:
    """
    Folder selector
    Returns the folder name selected by the user
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    out_path = filedialog.askdirectory(initialdir=path_init)
    root.destroy()
    if len(out_path) == 0:
        return None
    return out_path

def zip_folder(in_dir: str, f_out: str) -> Optional[bytes]:
    """
    Zips a folder and its contents.
    """
    if not os.path.exists(in_dir):
        return None
    else:
        shutil.make_archive(
            f_out, "zip", os.path.dirname(in_dir), os.path.basename(in_dir)
        )

        with open(f"{f_out}.zip", "rb") as f:
            download_dir = f.read()

        return download_dir

def unzip_zip_files(in_dir: str) -> None:
    """
    Unzips all ZIP files in the input dir and removes the original ZIP files.
    """
    if os.path.exists(in_dir):
        for filename in os.listdir(in_dir):
            if filename.endswith(".zip"):
                zip_path = os.path.join(in_dir, filename)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(in_dir)
                    os.remove(zip_path)

def copy_uploaded_file(in_file: BinaryIO, out_file: str) -> None:
    """
    Save uploaded file to the output path
    """
    if in_file is not None:
        with open(out_file, "wb") as f:
            shutil.copyfileobj(in_file, f)

def get_file_count(folder_path: str, file_suff: List[str] = []) -> int:
    """
    Returns the count of files matching any of the suffixes in `file_suff`
    within the output folder. If `file_suff` is empty, all files are counted.
    """
    count = 0
    if not os.path.exists(folder_path):
        return 0

    for root, dirs, files in os.walk(folder_path):
        if file_suff:
            count += sum(any(file.endswith(suffix) for suffix in file_suff) for file in files)
        else:
            count += len(files)

    return count

def remove_dir(dtype):
    folder_path = os.path.join(st.session_state.paths['task'], dtype)
    try:
        if os.path.islink(folder_path):
            os.unlink(folder_path)
        else:
            shutil.rmtree(folder_path)
        st.success(f"Removed dir: {folder_path}")
        time.sleep(2)
        return True
    except:
        st.error(f"Could not delete folder: {folder_path}")
        return False

def remove_file(fname):
    try:
        if os.path.islink(fname):
            os.unlink(fname)
        else:
            os.remove(fname)
        st.success(f"Removed file: {fname}")
        time.sleep(2)
        return True
    except:
        st.error(f"Could not delete file: {fname}")
        return False
    
def select_dir(out_dir: str, key_txt: str) -> Any:
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
            sel_dir = browse_folder(out_dir)

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

def copy_and_unzip_uploaded_files(in_files: list, d_out: str) -> None:
    """
    Copy uploaded files to the output dir and unzip zip files
    """
    # Save uploaded files
    print("Saving uploaded files")
    if in_files is not None:
        for in_file in in_files:
            f_out = os.path.join(d_out, in_file.name)
            if not os.path.exists(f_out):
                with open(os.path.join(d_out, in_file.name), "wb") as f:
                    f.write(in_file.getbuffer())
    # Unzip zip files
    print("Extracting zip files")
    if os.path.exists(d_out):
        unzip_zip_files(d_out)
        
def copy_uploaded_to_dir() -> None:
    '''
    Copies files to local storage
    '''
    if len(st.session_state["uploaded_input"]) > 0:
        copy_and_unzip_uploaded_files(
            st.session_state["uploaded_input"], st.session_state.paths["target_path"]
        )

def upload_file(
    out_file: str, title_txt: str, key_uploader: str
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
    copy_uploaded_file(in_file, out_file)
    return True

def select_file(out_dir: str, key_txt: str) -> Any:
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
            sel_file = browse_file(out_dir)  # FIXME

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


def upload_single_file(dtype: str, out_name: str, in_suff: str) -> None:
    """
    Upload user file to target folder
    """
    out_dir = os.path.join(
        st.session_state.paths['task'], dtype
    )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_file = os.path.join(out_dir, out_name)

    # Set target path
    st.session_state.paths["target_path"] = out_dir

    # Upload data
    with st.container(border=True):
        sel_file = st.file_uploader(
            "Input demographics file",
            key="uploaded_input_csv",
            accept_multiple_files=False
        )        
        if sel_file is not None:
            try:
                with open(out_file, "wb") as f:
                    f.write(sel_file.getbuffer())
                st.success(f"File '{sel_file.name}' saved to {out_file}")
            except:
                st.warning(f'Could not upload file: {sel_file}')


def upload_multiple_files(dtype: str) -> None:
    """
    Upload user data to target folder
    Input data may be a folder, multiple files, or a zip file (unzip the zip file if so)
    """
    out_dir = os.path.join(
        st.session_state.paths['task'], dtype
    )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Set target path
    st.session_state.paths["target_path"] = out_dir

    # Upload data
    with st.container(border=True):
        st.file_uploader(
            "Input files or folders",
            key="uploaded_input",
            accept_multiple_files=True,
            on_change=copy_uploaded_to_dir,
            help="Input files can be uploaded as a folder, multiple files, or a single zip file",
        )

def create_img_list(dtype: str) -> None:
    """
    Create a list of input images
    """
    out_dir = os.path.join(
        st.session_state.paths['task'], dtype
    )

    # Get all NIfTI files
    nifti_files = [
        f for f in os.listdir(out_dir) if f.endswith('.nii') or f.endswith('.nii.gz')
    ]

    # If no files, show warning
    if not nifti_files:
        st.warning("No NIfTI files found in the data folder.")
        return None
    else:
        # Remove common suffix to get mrid
        def remove_common_suffix(files):
            reversed_names = [f[::-1] for f in files]
            common_suffix = os.path.commonprefix(reversed_names)[::-1]
            return [f[:-len(common_suffix)] if common_suffix else f for f in files]

        mrids = remove_common_suffix(nifti_files)

        # Create the DataFrame
        df = pd.DataFrame({
            'MRID': mrids,
            'FileName': nifti_files
        })
        return df
    

def panel_input_multi(dtype: str) -> None:
    """
    Panel for selecting multiple input files or folder(s)
    """
    out_dir = os.path.join(
        st.session_state.paths['task'], dtype
    )
    with st.container(border=True):
        if st.session_state.app_type == "cloud":
            # Upload data
            upload_folder(
                out_dir,
                "Input files or folders",
                "Input files can be uploaded as a folder, multiple files, or a single zip file",
            )

        else:  # st.session_state.app_type == 'desktop'
            # Get user input
            sel_dir = select_dir(
                st.session_state.paths['init'], "sel_folder"
            )
            
            print(sel_dir)
            print(out_dir)

            if sel_dir is None:
                return False

            # Link it to out folder
            if not os.path.exists(out_dir):
                
                try:
                    os.symlink(sel_dir, out_dir)
                except:
                    st.error(
                        f"Could not link user input to destination folder: {out_dir}"
                    )
                    return False

        if get_file_count(out_dir, ['.nii', '.nii.gz']) > 0:
            return True
        
        return False
    


def get_subfolders(path: str) -> list:
    '''
    Returns a list of subfolders in input folder
    '''
    subdirs = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            subdirs.append(item)
    return sorted(subdirs)

def get_file_names(folder_path: str, file_suff: str = "") -> pd.DataFrame:
    '''
    Returns a dataframe with image names with given suffix and input path
    '''
    f_names = []
    if os.path.exists(folder_path):
        if file_suff == "":
            for root, dirs, files in os.walk(folder_path):
                f_names.append(files)
        else:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(file_suff):
                        f_names.append([file])
    df_out = pd.DataFrame(columns=["FileName"], data=f_names)
    return df_out

def get_file_list(folder_path: str, file_suff: str = "") -> List:
    '''
    Returns list of image names with given suffix and input path
    '''
    list_files: List[str] = []
    if not os.path.exists(folder_path):
        return list_files
    for f in os.listdir(folder_path):
        if f.endswith(file_suff):
            list_files.append(f)
    return list_files

def get_image_path(
    folder_path: str, file_pref: str, file_suff_list: list
) -> str:
    '''
    Returns full image name with given suffix, prefix and input path
    '''
    if not os.path.exists(folder_path):
        return ""
    for tmp_suff in file_suff_list:
        for f in os.listdir(folder_path):
            if f.startswith(file_pref) and f.endswith(tmp_suff):
                return os.path.join(folder_path, f)
    return ""
