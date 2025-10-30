import streamlit as st
import utils.utils_dicoms as utildcm
import utils.utils_session as utilss
import os
import pandas as pd
import numpy as np
import zipfile
import streamlit_antd_components as sac
import shutil
import time
from typing import Any, BinaryIO, List, Optional

from utils.utils_logger import setup_logger
logger = setup_logger()

##############################################################
## Generic IO functions
def get_file_count(folder_path: str, file_suff: List[str] = []) -> int:
    '''
    Returns the count of files matching any of the suffixes in `file_suff`
    within the output folder. If `file_suff` is empty, all files are counted.
    '''
    count = 0
    if not os.path.exists(folder_path):
        return 0

    for root, dirs, files in os.walk(folder_path):
        if file_suff:
            count += sum(any(file.endswith(suffix) for suffix in file_suff) for file in files)
        else:
            count += len(files)

    return count

def get_file_names(folder_path: str, file_suff: str = "") -> pd.DataFrame:
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

def remove_dir(out_dir):
    '''
    Delete a folder
    '''
    try:
        if os.path.islink(out_dir):
            os.unlink(out_dir)
        else:
            shutil.rmtree(out_dir)
        st.success(f"Removed dir: {out_dir}")
        time.sleep(2)
        return True
    except:
        st.error(f"Could not delete folder: {out_dir}")
        return False

def browse_file(path_init: str) -> Any:
    '''
    File selector
    Returns the file name selected by the user and the parent folder
    '''
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    out_file = filedialog.askopenfilename(initialdir=path_init)
    root.destroy()
    if len(out_file) == 0:
        return None
    return out_file

def browse_folder(path_init: str) -> Any:
    '''
    Folder selector
    Returns the folder name selected by the user
    '''
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    out_path = filedialog.askdirectory(initialdir=path_init)
    root.destroy()
    if len(out_path) == 0:
        return None
    return out_path

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

def zip_folder(in_dir: str, f_out: str) -> Optional[bytes]:
    '''
    Zips a folder and its contents.
    '''
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
    '''
    Unzips all ZIP files in the input dir and removes the original ZIP files.
    '''
    if os.path.exists(in_dir):
        for filename in os.listdir(in_dir):
            if filename.endswith(".zip"):
                zip_path = os.path.join(in_dir, filename)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(in_dir)
                    os.remove(zip_path)

def copy_and_unzip_uploaded_files(in_files: list, d_out: str) -> None:
    '''
    Copy uploaded files to the output dir and unzip zip files
    '''
    # Save uploaded files
    print("Saving uploaded files")
    if in_files is not None:
        for in_file in in_files:
            f_out = os.path.join(d_out, in_file.name)
            # Handle creating nested dirs if needed
            os.makedirs(os.path.dirname(f_out), exist_ok=True)
            if not os.path.exists(f_out):
                with open(os.path.join(d_out, in_file.name), "wb") as f:
                    f.write(in_file.getbuffer())
    # Unzip zip files
    print("Extracting zip files")
    if os.path.exists(d_out):
        unzip_zip_files(d_out)

def upload_files(out_dir, flag_single = False):
    '''
    Upload user data to target folder
    Input data may be a folder, multiple files, or a zip file (unzip the zip file if so)
    '''
    # Set target path
    st.session_state.paths["target"] = out_dir

    with st.form(key='my_form', clear_on_submit=True):

        sel_mod = sac.chip(
            items=[
                sac.ChipItem(label='T1'),
                sac.ChipItem(label='FL'),
                sac.ChipItem(label='CSV'),
            ], label='Data type', index=0, align='left', size='md', radius='md', multiple=False, color='cyan', 
            #description='Select type of your data type'
        )    
        out_path = None
        if sel_mod is not None:
            out_path = os.path.join(out_dir, sel_mod.lower())
            
        sel_files = st.file_uploader(
            "Input files or folders",
            key="_uploaded_input",
            accept_multiple_files=flag_single,
            help="Input files can be uploaded as a folder, multiple files, or a single zip file",
        )
        
        submitted = st.form_submit_button("Submit")
        if submitted:
            if out_path is None:
                return False
            if len(sel_files) == 0:
                return False
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            copy_and_unzip_uploaded_files(sel_files, out_path)
            st.info('Uploaded file')
            return True
    
    return False
        
        
def panel_load_data():
    st.markdown('##### User Input:')
    upload_files(st.session_state.paths['project'], True)

def panel_load_data_tmp():
    st.markdown('##### User Input:')
    sel_mod = sac.chip(
        items=[
            sac.ChipItem(label='T1'),
            sac.ChipItem(label='FL'),
            sac.ChipItem(label='Demog'),
        ], label='label', index=None, align='left', size='md', radius='md', multiple=False, color='cyan'
    )    
    if sel_mod is not None:
        out_dir = os.path.join(
            st.session_state.paths['project'], sel_mod.lower()
        )
        if upload_files(out_dir, True):
            st.info('Hello')
            sel_mod=None
            st.rerun()
