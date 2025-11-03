import streamlit as st
import utils.utils_dicoms as utildcm
import utils.utils_session as utilss
import utils.utils_data_view as utildv
import utils.utils_io as utilio

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

def zip_folder(out_dir: str, f_out: str) -> Optional[bytes]:
    '''
    Zips a folder and its contents.
    '''
    if not os.path.exists(out_dir):
        return None
    else:
        shutil.make_archive(
            f_out, "zip", os.path.dirname(out_dir), os.path.basename(out_dir)
        )

        with open(f"{f_out}.zip", "rb") as f:
            download_dir = f.read()

        return download_dir

def unzip_files(out_dir: str) -> None:
    '''
    Unzips all ZIP files in the input dir and removes the original ZIP files.
    '''
    if os.path.exists(out_dir):
        for filename in os.listdir(out_dir):
            if filename.endswith(".zip"):
                zip_path = os.path.join(out_dir, filename)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(out_dir)
                    os.remove(zip_path)

def copy_and_unzip_single(in_file: str, d_out: str) -> None:
    '''
    Copy uploaded files to the output dir and unzip zip files
    '''
    # Save uploaded files
    print("Saving uploaded files")
    fout = os.path.join(d_out, in_file.name)
    # Handle creating nested dirs if needed
    os.makedirs(os.path.dirname(fout), exist_ok=True)
    if not os.path.exists(fout):
        with open(fout, "wb") as f:
            f.write(in_file.getbuffer())
    ## Unzip zip files
    #print("Extracting zip files")
    #if os.path.exists(d_out):
        #unzip_files(d_out)

def copy_nifti(in_file_obj: str, d_out: str) -> None:
    '''
    Copy uploaded nifti image to the output dir
    '''
    fout = os.path.join(d_out, in_file_obj.name)
    # Handle creating nested dirs if needed
    os.makedirs(os.path.dirname(fout), exist_ok=True)
    if not os.path.exists(fout):
        with open(fout, "wb") as f:
            f.write(in_file_obj.getbuffer())
    ## Unzip zip files
    #print("Extracting zip files")
    #if os.path.exists(d_out):
        #unzip_files(d_out)

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

def copy_uploaded_files(in_files: list, d_out: str) -> None:
    '''
    Copy files to output folder and unzip zip files
    '''
    print("Copying uploaded files ...")
    if in_files is not None:
        for in_file in in_files:
            f_out = os.path.join(d_out, in_file.name)
            # Handle creating nested dirs if needed
            os.makedirs(os.path.dirname(f_out), exist_ok=True)
            if not os.path.exists(f_out):
                with open(os.path.join(d_out, in_file.name), "wb") as f:
                    f.write(in_file.getbuffer())
    # Unzip zip files
    print("Extracting zip files ...")
    if os.path.exists(d_out):
        unzip_zip_files(d_out)

def edit_participants(out_dir, fname):
    fpath = os.path.join(out_dir, fname)

    if not os.path.exists(fpath):
        return

    #sac.divider(key='_p2_div1')
    with st.container(horizontal=True, horizontal_alignment="left"):
        st.markdown("##### Edit Subject List: ", width='content')
        st.markdown(f"##### 📃 `{fname}`", width='content')

    df = pd.read_csv(fpath, dtype={'MRID':str, 'Age':float, 'Sex':str})
    
    # Define column options
    column_config = {
        "Age": st.column_config.NumberColumn(
            "Age",
            help="Select age",
            min_value=0,
            max_value=110,
            step=0.1,
            required=True
        ),
        "Sex": st.column_config.SelectboxColumn(
            "Sex",
            help="Select sex",
            options=["M", "F", "Other"],
            required=True
        )
    }
    df_user = st.data_editor(
        df,
        column_config=column_config,
        num_rows="fixed",
        use_container_width=True
    )
    if st.button('Save'):
        df_user.to_csv(fpath, index=False)
        st.success(f'Updated demographic file: {fpath}')


def select_project():
    """
    Panel for selecting a project
    """

@st.dialog("File viewer", width='medium')
def show_sel_item(fname):
        if fname.endswith('.csv'):
            try:
                df_tmp = pd.read_csv(fname)
                st.info(f'Data file: {fname}')
                st.dataframe(df_tmp)
            except:
                st.warning(f'Could not read csv file: {fname}')


def clear_folder(folder):
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

def panel_project_folder():
    '''
    Panel to select project folder
    '''
    sac.divider(key='_p1_div1')
    
    with st.container(horizontal=True, horizontal_alignment="left"):
        st.markdown("##### Project Folder: ", width='content')
        with st.popover("❓", width='content'):
            st.write(
                """
                **Project Folder Help**
                - All processing steps are performed inside a project folder.
                - By default, NiChart will create and use a current project folder for you.
                - You may also create a new project folder using any name you choose.
                - If needed, you can reset the current project folder (this will remove all files inside it, but keep the folder itself), allowing you to start fresh.
                - You may also switch to an existing project folder.
                
                **Note:** If you are using the cloud version, stored files will be removed periodically, so previously used project folders might not remain available.                
                """
            )
            
    placeholder = st.empty()
    placeholder.markdown(f"##### 📁 `{st.session_state.prj_name}`", width='content')

    sel_opt = st.selectbox(
        'Select an action',
        ['Create new project folder', 'Switch to existing project', 'Reset project folder'],
        label_visibility='collapsed',
        index=None
    )

    if sel_opt == 'Create new project folder':
        sel_prj = st.text_input(
            "Project name:",
            None,
            placeholder="user_new_study",
            label_visibility = 'collapsed'
        )
        if st.button("Select"):
            utilss.update_project(sel_prj)
            placeholder.markdown(f"##### 📃 `{st.session_state.prj_name}`", width='content')

    if sel_opt == 'Switch to existing project':
        list_projects = get_subfolders(st.session_state.paths['out_dir'])
        if len(list_projects) > 0:
            sel_ind = list_projects.index(st.session_state.prj_name)
            sel_prj = sac.chip(
                list_projects,
                label='', index=None, align='left', size='sm', radius='sm',
                multiple=False, color='cyan', description='Projects in output folder'
            )
            if st.button("Select"):
                utilss.update_project(sel_prj)
                placeholder.markdown(f"##### 📃 `{st.session_state.prj_name}`", width='content')
                if sel_prj is not None:
                    utilss.update_project(sel_prj)
                    placeholder.markdown(f"##### 📃 `{st.session_state.prj_name}`", width='content')
    
    if sel_opt == 'Reset project folder':
        st.warning("⚠️Are you sure you want to delete all files in the project folder? This cannot be undone.")
        flag_confirm = st.checkbox("I understand and want to delete all files in this folder")

        with st.container(horizontal=True, horizontal_alignment="center"):
            if st.button("Delete") and flag_confirm:
                clear_folder(st.session_state.paths['prj_dir'])
                st.toast(f"Files in project {st.session_state.prj_name} have been successfully deleted.")
                utilss.update_project(st.session_state.prj_name)

def panel_upload_single_subject():
    '''
    Upload user data to target folder
    '''
    sac.divider(key='_p2_div1')
    
    with st.container(horizontal=True, horizontal_alignment="left"):
        st.markdown("##### Upload File(s): ", width='content')
        with st.popover("❓", width='content'):
            st.write(
                """
                **Data Upload Help**
                  - Upload MRI scans in **NIfTI** (.nii / .nii.gz) or **DICOM** (either a folder of .dcm files or a single .zip archive).
                  - A **subject list** will be created automatically as MRI scans are added
                  - You may also upload non-imaging data (e.g., clinical variables) as a **CSV** containing an **MRID** column that matches the subject list.                
                """
            )
            
    #placeholder = st.empty()
    #placeholder.markdown(f"##### 📃 `{st.session_state.prj_name}`", width='content')
    
    with st.form(key='my_form', clear_on_submit=True, border=False):

        out_tmp = os.path.join(st.session_state.paths['prj_dir'], 'tmp_upload')
            
        sel_files = st.file_uploader(
            "Input files or folders",
            key="_uploaded_input",
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        submitted = st.form_submit_button("Submit")
        if not submitted:
            return
        
        copy_uploaded_files(sel_files, out_tmp)
        return True
    
    return False

def panel_view_files():
    '''
    Show files in data folder
    '''
    sac.divider(key='_p3_div1')
    
    with st.container(horizontal=True, horizontal_alignment="left"):
        st.markdown("##### Review File(s): ", width='content')
        with st.popover("❓", width='content'):
            st.write(
                """
                **Review Files Help**
                  - View files stored in the project folder.
                """
            )
            
    placeholder = st.empty()
    placeholder.markdown(f"##### 📁 `{st.session_state.prj_name}`", width='content')
    
    tree_items, list_paths = utildv.build_folder_tree(
        st.session_state.paths['prj_dir'], st.session_state.out_dirs
    )
    selected = sac.tree(
        items=tree_items,
        #label='Project Folder',
        index=None,
        align='left', size='xl', icon='table',
        checkbox=False,
        #checkbox_strict = True,
        open_all = True,
        return_index = True
        #height=400
    )
    
    if selected:
        if isinstance(selected, list):
            selected = selected[0]
        fname = list_paths[selected]
        show_sel_item(fname)
    


