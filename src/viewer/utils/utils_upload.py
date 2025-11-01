import streamlit as st
import utils.utils_dicoms as utildcm
import utils.utils_session as utilss
import utils.utils_data_view as utildv
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

def panel_upload_single_subject(out_dir):
    '''
    Upload user data to target folder
    '''
    sac.divider(key='_p1_div1')

    st.markdown('##### Upload Files:')

    f_part = os.path.join(out_dir, 'participants', 'participants.csv')
        
    ###############################
    # Select data/file types
    mvals = [
        'T1', 'FL', 'Non-Imaging'
    ]
    fvals = [
        'Nifti (.nii.gz, .nii))', 'Compressed DICOM (.zip)',
        'Multiple DICOM Files', 'Tabular (.csv)'
    ]
    
    sel_mod = sac.chip(
        mvals, label='', key='_selmod', index=0, align='left', 
        size='md', radius='md', multiple=False, color='cyan', 
        description='Select data type'
    )    

    sel_index = 0
    if sel_mod == 'Non-Imaging':
        sel_index = len(fvals)-1
        
    sel_ftype = sac.chip(
        fvals, label='', key='_selftype', index=sel_index, align='left', 
        size='md', radius='md', multiple=False, color='cyan', 
        description='Select file format'
    )
    
    flag_multi = False
    if sel_ftype == 'Multiple DICOM Files':
        flag_multi = True
        
    ###############################
    # Upload data
    with st.form(key='my_form', clear_on_submit=True, border=False):

        out_path = os.path.join(out_dir, sel_mod.lower().replace(' ', '_'))
            
        sel_files = st.file_uploader(
            "Input files or folders",
            key="_uploaded_input",
            accept_multiple_files=flag_multi,
            label_visibility="collapsed"
        )
        
        submitted = st.form_submit_button("Submit")
        if not submitted:
            return
        
        if not sel_files:
            st.error('Please select files')
            return
        
        if sel_ftype == 'Nifti (.nii.gz, .nii))':
            fname = sel_files.name
            if not fname.endswith(('.nii', '.nii.gz', '.zip')):
                st.error('Selected file is not a Nifti image!')
                return
            copy_nifti(sel_files, out_path)

        elif sel_ftype == 'Tabular (.csv)':
            fname = sel_files.name
            if not fname.endswith(('.csv')):
                st.error('Selected file is not a .csv file!')
                return
            
        elif sel_ftype == 'Compressed DICOM (.zip)':
            fname = sel_files.name
            if not fname.endswith(('.csv')):
                st.error('Selected file is not a .zip file!')
                return

    ###############################
    # Create participants list
    if not os.path.exists(f_part):
        if fname.endswith('.nii.gz'):
            os.makedirs(os.path.dirname(f_part), exist_ok=True)
            mrid = fname.replace('.nii.gz', '').replace('_T1', '')
            df = pd.DataFrame({'MRID':[mrid], 'Age':[None], 'Sex':['']})
            df.to_csv(f_part, index=False)
            st.success(f'Created Participant file')
    
    
    return False

def panel_edit_participants(out_dir, fname):

    fpath = os.path.join(out_dir, fname)

    if not os.path.exists(fpath):
        return

    sac.divider(key='_p2_div1')
    with st.container(horizontal=True, horizontal_alignment="left"):
        st.markdown("##### Review & Edit Subject List: ", width='content')
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

        

def panel_view_folder(out_dir):
    '''
    Show files in data folder
    '''
    dname = os.path.basename(out_dir)
    sac.divider(key='_p2_div2')
    with st.container(horizontal=True, horizontal_alignment="left"):
        st.markdown("##### Review Project Folder: ", width='content')
        st.markdown(f"##### 📃 `{dname}`", width='content')

    if not os.path.exists(out_dir):
        return

    sel_opt = sac.chip(
        items=[
            sac.ChipItem(label='View'),
            sac.ChipItem(label='Change'),
            sac.ChipItem(label='Reset'),
        ],
        label='', index=0, align='left', 
        size='md', radius='md', multiple=False, color='cyan', 
        description='Select an action'
    )    

    if sel_opt == 'View':
        tree_items, list_paths = utildv.build_folder_tree(out_dir, st.session_state.out_dirs)
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
    
        #if selected:
            #if isinstance(selected, list):
                #selected = selected[0]
            #fname = list_paths[selected]
            #show_sel_item(fname)

