import streamlit as st
import utils.utils_dicoms as utildcm
import utils.utils_session as utilss
import os
import pandas as pd
import zipfile
import streamlit_antd_components as sac
import shutil
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
            if not os.path.exists(f_out):
                with open(os.path.join(d_out, in_file.name), "wb") as f:
                    f.write(in_file.getbuffer())
    # Unzip zip files
    print("Extracting zip files")
    if os.path.exists(d_out):
        unzip_zip_files(d_out)

def callback_copy_uploaded():
    '''
    Copies files to local storage
    '''
    if len(st.session_state['_uploaded_input']) > 0:
        copy_and_unzip_uploaded_files(
            st.session_state['_uploaded_input'], st.session_state.paths["target"]
        )

def upload_multiple_files(out_dir):
    '''
    Upload user data to target folder
    Input data may be a folder, multiple files, or a zip file (unzip the zip file if so)
    '''
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Set target path
    st.session_state.paths["target"] = out_dir

    # Upload data
    with st.container(border=True):
        st.file_uploader(
            "Input files or folders",
            key="_uploaded_input",
            accept_multiple_files=True,
            on_change = callback_copy_uploaded,
            help="Input files can be uploaded as a folder, multiple files, or a single zip file",
        )

def upload_multi(out_dir):
    """
    Panel for uploading multiple input files or folder(s)
    """
    # Check if data exists
    if st.session_state.app_type == "cloud":
        # Upload data
        upload_folder(
            out_dir,
            "Input files or folders",
            False,
            "Input files can be uploaded as a folder, multiple files, or a single zip file",
        )

    else:  # st.session_state.app_type == 'desktop'
        if not os.path.exists(out_dir):
            try:
                os.symlink(sel_dir, out_dir)
            except:
                st.error(
                    f"Could not link user input to destination folder: {out_dir}"
                )

    # Check out files
    fcount = get_file_count(st.session_state.paths[dtype])
    if fcount > 0:
        st.session_state.flags[dtype] = True
        p_dicom = st.session_state.paths[dtype]
        st.success(
            f" Uploaded data: ({p_dicom}, {fcount} files)",
            icon=":material/thumb_up:",
        )
        time.sleep(4)

        st.rerun()

def upload_single_file(out_dir, out_name, label) -> None:
    '''
    Upload user file to target folder
    '''
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_file = os.path.join(out_dir, out_name)

    # Set target path
    st.session_state.paths["target"] = out_dir

    # Upload data
    with st.container(border=True):
        sel_file = st.file_uploader(
            label,
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


def create_img_list(dtype: str) -> None:
    '''
    Create a list of input images
    '''
    out_dir = os.path.join(
        st.session_state.paths['project'], dtype
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

##############################################################
## Panels for IO

def load_dicoms():
    tab = sac.tabs(
        items=[
            sac.TabsItem(label='Upload'),
            sac.TabsItem(label='Detect Series'),
            sac.TabsItem(label='Extract Scans'),
            sac.TabsItem(label='View'),
            sac.TabsItem(label='Reset'),
        ],
        size='lg',
        align='left'
    )

    dicom_dir = os.path.join(
        st.session_state.paths['project'], 'dicoms'
    )
    
    if tab == "Upload":
        upload_multiple_files(dicom_dir)

        fcount = get_file_count(dicom_dir)
        if fcount > 0:
            st.success(
                f"Data is ready ({dicom_dir}, {fcount} files)",
                icon=":material/thumb_up:",
            )
        
    elif tab == "Detect Series":
        utildcm.panel_detect_dicom_series(
            dicom_dir
        )
        
    elif tab == "Extract Scans":
        utildcm.panel_extract_nifti(
            st.session_state.paths['project']
        )
        
    elif tab == "View":
        st.info('not there yet')
        # panel_view('T1')

    elif tab == "Reset":
        st.info(f'Out folder name: {out_dir}')
        if st.button("Delete"):
            remove_dir(out_dir)

def load_nifti():
    '''
    Panel to load nifti images
    '''
    tab = sac.tabs(
        items=[
            sac.TabsItem(label='Upload'),
            sac.TabsItem(label='View'),
            sac.TabsItem(label='Reset'),
        ],
        size='lg',
        align='left'
    )

    sel_mod = sac.segmented(
        items=st.session_state.list_mods,
        size='sm',
        align='left'
    )

    if sel_mod is None:
        return

    out_dir = os.path.join(
        st.session_state.paths['project'], sel_mod.lower()
    )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    if tab == 'Upload':
        if st.button("Upload"):
            # Upload data
            upload_multiple_files(out_dir)
        
        ## Create list of scans
        #df = create_img_list(sel_mod.lower())
        #if df is not None:
            #out_file = os.path.join(
                #lists_path, 'list_nifti.csv'
            #)
            #df.to_csv(out_file, index=False)
            
    elif tab == 'View':
        st.info('Not there yet')

    elif tab == 'Reset':
        st.info(f'Out folder name: {out_dir}')
        if st.button("Delete"):
            remove_dir(out_dir)
    
    fcount = get_file_count(out_dir, ['.nii', '.nii.gz'])
    if fcount > 0:
        st.success(
            f" Input data available: ({fcount} nifti image files)",
            icon=":material/thumb_up:",
        )

def load_csv():
    '''
    Panel for uploading covariates
    '''    
    tab = sac.tabs(
        items=[
            sac.TabsItem(label='Upload'),
            sac.TabsItem(label='Enter Manually'),
            sac.TabsItem(label='View'),
            sac.TabsItem(label='Reset'),
        ],
        size='lg',
        align='left'
    )

    out_dir = os.path.join(st.session_state.paths['project'], 'demog')
    fname = 'demog.csv'
    out_csv = os.path.join(out_dir, fname)
    
    if tab == 'Upload':
        upload_single_file(out_dir, fname, 'Select demog file')

    elif tab == 'Enter Manually':
        try:
            df = pd.read_csv(out_csv)
            df = df[['MRID']]
            
        except:
            st.warning('Could not read id list')
            return

        df['Age'] = pd.Series([np.nan] * len(df), dtype='float')
        df['Sex'] = pd.Series([''] * len(df), dtype='string')
            
        st.info("Please enter values for your sample")
        
        # Define column options
        column_config = {
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
            num_rows="dynamic",  # allows adding rows if you want
            use_container_width=True
        )
        if st.button('Save'):
            df_user.to_csv(out_csv, index=False)
            st.success(f'Updated demographic file: {out_csv}')
        

    elif tab == "View":
        if not os.path.exists(out_csv):
            st.warning('Covariate file not found!')
            return
        try:
            df_cov = pd.read_csv(out_csv)
            st.dataframe(df_cov)
        except:
            st.warning(f'Could not load covariate file: {out_csv}')
        
    elif tab == "Reset":
        if st.button('Delete demog file'):
            remove_dir(out_dir)

##############################################################
## Streamlit panels for IO

def panel_select_project(out_dir, curr_project):
    '''
    Panel for creating/selecting a project name/folder (to keep all data for the current project)
    '''
    items = ['Create New', 'Select Existing']
    if st.session_state.has_cloud_session:
        items.append('Generate Demo Data')
    sel_mode = sac.tabs(
        items=items,
        size='lg',
        align='left'
    )
    
    if sel_mode is None:
        return None

    if sel_mode == 'Generate Demo Data':  
        # Copy demo dirs to user folder (TODO: make this less hardcoded)
        demo_dir_paths = [
            os.path.join(
                st.session_state.paths["root"],
                "output_folder",
                "NiChart_sMRI_Demo1",
            ),
            os.path.join(
                st.session_state.paths["root"],
                "output_folder",
                "NiChart_sMRI_Demo2",
            ),
        ]
        demo_names = []
        for demo in demo_dir_paths:
            demo_name = os.path.basename(demo)
            demo_names.append(demo_name)
            destination_path = os.path.join(
                st.session_state.paths["out_dir"], demo_name
            )
            if os.path.exists(destination_path):
                shutil.rmtree(destination_path)
            shutil.copytree(demo, destination_path, dirs_exist_ok=True)
        st.success(f"NiChart demonstration projects have been added to your projects list: {', '.join(demo_names)} ")
        return
      
    if sel_mode == 'Create New':
        sel_project = st.text_input(
            "Task name:",
            None,
            placeholder="My_new_study",
            label_visibility = 'collapsed'
        )   
    if sel_mode == 'Select Existing':
        list_projects = get_subfolders(out_dir)
        if len(list_projects) > 0:
            sel_ind = list_projects.index(curr_project)
            sel_project = st.selectbox(
                "Select Existing Project",
                options = list_projects,
                index = sel_ind,
                label_visibility = 'collapsed',
            )
    if sel_project is None:
        return
    
    
    if st.button("Select"):
        if sel_project != curr_project:
            utilss.update_project(sel_project)
        return sel_project

def panel_load_data():
    '''
    Panel for loading user data
    '''
    sel_dtype = sac.tabs(
        items=[
            sac.TabsItem(label='Nifti'),
            sac.TabsItem(label='Dicom'),
            sac.TabsItem(label='Lists')
        ],
        size='lg',
        align='left'
    )

    if sel_dtype is None:
        return

    if sel_dtype == "Nifti":
        with st.container(border=True):
            st.markdown(
                """
                ***NIfTI Images***
                - Upload NIfTI images
                """
            )
            load_nifti()

    elif sel_dtype == "Dicom":
        with st.container(border=True):
            st.markdown(
                """
                ***DICOM Files***
                
                - Upload a folder containing raw DICOM files
                - DICOM files will be converted to NIfTI scans
                """
            )
            load_dicoms()
            
    elif sel_dtype == "Lists":
        with st.container(border=True):
            st.markdown(
                """
                ***Covariate File***
                - Upload a ***:red[csv file with covariate info]*** (Age, Sex, DX, etc.)
                """
            )
            load_csv()
