import streamlit as st
import utils.utils_dicoms as utildcm
import utils.utils_session as utilss
import utils.utils_data_view as utildv
import utils.utils_toolloader as utiltl
import utils.utils_csvparsing as utilcsv
import os
import pandas as pd
import numpy as np
import zipfile
import re
import streamlit_antd_components as sac
import shutil
import time
from typing import Any, BinaryIO, List, Optional
from dataclasses import dataclass, asdict

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
    with st.container(border=True):
        sel_file = st.file_uploader(
            label,
            key="uploaded_input_csv",
            accept_multiple_files=False
        )        
        if sel_file is not None:
            try:
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                out_file = os.path.join(out_dir, out_name)
                
                with open(out_file, "wb") as f:
                    f.write(sel_file.getbuffer())
                st.success(f"File '{sel_file.name}' saved to {out_file}")
                return True
            except:
                st.warning(f'Could not upload file: {sel_file}')
                return False
        return False


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

def create_scan_csv() -> None:
    '''
    Create a csv with MRID (and other required fields if available)
    '''
    out_dir = os.path.join(
        st.session_state.paths['project'], 'lists'
    )

    # Detect common suffix
    def detect_common_suffix(files):
        reversed_names = [f[::-1] for f in files]
        common_suffix = os.path.commonprefix(reversed_names)[::-1]
        return common_suffix
    
    # Remove common suffix to get mrid
    def remove_common_suffix(files):
        reversed_names = [f[::-1] for f in files]
        common_suffix = os.path.commonprefix(reversed_names)[::-1]
        return [f[:-len(common_suffix)] if common_suffix else f for f in files]

    # Get all NIfTI files
    dfs = []
    for mod in ['t1', 't2', 'fl', 'dti', 'fmri']:
        img_dir = os.path.join(
            st.session_state.paths['project'], mod
        )
        if os.path.exists(img_dir):
            nifti_files = [
                f for f in os.listdir(img_dir) if f.endswith('.nii') or f.endswith('.nii.gz')
            ]
            suff = detect_common_suffix(nifti_files)
            for fname in nifti_files:


                # Read info from csv file if exists
                fcsv = os.path.join(
                    img_dir, f'{fname.replace('.nii.gz','').replace('.nii','')}.csv'
                )
                if os.path.exists(fcsv):
                    df = pd.read_csv(fcsv)

                else:
                    # Detect mrid from file name otherwise
                    mrid = fname.replace(suff,'')
                    df = pd.DataFrame({'MRID': [mrid], 'Age': [None], 'Sex': [None]})
                
                dfs.append(df)
    if len(dfs) == 0:
        return None
    
    df = pd.concat(dfs, axis=0).sort_values(by='MRID')
    df = df.drop_duplicates().reset_index().drop('index', axis=1)
    
    # Add columns for batch and dx
    df[['Batch']] = f'{st.session_state.project}_Batch1'
    df[['IsCN']] = 1
    
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

    out_dir = os.path.join(
        st.session_state.paths['project'], 'dicoms'
    )
    
    if tab == "Upload":
        if st.button("Upload"):
            # Upload data
            upload_multiple_files(out_dir)

        fcount = get_file_count(out_dir)
        if fcount > 0:
            st.success(f'Dicom data available ({fcount} files)')
        
    elif tab == "Detect Series":
        utildcm.panel_detect_dicom_series(out_dir)
        
    elif tab == "Extract Scans":
        utildcm.panel_extract_nifti(st.session_state.paths['project'])
        
    elif tab == "View":
        # Create list of scans
        sel_mod='T1'
        df = create_img_list(sel_mod.lower())
        st.dataframe(df)

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
        
        fcount = get_file_count(out_dir, ['.nii', '.nii.gz'])
        if fcount > 0:
            st.success(
                f" Detected {fcount} nifti image files", icon=":material/thumb_up:"
            )
        else:
            st.info(
                f" No nifti image files", icon=":material/thumb_down:"
            )
            
    elif tab == 'View':
        # Create list of scans
        df = create_img_list(sel_mod.lower())
        st.dataframe(df)        

    elif tab == 'Reset':
        st.info(f'Out folder name: {out_dir}')
        if st.button("Delete"):
            remove_dir(out_dir)
    
def load_subj_list():
    '''
    Panel for uploading subject list with variables required for processing
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

    out_dir = os.path.join(st.session_state.paths['project'], 'participants')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    fname = 'participants.csv'
    out_csv = os.path.join(out_dir, fname)
    
    if tab == 'Upload':
        upload_single_file(out_dir, fname, 'Select participants file')

    elif tab == 'Enter Manually':
        df = create_scan_csv()
            
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


def load_user_csv():
    '''
    Panel for uploading data file
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

    out_dir = os.path.join(st.session_state.paths['project'], 'user_data')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fname = 'user_data.csv'
    out_csv = os.path.join(out_dir, fname)
            
    if tab == 'Upload':
        # Upload file
        if upload_single_file(out_dir, fname, 'Select data file'):
            # Update variable dictionary
            df_user = pd.read_csv(out_csv)
            df_dict = st.session_state.dicts['df_var_groups']
            if 'user_data' not in df_dict.group.tolist():
                df_dict.loc[len(df_dict)] = {
                    'group': 'user_data',
                    'category': 'user',
                    'vtype': 'name',
                    'atlas': None,
                    'values': df_user.columns.sort_values().tolist()
                }
                st.session_state.dicts['df_var_groups'] = df_dict

    elif tab == "View":
        if not os.path.exists(out_csv):
            st.warning('Data file not found!')
            return
        try:
            df_data = pd.read_csv(out_csv)
            st.dataframe(df_data)
        except:
            st.warning(f'Could not load data file: {out_csv}')
        
    elif tab == "Reset":
        if st.button('Delete data file'):
            remove_dir(out_dir)


##############################################################
## Streamlit panels for IO

def panel_import_demo_data():
    st.info("You can import some demonstration data into your projects list by clicking the button below.")
    if st.button("Import"):
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

def get_path_for_project(project):
    return os.path.join(st.session_state.paths['out_dir'], project)

def preview_project_folder(project):
    """
    Panel for viewing files in a project folder
    """
    with st.container(border=True):
        in_dir = get_path_for_project(project)
        utildv.data_overview(in_dir)

def panel_select_existing_with_preview(out_dir):
    left, right = st.columns([1, 2], gap='large')
    
    list_projects = get_subfolders(out_dir)
    curr_project = st.session_state.project
    sel_project = curr_project
    with left:
        st.markdown("### Select Project")
        if len(list_projects) > 0:
            sel_ind = list_projects.index(curr_project)
            sel_project = st.selectbox(
                "Select Existing Project",
                options = list_projects,
                index = sel_ind,
                label_visibility = 'collapsed',
            )
    with right:
        st.markdown("### Preview Project Data")
        preview_project_folder(sel_project)

    if sel_project is None:
        return
    else:
        utilss.update_project(sel_project)
        st.success(f"Selected project {sel_project}")

def validate_project_name(string):
    """
    Return True if `name` is safe for filenames/directories.
    Allows only alphanumerics and underscores.
    Rejects spaces, slashes, dots, colons, brackets, etc.
    """
    # Must contain only letters, digits, or underscores
    return bool(re.fullmatch(r'[A-Za-z0-9_]+', string))

def panel_create_new():
    with st.container(border=True):
        st.info("Write a new project name and hit enter to save.")
        sel_project = st.text_input(
                "Type a project name and hit enter:",
                None,
                placeholder="",
                label_visibility = 'collapsed'
            )
        if sel_project is None:
            return
        project_name_ok = validate_project_name(sel_project)
        if not project_name_ok:
            st.error(f"Project names can only contain alphanumeric characters and underscores. Please revise.")
        else:
            utilss.update_project(sel_project)
            st.success(f"Created project {sel_project}.")

def panel_select_project(out_dir, curr_project):
    '''
    Panel for creating/selecting a project name/folder (to keep all data for the current project)
    '''
    items = ['Select Existing', 'Create New']
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
        st.info("You can import some demonstration data into your projects list by clicking the button below.")
        if st.button("Generate"):
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

@dataclass
class RequirementStatus:
    name: str
    status: str # 'green' | 'yellow' | 'red'
    count: int # how many items available/satsified
    target: int # reference count for 'green'
    note: str = '' # short human message

def _csv_severity(report) -> str:
    if not report.file_ok or not report.columns_ok:
        return "red"
    return "yellow" if report.issues else "green"

def _issues_dataframe(issues) -> pd.DataFrame:
    """Makes a nice table for streamlit from List[CSVIssue]"""
    if not issues:
        return pd.DataFrame(columns=["mrid", "row", "column", "value", "reason"])
    df = pd.DataFrame([asdict(i) for i in issues])
    cols = [c for c in ["mrid", "row", "column", "value", "reason"] if c in df.columns]
    return df[cols]


def count_csv_rows(csv_path: str) -> int:
    try:
        # Load the CSV safely — low_memory=False avoids dtype guessing issues on large files
        df = pd.read_csv(csv_path, low_memory=False)
        # Count rows (header is automatically excluded by pandas)
        return len(df)
    except FileNotFoundError:
        print(f"Error: File '{csv_path}' not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: File '{csv_path}' is empty.")
    except pd.errors.ParserError as e:
        print(f"Error parsing '{csv_path}': {e}")
    return 0

def compute_counts(ctx: dict = {}) -> dict:
    """
    ctx can contain other contextual info, use as needed to pass things from ui
    """
    sel_project = st.session_state.project
    project_path = get_path_for_project(sel_project)
    t1_path = os.path.join(project_path, "t1")
    flair_path = os.path.join(project_path, "fl")
    demog_csv_path = os.path.join(project_path, "participants", "participants.csv")

    t1_count = get_file_count(t1_path, ['.nii', '.nii.gz'])
    flair_count = get_file_count(flair_path, ['.nii', '.nii.gz'])
    csv_rows = count_csv_rows(demog_csv_path)
    res = {
        "needs_T1": t1_count,
        "needs_FLAIR": flair_count,
        "needs_demographics": csv_rows,
    }
    return res

def classify_cardinality(req_order, counts: dict):
    """
    req_order: list[(name, params)]
    counts: dict name-> int
    Returns: list[RequirementStatus]
    Rule:
        target = max(counts of all present requirements among T1/FLAIR images, MRID rows)
        red = count == 0 for a required item
        yellow = 0 < count < target
        green = count >= target
    """
    present_names = [name for (name, _) in req_order if name in counts]
    target = max([counts[n] for n in present_names], default=0)
    out = []
    for (name, _) in req_order:
        if name not in counts:
            # Non-cardinality requirements (e.g. csv_has_columns -> treat as bool)
            # Skip otherwise
            continue
        c = counts[name]
        if c == 0:
            status, note = "red", "missing"
        elif c < target:
            status, note = "yellow", f"{c}/{target} available"
        else:
            status, note = "green", f"{c}/{target} available"

        out.append(RequirementStatus(name=name, status=status, count=c, target=target, note=note))
    return out

def panel_guided_upload_data():
    # That's right, emojis in the code. >:^)
    STATUS_ICON = {"green": "✅", "yellow": "⚠️", "red": "❌"}
    REQ_TO_HUMAN_READABLE = {
        'needs_T1': 'T1 Scans Required',
        'needs_FLAIR': 'FLAIR Scans Required',
        'needs_demographics': 'Demographic CSV Required', 
    }
    pipeline = st.session_state.sel_pipeline
    pipeline_selected_explicitly = st.session_state.pipeline_selected_explicitly
    if not pipeline_selected_explicitly:
        st.info("No pipeline was selected, so we auto-selected DLMUSE.")
    else:
        st.info(f"Pipeline {pipeline} was selected, so we'll guide you through the required inputs.")

    pipeline_id = utiltl.get_pipeline_id_by_name(pipeline)
    reqs_set, reqs_params, req_order = utiltl.parse_pipeline_requirements(pipeline_id)

    if "needs_demographics" in reqs_set:
        required_cols = reqs_params.get("csv_has_columns", [])
        csv_path = os.path.join(st.session_state.paths["project"], 'participants.csv')
        csv_report = utilcsv.validate_csv(csv_path=csv_path, required_cols=required_cols, mrid_col="MRID")
        severity = _csv_severity(csv_report)
        icon = STATUS_ICON[severity]

        # Build a concise label
        if not csv_report.file_ok:
            note = "CSV file not found."
        elif not csv_report.columns_ok:
            note = f"Missing columns: {', '.join(csv_report.missing_cols)}"
        elif csv_report.issues:
            note = f"{len(csv_report)} issue(s) detected"
        else:
            note = "All required columns found and passed validation; no issues"

        csv_expanded = (severity != "green")
        with st.expander(f"{icon} Demographics CSV - {note}", expanded=csv_expanded):
            if csv_report.file_ok:
                if csv_report.missing_cols:
                    st.error("Missing: " + ", ".join(csv_report.missing_cols))
                if csv_report.present_cols:
                    st.success("Present: " + ", ".join(csv_report.present_cols))
                if csv_report.extra_cols:
                    st.info("Extra (not used): " + ", ".join(csv_report.extra_cols))
                st.caption(f"Rows in CSV: {csv_report.rows}")

                if csv_report.issues:
                    st.subheader("Issues")
                    issues_df = _issues_dataframe(csv_report.issues)
                    group_by_col = st.selectbox(
                        "Group issues by", ["(none)", "column", "reason"],
                        index=1 if "column" in issues_df.columns else 0,      
                    )
                    if group_by_col != "(none)" and group_by_col in issues_df.columns:
                        for key, sub in issues_df.groupby(group_by_col):
                            st.markdown(f"**{group_by_col}: {key}** - {len(sub)} row(s)")
                            st.dataframe(sub, use_container_width=True, height=220)
                    else:
                        st.dataframe(issues_df, use_container_width=True, height=320)
                    
                    st.info("Tip: fix the data and reupload or hit save to refresh validation")
            else:
                st.warning("Upload or enter a demographics CSV to validate.")
    
    # need to generate counts
    counts = compute_counts()
    items = classify_cardinality(req_order, counts)

    for item in items:
        icon = STATUS_ICON[item.status]
        expanded = (item.status != "green")
        
        label = f"{icon} {REQ_TO_HUMAN_READABLE[item.name]} - {item.note}"
        with st.expander(label, expanded=expanded):
            if item.name == "needs_T1":
                st.write("Please upload T1 images.")
                st.write("PLACEHOLDER, WIDGET GOES HERE")
            elif item.name == "needs_FLAIR":
                st.write("Please upload FLAIR images.")
                st.write("PLACEHOLDER, WIDGET GOES HERE")
            elif item.name == "needs_demographics":
                req_cols = reqs_params.get('csv_has_columns', [])
                st.write("Please provide demographics CSV.", "Required columns:", ", ".join(req_cols) or "(none)")
                st.write("PLACEHOLDER, WIDGET GOES HERE")
            elif item.name == "csv_has_columns":
                pass # Handled in needs_demographics case
            else:
                raise ValueError(f"Requirement {item.name} for pipeline {pipeline_id} has no associated rule. Please submit a bug report.")

    ready = True
    if any(s.status == "red" for s in items):
        ready = False
    
    if "needs_demographics" in reqs_set:
        ready = ready and (csv_report is not None) and _csv_severity(csv_report) == "green"
    
    st.markdown("---")
    if ready:
        st.success("All requirements satisfied. You can proceed.")
        st.button("Continue", type="primary")
    else:
        st.info("Resolve the issues above to proceed.")
    pass

def panel_guided_nifti_upload():
    pass

def panel_guided_demographics_upload():
    pass

def panel_guided_upload_additionaldata():
    pass

def panel_load_data(default=None, default_nifti_type=None):
    '''
    Panel for loading user data
    '''
    sel_dtype = sac.tabs(
        items=[
            sac.TabsItem(label='Nifti'),
            sac.TabsItem(label='Dicom'),
            sac.TabsItem(label='Subject List'),
            sac.TabsItem(label='Additional Data')            
        ],
        size='lg',
        align='left',
        
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
            
    elif sel_dtype == "Subject List":
        with st.container(border=True):
            st.markdown(
                """
                ***Subject List***
                - List file with columns required for running pipelines
                - Required fields: MRID, Age, Sex, Batch
                """
            )
            load_subj_list()

    elif sel_dtype == "Additional Data":
        with st.container(border=True):
            st.markdown(
                """
                ***Additional data files***
                - Example: Clinical data
                """
            )
            load_user_csv()
