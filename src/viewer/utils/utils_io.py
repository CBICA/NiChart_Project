import streamlit as st
import utils.utils_dicoms as utildcm
import utils.utils_session as utilss
import utils.utils_data_view as utildv
import utils.utils_toolloader as utiltl
import utils.utils_csvparsing as utilcsv
import os
import pandas as pd
import numpy as np
from NiChart_common_utils.nifti_parser import NiftiMRIDParser
import zipfile
import re
import streamlit_antd_components as sac
import shutil
import time
from typing import Any, BinaryIO, List, Optional, Dict, Tuple
from dataclasses import dataclass, asdict

import difflib


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

def clear_folder(folder):
    if os.path.islink(folder):
        st.warning("Target folder is a symlink. Cannot delete contents")
        return        

    try:
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        logger.debug(f"Removed dir: {folder}")
        time.sleep(2)
        return True
    except:
        st.error(f"Could not delete folder: {folder}")
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

def unzip_zip_file(f_in, d_out):
    '''
    Unzips a ZIP file to a new folder and deletes the zip file
    '''
    os.makedirs(d_out, exist_ok=True)
    with zipfile.ZipFile(f_in, "r") as zip_ref:
        zip_ref.extractall(d_out)
    os.remove(f_in)
    st.toast(f'Unzipped zip file ...')

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

    with st.form(key='my_form', clear_on_submit=True, border=False):

        sel_mod = sac.chip(
            items=[
                sac.ChipItem(label='T1'),
                sac.ChipItem(label='FL'),
                sac.ChipItem(label='CSV'),
            ], label='', index=0, align='left', size='md', radius='md', multiple=False, color='cyan', 
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
            label_visibility="collapsed"
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
    #with st.container(border=True):
    st.markdown('##### User Input:')
    #st.markdown(
        #'''
        #- Upload your input data file(s) here
        #- MRI scan (Nifti or Dicom) or a data file (.csv)
        #'''
    #)
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
        return False


def create_img_list(dtype: str, show_warning=False) -> None:
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
    if not nifti_files and show_warning:
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
    mod_dirs = {mod: os.path.join(st.session_state.paths['project'], mod) for mod in ['t1', 't2', 'fl', 'dti', 'fmri']}
    dir_dict = {'T1': mod_dirs['t1'],
                            'T2': mod_dirs['t2'],
                            'FLAIR': mod_dirs['fl'],
                            'DTI': mod_dirs['dti'],
                            'FMRI': mod_dirs['fmri'],
                            }
    nifti_parser = NiftiMRIDParser()
    heuristic_df = nifti_parser.create_master_csv(dir_dict, os.path.join(st.session_state.paths['project'], 'inferred_data_paths.csv'))
    
    
    df = heuristic_df.sort_values(by='MRID')
    df = df.drop_duplicates().reset_index().drop('index', axis=1)
    
    # Add columns for batch and dx
    df[['Batch']] = f'{st.session_state.project}_Batch1'
    df[['IsCN']] = 1
    
    return df


def normalize_demographics_df(
    df_raw: pd.DataFrame,
    mrid_reference_df: pd.DataFrame,
    *,
    required_cols=("MRID", "Age", "Sex"),
    mrid_col_in_ref="MRID",
    age_range=(0, 120),
    mrid_similarity_threshold=0.85
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Normalize a demographics DataFrame to have canonical columns (MRID, Age, Sex),
    coerce types/values, and map MRIDs to a reference set using fuzzy matching.

    Parameters
    ----------
    df_raw : pd.DataFrame
        DataFrame loaded from user CSV.
    mrid_reference_df : pd.DataFrame
        DataFrame that contains canonical MRIDs in column `mrid_col_in_ref`.
    required_cols : tuple
        Canonical columns required in the output, in desired order.
    mrid_col_in_ref : str
        Column name in `mrid_reference_df` containing canonical MRIDs.
    age_range : tuple
        Acceptable (min_age, max_age) inclusive range for Age.
    mrid_similarity_threshold : float
        Threshold in [0,1] for fuzzy match acceptance (difflib ratio).

    Returns
    -------
    corrected_df : pd.DataFrame
        DataFrame with columns MRID (string), Age (Int64), Sex ('M'/'F').
    errors : list of dict
        Each error dict includes: {'row', 'column', 'value', 'reason'}.
    """

    errors: List[Dict[str, Any]] = []

    # ---- 0) Copy and normalize column names (strip + TitleCase to match 'MRID','Age','Sex') ----
    df = df_raw.copy()
    normalized_cols = {c: c.strip() for c in df.columns}
    df.rename(columns=normalized_cols, inplace=True)

    def _title_no_space(name: str) -> str:
        # Keep MRID uppercase special-case; otherwise Title Case
        s = name.strip()
        if s.lower().replace(" ", "") == "mrid":
            return "MRID"
        return s.title()

    target_map = {}
    for c in df.columns:
        new = _title_no_space(c)
        if new != c:
            # avoid collisions by disambiguating
            if new in df.columns:
                # If collision, keep original; we'll handle via aliasing below
                continue
            target_map[c] = new
    if target_map:
        df.rename(columns=target_map, inplace=True)

    # ---- 1) Try to resolve aliases if the exact required columns are missing ----
    # Common aliases
    aliases = {
        "MRID": ["Subject", "SubjectID", "Subject_Id", "Id", "ID", "Mrid", "Mrn", "PatientId", "PatientID"],
        "Age": ["Years", "AgeYears", "age", "AGE"],
        "Sex": ["Gender", "SexAssigned", "sex", "SEX"],
    }

    for canonical in required_cols:
        if canonical not in df.columns:
            # find best alias present (direct or fuzzy)
            candidates = aliases.get(canonical, [])
            present = [c for c in df.columns if c in candidates]
            if not present:
                # Fuzzy on column names
                match = difflib.get_close_matches(canonical, df.columns, n=1, cutoff=0.7)
                if match:
                    present = [match[0]]
            if present:
                df.rename(columns={present[0]: canonical}, inplace=True)

    # ---- 2) Ensure required columns exist ----
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        for c in missing:
            errors.append({"row": None, "column": c, "value": None,
                           "reason": f"Missing required column '{c}'."})
        # Add empty columns so downstream steps still run
        for c in missing:
            df[c] = pd.NA

    # Reorder to canonical order
    df = df[list(required_cols) + [c for c in df.columns if c not in required_cols]]

    # ---- 3) Normalize Sex to {'M','F'} ----
    def normalize_sex(x) -> Any:
        if pd.isna(x):
            return pd.NA
        s = str(x).strip().lower()
        m_vals = {"m", "male", "man", "1", "boy"}
        f_vals = {"f", "female", "woman", "0", "girl"}
        if s in m_vals:
            return "M"
        if s in f_vals:
            return "F"
        return pd.NA

    df["Sex"] = df["Sex"].apply(normalize_sex)

    for i, v in df["Sex"].items():
        if pd.isna(v):
            errors.append({"row": i, "column": "Sex", "value": df_raw.iloc[i][df_raw.columns.get_loc(df.columns[df.columns.get_loc('Sex')]) if 'Sex' in df.columns else 0] if 'Sex' in df.columns else None,
                           "reason": "Unrecognized or missing sex; expected one of M/F (accepted aliases: m/male/man/1, f/female/woman/0)."})

    # ---- 4) Normalize Age to pandas nullable Int64 within range ----
    age_min, age_max = age_range
    age_numeric = pd.to_numeric(df["Age"], errors="coerce")
    # Flag out-of-range or NaN
    bad_age_mask = age_numeric.isna() | (age_numeric < age_min) | (age_numeric > age_max) | (age_numeric % 1 != 0)
    for i, bad in bad_age_mask.items():
        if bad:
            errors.append({"row": i, "column": "Age", "value": df.loc[i, "Age"],
                           "reason": f"Invalid age (must be integer in [{age_min}, {age_max}])."})
    df["Age"] = age_numeric.round().astype("Int64")

    # ---- 5) Map MRIDs to nearest in reference using fuzzy matching when no exact match ----
    ref_mrids = mrid_reference_df[mrid_col_in_ref].astype(str).str.strip().unique().tolist()
    ref_set = set(ref_mrids)

    def best_mrid_match(mrid: str) -> Tuple[str, float]:
        # returns (best_match, similarity) using difflib ratio
        if not ref_mrids:
            return mrid, 0.0
        matches = difflib.get_close_matches(mrid, ref_mrids, n=1, cutoff=0)
        if not matches:
            return mrid, 0.0
        best = matches[0]
        sim = difflib.SequenceMatcher(None, mrid, best).ratio()
        return best, sim

    corrected_mrids = []
    for i, raw in df["MRID"].astype(str).str.strip().items():
        if raw in ref_set:
            corrected_mrids.append(raw)
            continue
        best, sim = best_mrid_match(raw)
        if sim >= mrid_similarity_threshold:
            corrected_mrids.append(best)
        else:
            corrected_mrids.append(raw)  # leave as-is but record error
            errors.append({"row": i, "column": "MRID", "value": raw,
                           "reason": f"MRID not found; no close match above threshold {mrid_similarity_threshold:.2f}."})

    df["MRID"] = corrected_mrids

    # ---- 6) Post-check: duplicates introduced by MRID mapping ----
    dup_mask = df["MRID"].duplicated(keep=False)
    if dup_mask.any():
        for i in df.index[dup_mask]:
            errors.append({"row": i, "column": "MRID", "value": df.loc[i, "MRID"],
                           "reason": "Duplicate MRID after mapping; manual disambiguation required."})

    # Final: return only canonical cols first for convenience
    ordered_cols = ["MRID", "Age", "Sex"] + [c for c in df.columns if c not in ("MRID", "Age", "Sex")]
    df = df[ordered_cols]

    return df, errors

##############################################################
## Panels for IO

def load_dicoms(default_modality='t1'):
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
    
    upload_multiple_files(out_dir)

    fcount = get_file_count(out_dir)
    if fcount > 0:
        st.success(f'Dicom data available ({fcount} files)')
        
    utildcm.panel_detect_dicom_series(out_dir)
        
    utildcm.panel_extract_nifti(st.session_state.paths['project'])
        
    # Create list of scans
    sel_mod='T1'
    df = create_img_list(sel_mod.lower())
    st.dataframe(df)

    if st.button("Delete"):
        remove_dir(out_dir)

def load_nifti(default_modality='t1', forced_modality=None):
    '''
    Panel to load nifti images
    '''
    if forced_modality is None:
        sel_mod = sac.segmented(
            items=st.session_state.list_mods,
            size='sm',
            align='left'
        )
    else:
        sel_mod = forced_modality

    if sel_mod is None:
        return

    out_dir = os.path.join(
        st.session_state.paths['project'], sel_mod.lower()
    )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    left, right = st.columns([2, 1])
    upload_multiple_files(out_dir)
    
    fcount = get_file_count(out_dir, ['.nii', '.nii.gz'])
    if fcount > 0:
        st.success(
            f" Detected {fcount} nifti image files", icon=":material/thumb_up:"
        )
    else:
        st.info(
            f" No nifti image files detected yet. Try uploading some!", icon=":material/thumb_down:"
        )  
    
    
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
        
    fname_tmp = 'participants_tmp.csv'
    fname = 'participants.csv'
    out_csv = os.path.join(out_dir, fname)
    
    if tab == 'Upload':
        upload_single_file(out_dir, fname_tmp, 'Select participants file')
        
        if os.path.exists(fname_tmp):
            st.success("We received your demographics file! Converting it to our format...")
            df_user = pd.read_csv(fname_tmp)
            mod_dirs = {mod: os.path.join(st.session_state.paths['project'], mod) for mod in ['t1', 't2', 'fl', 'dti', 'fmri']}
            dir_dict = {'T1': mod_dirs['t1'],
                            'T2': mod_dirs['t2'],
                            'FLAIR': mod_dirs['fl'],
                            'DTI': mod_dirs['dti'],
                            'FMRI': mod_dirs['fmri'],
                            }
            nifti_parser = NiftiMRIDParser()
            heuristic_df = nifti_parser.create_master_csv(dir_dict, os.path.join(st.session_state.paths['project'], 'inferred_data_paths.csv'))
            heuristic_df = heuristic_df.drop(df.filter(regex='_path$').columns, axis=1)
            corrected_df, issues = normalize_demographics_df(df_user, heuristic_df)
            corrected_df = corrected_df.sort_values(by='MRID')
            corrected_df = corrected_df.drop_duplicates().reset_index().drop('index', axis=1)
    
            # Add columns for batch and dx
            if 'Age' not in corrected_df.columns:
                corrected_df[['Age']] = '?'
            if 'Sex' not in corrected_df.columns:
                corrected_df[['Sex']] = '?'
            if 'Batch' not in corrected_df.columns:
                corrected_df[['Batch']] = f'{st.session_state.project}_Batch1'
            if 'IsCN' not in corrected_df.columns:
                corrected_df[['IsCN']] = 1
            corrected_df.to_csv(fname, index=False)
            st.success("Your CSV has been converted successfully.")
        

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
        if upload_files(out_dir, True):
            st.info('Hello')
            sel_mod=None
            st.rerun()


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

def panel_ask_harmonize():
    sel_method = st.session_state.sel_pipeline
    harmonizable = ['spare-ad', 'spare-ba', 'dlmuse', 'dlmuse-dlwmls', 'spare-smoking', 'spare-hypertension', 'spare-obesity', 'spare-diabetes']
    if sel_method in harmonizable:
        st.markdown("""
                    Do you want to harmonize your results to the reference data?
                    This requires at least 30 subjects and demographics (Age, Sex) data for each scan.
                    """)
        
        harmonize = st.checkbox("Harmonize to reference data? (Requires >= 30 scans)")
        st.session_state.do_harmonize = harmonize

def panel_guided_upload_data():
    # That's right, emojis in the code. >:^)
    STATUS_ICON = {"green": "✅", "yellow": "⚠️", "red": "❌"}
    REQ_TO_HUMAN_READABLE = {
        'needs_T1': 'T1 Scans',
        'needs_FLAIR': 'FLAIR Scans',
        'needs_demographics': 'Demographic CSV', 
    }
    pipeline = st.session_state.sel_pipeline
    pipeline_selected_explicitly = st.session_state.pipeline_selected_explicitly
    if not pipeline_selected_explicitly:
        st.info("No pipeline was selected, so we auto-selected DLMUSE.")
    else:
        st.info(f"Pipeline {pipeline} was selected, so we'll guide you through the required inputs.")

    pipeline_id = utiltl.get_pipeline_id_by_name(pipeline, harmonized=st.session_state.do_harmonize)
    reqs_set, reqs_params, req_order = utiltl.parse_pipeline_requirements(pipeline_id)

    # need to generate counts
    counts = compute_counts()
    
    items = classify_cardinality(req_order, counts)
    
    count_max_key = max(counts, key=counts.get)
    count_max_value = counts[count_max_key]
    count_diffs = {key: abs(counts[key]-count_max_value) for key in counts.keys() if key != count_max_key}

    for item in items:
        icon = STATUS_ICON[item.status]
        expanded = (item.status != "green")
        
        label = f"{icon} {REQ_TO_HUMAN_READABLE[item.name]} - {item.note}"
        with st.expander(label, expanded=expanded):
            if item.name == "needs_T1":
                st.write("Please upload T1 images.")
                panel_guided_nifti_upload(modality='T1')
            elif item.name == "needs_FLAIR":
                st.write("Please upload FLAIR images.")
                panel_guided_nifti_upload(modality='FLAIR')
            elif item.name == "needs_demographics":
                pass # Handled above 
            elif item.name == "csv_has_columns":
                pass # Handled in needs_demographics case
            else:
                raise ValueError(f"Requirement {item.name} for pipeline {pipeline_id} has no associated rule. Please submit a bug report.")
    if "needs_demographics" in reqs_set:
        required_cols = reqs_params.get("csv_has_columns", [])
        csv_path = os.path.join(st.session_state.paths["project"], 'participants' ,'participants.csv')
        csv_report = utilcsv.validate_csv(csv_path=csv_path, required_cols=required_cols, mrid_col="MRID")
        severity = _csv_severity(csv_report)
        icon = STATUS_ICON[severity]
        row_note = ""
        # Build a concise label
        if not csv_report.file_ok:
            note = "CSV file not found."
        elif not csv_report.columns_ok:
            note = f"Missing columns: {', '.join(csv_report.missing_cols)}"
        elif csv_report.issues:
            note = f"{len(csv_report)} issue(s) detected"
        else:
            note = "All required columns found and passed validation; no issues"
        if severity == "green":
            if count_max_key == "needs_demographics":
                for key, val in count_diffs:
                    if val < count_max_value:
                        row_note += f"{REQ_TO_HUMAN_READABLE[key]}: {val} MRIDs are in demographics CSV but not in available.\n"
                        severity = "yellow"
            else:
                for key, val in count_diffs:
                    if count_diffs["needs_demographics"] > val:
                        row_note += f"{REQ_TO_HUMAN_READABLE[key]}: {val} CSV entries are present which have no associated scan.\n"
                    elif count_diffs["needs_demographics"] < val:
                        row_note += f"{REQ_TO_HUMAN_READABLE[key]}: {val} scans are present which have no demographics CSV entry.\n"

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
                    panel_guided_demographics_upload()
            else:
                st.warning("Upload or enter a demographics CSV to validate.")
                panel_guided_demographics_upload()
    ready = True
    if any(s.status == "red" for s in items):
        ready = False
    
    if "needs_demographics" in reqs_set:
        ready = ready and (csv_report is not None) and _csv_severity(csv_report) == "green"
    if ready:
        st.success("All requirements satisfied. You can proceed.")
        st.button("Continue", type="primary")
    else:
        st.info("Resolve the issues above to proceed. Click to expand each requirement for more details.")
    pass

def panel_guided_nifti_upload(modality='t1'):
    left, right = st.columns(2)
    with left:
        do_nifti = st.button("Upload NIFTI files")
        if do_nifti:
            st.session_state.nifti_dicom_upload_mode = "nifti"
    with right:
        do_dicom = st.button("Upload and Convert DICOM")
        if do_dicom:
            st.session_state.nifti_dicom_upload_mode = "dicom"
    if st.session_state.nifti_dicom_upload_mode == "nifti":
        st.info("Drag and drop your NIFTI files to the gray box below, or browse for them using the button. Folders, .zip archives and image files are all accepted.")
        load_nifti(default_modality=modality, forced_modality=modality)
    elif st.session_state.nifti_dicom_upload_mode == "dicom":
        st.info("Follow these steps to convert your DICOM files.")
        load_dicoms()
    pass

def panel_guided_demographics_upload():
    load_subj_list()

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
