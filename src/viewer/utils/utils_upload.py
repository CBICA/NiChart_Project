import streamlit as st
import utils.utils_dicoms as utildcm
import utils.utils_session as utilss
import utils.utils_data_view as utildv
import utils.utils_misc as utilmisc
import utils.utils_io as utilio
import gui.utils_mriview as utilmri
import streamlit_antd_components as sac

import os
import time
import pandas as pd
import numpy as np
import zipfile
from NiChart_common_utils.nifti_parser import NiftiMRIDParser
import shutil
import time
from typing import Any, BinaryIO, List, Optional

from utils.utils_logger import setup_logger
logger = setup_logger()

def create_project_folder(sel_project) -> None:
    '''
    Creates a new project folder
    '''
    if sel_project is None:
        return False

    try:
        p_prj = os.path.join(st.session_state.paths['out_dir'], sel_project)
        os.makedirs(p_prj, exist_ok=True)
        p_tmp = os.path.join(p_prj, '_upload')
        os.makedirs(p_tmp, exist_ok=True)
        p_tmp = os.path.join(p_prj, 'plot_data')
        os.makedirs(p_tmp, exist_ok=True)
        p_tmp = os.path.join(p_prj, 'reports')
        os.makedirs(p_tmp, exist_ok=True)
        return True

    except:
        st.error(f'Could not create project folder: {p_prj}')
        return False
    
@st.dialog("Participant Information", width='medium')
def edit_participants(in_file):
    if not os.path.exists(in_file):
        return
    
    fname = os.path.basename(in_file)

    #sac.divider(key='_p2_div1')
    with st.container(horizontal=True, horizontal_alignment="left"):
        st.markdown("##### Edit Subject List: ", width='content')
        st.markdown(f"##### 📃 `{fname}`", width='content')

    df = pd.read_csv(in_file, dtype={'MRID':str, 'Age':float, 'Sex':str})
    
    # Define column options
    column_config = {
        "MRID": st.column_config.TextColumn(
            "MRID",
            disabled=True,
            required=True
        ),
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
        hide_index=True,
        column_config=column_config,
        num_rows="fixed",
        width='stretch'
    )
    
    with st.container(horizontal=True, horizontal_alignment="center"):
        if st.button('Save'):
            df_user.to_csv(in_file, index=False)
            utilmisc.show_temp_message(
                f'Updated participants file: {fname}', 'success', 2
            )
            st.rerun()

def update_participant_csv():
    mrid = st.session_state.participant['mrid']
    age = st.session_state.participant['age']
    sex = st.session_state.participant['sex']
    df = pd.DataFrame(
        {'MRID':[mrid], 'Age':[age], 'Sex':[sex]}
    )
    odir = os.path.join(st.session_state.paths['prj_dir'], 'participants')
    ofile = os.path.join(odir, 'participants.csv')
    os.makedirs(odir, exist_ok=True)
    df.to_csv(ofile, index=False)

@st.dialog("User csv", width='medium')
def consolidate_user_csv(fname):
    
    if fname is None:
        return False
    in_dir = os.path.join(st.session_state.paths['prj_dir'], '_upload')
    in_fpath = os.path.join(in_dir, fname)

    # Move csv to consolidated path
    out_dir = os.path.join(st.session_state.paths['prj_dir'], 'participants')
    out_fpath = os.path.join(out_dir, 'participants.csv')
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(out_fpath):
        st.warning('CSV file exists, will be overwritten!')
    shutil.move(in_fpath, out_fpath)        
        
    utilio.clear_folder(in_dir)

    utilmisc.show_temp_message(
        'CSV file uploaded ...', 'success', 2
    )
    st.rerun()   

def consolidate_nifti():
    '''
    Set meta data for uploaded image
    '''
    logger.debug(f'    Function: consolidate_nifti')
    
    # Get full name for the current file
    # Input image file is kept in a temporary upload folder
    in_fname = st.session_state.curr_scan
    if in_fname is None:
        return False
    
    in_dir = os.path.join(st.session_state.paths['prj_dir'], '_upload', 'nifti')
    in_fpath = os.path.join(in_dir, in_fname)

    logger.debug(f'      Input: {in_fpath}')

    # Get saved scan/participant info 
    mrid = st.session_state.participant['mrid']
    sex = st.session_state.participant['sex']
    if sex is None:
        ind_sex = None
    else:
        ind_sex = ['M', 'F', 'Other'].index(sex)
    age = st.session_state.participant['age']

    # Update values based on user iput
    with st.form(key='_form_scan_info'):
        mod = st.selectbox('Image Modality:', ['T1', 'FL'])
        mrid = st.text_input('MRID:', value = mrid)
        sex = st.selectbox('Sex (optional):', ['M', 'F', 'Other'], index=ind_sex)
        age = st.number_input('Age (optional):', min_value=20.0, max_value=110.0, value=age)
        
        submitted = st.form_submit_button("Submit")
        flag_submit = False
        if submitted:
            flag_submit = True
        
    if flag_submit:
        # Update participant info
        st.session_state.participant = {'mrid': mrid, 'age': age, 'sex': sex}
        update_participant_csv()

        # Move scan to consolidated path
        out_dir = os.path.join(st.session_state.paths['prj_dir'], mod.lower())
        out_fpath = os.path.join(out_dir, mrid + '_' + mod + '.nii.gz')
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(out_fpath):
            st.warning('Scan exists, will be updated!')
        shutil.move(in_fpath, out_fpath)
        utilio.clear_folder(in_dir)

        return True

    return False

@st.dialog("Scan/Participant Info", width='medium')
def dialog_consolidate_nifti():
    '''
    Dialog to save uploaded file as a single nifti image file
    '''
    logger.debug('    Function: dialog_consolidate_nifti')
    
    # Detect mrid
    mrid = st.session_state.participant['mrid']
    if mrid is None:
        mrid = st.session_state.curr_scan
        for suffix in ['.nii.gz', '.nii', '_T1', '_t1', '_FL', '_fl']:
            mrid = mrid.replace(suffix, '')
        st.session_state.participant['mrid'] = mrid
    
    # Consolidate scan
    stat_cons =  consolidate_nifti()

    if stat_cons:
        utilmisc.show_temp_message(
            f'Nifti file uploaded', 'success', 2
        )
        st.rerun()   

def detect_common_suffix(files):
    reversed_names = [f[::-1] for f in files]
    common_suffix = os.path.commonprefix(reversed_names)[::-1]
    return common_suffix

def consolidate_nifti_multi():
    logger.debug(f'    Function: consolidate_nifti')
    
    # Detect common suffix
    in_dir = os.path.join(st.session_state.paths['prj_dir'], '_upload', 'nifti')
    nifti_files = [
        f for f in os.listdir(in_dir) if f.endswith('.nii') or f.endswith('.nii.gz')
    ]
    if not nifti_files:
        st.warning("No NIfTI files found in the data folder.")
        return False

    # Remove common suffix to get mrid
    suff = detect_common_suffix(nifti_files)

    # Create the DataFrame
    df = pd.DataFrame({'MRID': nifti_files, 'Age': None, 'Sex': None})

    # Update values based on user iput
    with st.form(key='_form_scan_info'):
        mod = st.selectbox('Image Modality:', ['T1', 'FL'])
        suffix = st.text_input('Image Suffix:', value = suff)
        
        submitted = st.form_submit_button("Submit")
        flag_submit = False
        if submitted:
            flag_submit = True
        
    if flag_submit:
        # Move scans to consolidated path
        out_dir = os.path.join(st.session_state.paths['prj_dir'], mod)
        os.makedirs(out_dir, exist_ok=True)
        for fname in nifti_files:
            mrid = fname.replace(suffix, '')
            in_fpath = os.path.join(in_dir, fname)
            out_fpath = os.path.join(out_dir, mrid + '_' + mod + '.nii.gz')
            if os.path.exists(out_fpath):
                st.warning('Scan exists, will be updated!')
            shutil.move(in_fpath, out_fpath)
        utilio.clear_folder(in_dir)
        
        # Create participants list
        df['MRID'] = df.MRID.str.replace(suffix, '')
        odir = os.path.join(st.session_state.paths['prj_dir'], 'participants')
        ofile = os.path.join(odir, 'participants.csv')
        os.makedirs(odir, exist_ok=True)
        if not os.path.exists(ofile):
            df.to_csv(ofile, index=False)
        else:
            st.info('Participants list already exists, will not overwrite!')
        
        return True

    return False

@st.dialog("Scan/Participant Info", width='medium')
def dialog_consolidate_nifti_multiple():
    logger.debug('    Function: dialog_consolidate_nifti_multiple')
   
    if consolidate_nifti_multi():

        utilmisc.show_temp_message(
            f'Nifti files uploaded', 'success', 2
        )
        st.rerun()   
                   
@st.dialog("Dicom extraction", width='medium')
def dialog_extract_dicoms(in_dir, out_dir):
    if st.session_state.dicoms['df_dicoms'] is None:
        df_dicoms = utildcm.detect_series(in_dir)
        st.session_state.dicoms['df_dicoms'] = df_dicoms
        st.session_state.dicoms['list_series'] = df_dicoms.SeriesDesc.unique()
        st.session_state.dicoms['num_dicom_scans'] = df_dicoms[["PatientID", "StudyDate", "SeriesDesc"]].drop_duplicates().shape[0]

    dicoms = st.session_state.dicoms
    st.success(
        f"Detected {st.session_state.dicoms['num_dicom_scans']} scans in {len(st.session_state.dicoms['list_series'])} series!",
            icon=":material/thumb_up:"
    )

    sel_serie = st.selectbox(
        "Select series:", dicoms['list_series'], key = "key_select_dseries", index=0
    )
    if st.button("Convert to Nifti"):
        try:
            utildcm.convert_serie(dicoms['df_dicoms'], sel_serie, out_dir)
            
        except Exception as e:

            st.warning(":material/thumb_down: Nifti conversion failed!")
            utilmisc.show_temp_message(
                f'Conversion failed: {e}', 'warning', 2
            )

        st.session_state.curr_scan = st.session_state.participant['mrid'] + '.nii.gz'
        utilss.init_dicoms()
    
    if consolidate_nifti():

        utilmisc.show_temp_message(
            f'Nifti file uploaded', 'success', 2
        )
        st.rerun()   

def upload_files(in_files, out_dir):
    '''
    Copy uploaded files to target folder
    '''
    logger.debug('    Function: Upload_files')
    if in_files is None or len(in_files)==0:
        return False
    try:
        os.makedirs(out_dir, exist_ok=True)
        for in_file in in_files:
            fname = os.path.basename(in_file.name.replace("\\", "/"))
            f_out = os.path.join(out_dir, fname)
            if not os.path.exists(f_out):
                with open(f_out, "wb") as f:
                    f.write(in_file.getbuffer())
        return True    
    except:
        return False

def upload_nifti_single(in_file):
    '''
    Copy input nifti image to output folder
    '''
    logger.debug('    Function: Upload_nifti')

    # Upload file
    fname = in_file.name
    out_dir = os.path.join(st.session_state.paths['prj_dir'], '_upload', 'nifti')
    stat_upload = upload_files([in_file], out_dir)

    # Consolidate file
    if stat_upload:
        st.session_state.curr_scan = fname
        dialog_consolidate_nifti()

def upload_nifti_multi(in_files):
    '''
    Copy input nifti images to output folder
    '''
    logger.debug('    Function: Upload_nifti')

    # Upload files
    out_dir = os.path.join(st.session_state.paths['prj_dir'], '_upload', 'nifti')
    stat_upload = upload_files(in_files, out_dir)

    # Consolidate file
    if stat_upload:
        dialog_consolidate_nifti_multiple()

def upload_csv(in_file):
    '''
    Upload csv file
    '''
    logger.debug('    Function: Upload_csv')

    # Upload file
    fname = in_file.name
    out_dir = os.path.join(st.session_state.paths['prj_dir'], '_upload', 'nifti')
    stat_upload = upload_files([in_file], out_dir)

    # Consolidate file
    if stat_upload:
        consolidate_user_csv(fname)

def upload_dicom_zipped(in_file):
    '''
    Copy zipped dicms to output folder
    '''
    logger.debug('    Function: Upload_dicom_zipped')

    # Upload file
    fname = in_file.name
    dcm_dir = os.path.join(st.session_state.paths['prj_dir'], '_upload', 'dicoms')
    stat_upload = upload_files([in_file], dcm_dir)

    # Unzip file
    if stat_upload:
        # Unzip file
        utilio.unzip_zip_files(dcm_dir)

    # Extract dicoms
        nii_dir = os.path.join(st.session_state.paths['prj_dir'], '_upload', 'nifti')
        os.makedirs(nii_dir, exist_ok=True)
        dialog_extract_dicoms(dcm_dir, nii_dir)
    
def upload_dicom_folder(in_files):
    '''
    Copy dicoms folder to output folder
    '''
    logger.debug('    Function: Upload_dicom_folder')

    # Upload files
    dcm_dir = os.path.join(st.session_state.paths['prj_dir'], '_upload', 'dicoms')
    stat_upload = upload_files(in_files, dcm_dir)

    # Extract dicoms
    if stat_upload:
        nii_dir = os.path.join(st.session_state.paths['prj_dir'], '_upload', 'nifti')
        os.makedirs(nii_dir, exist_ok=True)
        dialog_extract_dicoms(dcm_dir, nii_dir)

    dialog_extract_dicoms(up_dir, ni_dir)

def extract_bids(bids_dir):
    '''
    Function to move image data in BIDS format to NiChart input format
    - T1 scans will be moved to os.path.join(st.session_state.paths['prj_dir'], 't1')
    - FL scans will be moved to os.path.join(st.session_state.paths['prj_dir'], 'fl')
    - image names will be {MRID}_{mod}.nii.gz
    - participants list in BIDS folder will be used to create the file participants/participants.csv with columns MRID, Age, Sex
    '''
    prj_dir = st.session_state.paths['prj_dir']

    # BIDS suffix → (output subdir, NiChart modality label)
    MOD_MAP = {
        'T1w':     ('t1', 'T1'),
        'FLAIR':   ('fl', 'FL'),
        'T2FLAIR': ('fl', 'FL'),
    }

    # ── 1. participants.tsv → participants/participants.csv ──────────────
    parts_tsv = os.path.join(bids_dir, 'participants.tsv')
    if os.path.isfile(parts_tsv):
        df_tsv = pd.read_csv(parts_tsv, sep='\t')
        col_map = {}
        for col in df_tsv.columns:
            lc = col.lower()
            if lc == 'participant_id':
                col_map[col] = 'MRID'
            elif lc == 'age':
                col_map[col] = 'Age'
            elif lc in ('sex', 'gender'):
                col_map[col] = 'Sex'
        df_parts = df_tsv.rename(columns=col_map)
        for req in ('MRID', 'Age', 'Sex'):
            if req not in df_parts.columns:
                df_parts[req] = None
        out_parts_dir = os.path.join(prj_dir, 'participants')
        os.makedirs(out_parts_dir, exist_ok=True)
        df_parts[['MRID', 'Age', 'Sex']].to_csv(
            os.path.join(out_parts_dir, 'participants.csv'), index=False
        )
        st.toast(f'Participants file created ({len(df_parts)} subjects)')
    else:
        st.toast('No participants.tsv found — participants.csv not created', icon='⚠️')

    # ── 2. Walk sub- dirs and copy images ────────────────────────────────
    n_copied = 0
    for entry in sorted(os.scandir(bids_dir), key=lambda e: e.name):
        if not entry.is_dir() or not entry.name.startswith('sub-'):
            continue
        mrid = entry.name  # e.g. 'sub-001'

        # collect anat dirs — handle optional ses- layer
        anat_dirs = []
        for sub in os.scandir(entry.path):
            if sub.is_dir():
                if sub.name == 'anat':
                    anat_dirs.append(sub.path)
                elif sub.name.startswith('ses-'):
                    ses_anat = os.path.join(sub.path, 'anat')
                    if os.path.isdir(ses_anat):
                        anat_dirs.append(ses_anat)

        for anat_dir in anat_dirs:
            for fname in sorted(os.listdir(anat_dir)):
                if not (fname.endswith('.nii.gz') or fname.endswith('.nii')):
                    continue
                for bids_suffix, (mod_dir, mod_label) in MOD_MAP.items():
                    if f'_{bids_suffix}.' in fname:
                        src = os.path.join(anat_dir, fname)
                        dst_dir = os.path.join(prj_dir, mod_dir)
                        os.makedirs(dst_dir, exist_ok=True)
                        dst = os.path.join(dst_dir, f'{mrid}_{mod_label}.nii.gz')
                        shutil.copy2(src, dst)
                        n_copied += 1
                        break

    st.toast(f'BIDS extraction complete — {n_copied} image(s) copied')

def upload_bids_folder(in_files):
    '''
    Copy bids folder to output folder
    '''
    logger.debug('    Function: Upload_bids_folder')

    # Upload files
    bids_dir = os.path.join(st.session_state.paths['prj_dir'], '_upload', 'bids')
    stat_upload = upload_files(in_files, bids_dir)

    # Extract BIDS
    if stat_upload:
        extract_bids(bids_dir)

##############################################################
## Main panels

def panel_project_folder():
    '''
    Panel to select project folder
    '''
    logger.debug('    Function: panel_project_folder')
    
    utilmisc._show_curr_prj()

    if st.checkbox("Change project folder"):
        with st.container(border=True):
            action = st.radio(
                "Action",
                ["Create new project", "Switch to existing project", "Reset project folder"],
                label_visibility='collapsed',
                index=0,
            )

            if action == "Create new project":
                sel_prj = st.text_input("Project name:", placeholder="my_study", label_visibility='collapsed')
                if st.button("Create") and sel_prj:
                    if utilss.init_project(sel_prj):
                        utilmisc.show_temp_message(
                            f'Project folder ready: {sel_prj}', 'success', 2
                        )
                        st.rerun()

            elif action == "Switch to existing project":
                list_projects = utilio.get_subfolders(st.session_state.paths['out_dir'])
                if not list_projects:
                    return
                sel_prj = sac.chip(
                    list_projects, label='', index=None, align='left', size='sm', radius='sm',
                    multiple=False, color='cyan'
                )
                if sel_prj is not None:
                    if st.button('Select'):
                        if utilss.init_project(sel_prj):
                            utilmisc.show_temp_message(
                                f'Project folder ready: {sel_prj}', 'success', 2
                            )
                            st.rerun()

            elif action == "Reset project folder":
                st.warning(":material/warning: This will delete all files in the project folder. This cannot be undone.")
                flag_confirm = st.checkbox("I understand and want to delete all files")
                if st.button("Delete", disabled=not flag_confirm):
                    utilio.clear_folder(st.session_state.paths['prj_dir'])
                    prj_name = st.session_state.prj_name
                    st.session_state.prj_name = None
                    utilss.init_project(prj_name)
                    utilmisc.show_temp_message(
                        'The project folder has been reset', 'success', 2
                    )
                    st.rerun()

def panel_upload_single_subject():
    '''
    Upload user data to target folder
    '''
    logger.debug('    Function: panel_upload_single_subject')
    utilmisc._show_curr_prj()

    # Select input file type
    dtype = st.radio(
        'Input data type:',
        ['Nifti', 'Dicom (single .zip)', 'Dicom (folder)', '.csv', 'BIDS (folder)'],
        horizontal=True
    )
    ftype = None
    accept_multiple_files = False
    if dtype == 'Nifti':
        ftype = ['.nii', '.nii.gz']
    elif dtype == 'Dicom (single .zip)':
        ftype = ['.zip']
    elif dtype == 'Dicom (folder)':
        accept_multiple_files = 'directory'
    elif dtype == '.csv':
        ftype = ['.csv']
    elif dtype == 'BIDS (folder)':
        accept_multiple_files = 'directory'

    # Upload file(s)
    with st.form(
        key='_form_upload_single_subj', clear_on_submit=True, border=False
    ):
        f = st.file_uploader(
            label = "Upload a file",
            type = ftype,
            accept_multiple_files = accept_multiple_files,
            label_visibility = 'collapsed'
        )
        if st.form_submit_button("Upload", use_container_width=False):
            if dtype == 'Nifti':
                upload_nifti_single(f)
            elif dtype == 'Dicom (single .zip)':
                upload_dicom_zipped(f)
            elif dtype == 'Dicom (folder)':
                upload_dicom_folder(f)
            elif dtype == '.csv':
                upload_csv(f)
            elif dtype == 'BIDS (folder)':
                upload_bids_folder(f)

def generate_template_csv():
    mod_dirs = {mod: os.path.join(st.session_state.paths['project'], mod) for mod in ['t1', 't2', 'fl', 'dti', 'fmri']}
    dir_dict = {'T1': mod_dirs['t1'],
                            'T2': mod_dirs['t2'],
                            'FLAIR': mod_dirs['fl'],
                            'DTI': mod_dirs['dti'],
                            'FMRI': mod_dirs['fmri'],
                            }
    nifti_parser = NiftiMRIDParser()
    heuristic_df = nifti_parser.create_master_csv(
        dir_dict,
        os.path.join(st.session_state.paths['project'], '_working' 'inferred_data_paths.csv')
    )
    
    
    df = heuristic_df.sort_values(by='MRID')
    df = df.drop_duplicates().reset_index().drop('index', axis=1)
    df = df.drop(columns=['T1_path', 'FLAIR_path', 'DTI_path', 'FMRI_path'], errors='ignore')
    
    # Add columns for batch and dx
    df[['Age']] = ''
    df[['Sex']] = ''
    df[['Batch']] = f'{st.session_state.project}_Batch1'
    df[['IsCN']] = 1
    
    
    return df

def panel_upload_multi_subject():
    '''
    Upload user data to target folder
    '''
    logger.debug('    Function: panel_upload_multi_subject')
    utilmisc._show_curr_prj()

    # Select input file type
    dtype = st.radio(
        'Input data type:',
        ['Nifti (folder)', 'Nifti (files)', 'Dicom (single .zip)', 'Dicom (folder)', '.csv'],
        horizontal=True
    )
    ftype = None
    accept_multiple_files = False
    if dtype == 'Nifti (folder)':
        ftype = ['.nii', '.nii.gz']
        accept_multiple_files = 'directory'
    elif dtype == 'Nifti (files)':
        ftype = ['.nii', '.nii.gz']
        accept_multiple_files = True
    elif dtype == 'Dicom (single .zip)':
        ftype = ['.zip']
    elif dtype == 'Dicom (folder)':
        accept_multiple_files = 'directory'
    elif dtype == '.csv':
        ftype = ['.csv']

    # Upload file(s)
    with st.form(
        key='_form_upload_single_subj', clear_on_submit=True, border=False
    ):
        f = st.file_uploader(
            label = "Upload a file",
            type = ftype,
            accept_multiple_files = accept_multiple_files,
            label_visibility = 'collapsed'
        )
        if st.form_submit_button("Upload", use_container_width=False):
            if dtype == 'Nifti (folder)':
                upload_nifti_multi(f)
            elif dtype == 'Nifti (files)':
                upload_nifti_multi(f)
            elif dtype == 'Dicom (single .zip)':
                upload_dicom_zipped(f)
            elif dtype == 'Dicom (folder)':
                upload_dicom_folder(f)
            elif dtype == '.csv':
                upload_csv(f)

def panel_upload_multi_subject_v2():
    with tab_t1:
        t1_out_dir = os.path.join(st.session_state.paths['prj_dir'], 't1')
        utilio.upload_multiple_files(out_dir=t1_out_dir)

    with tab_fl:
        fl_out_dir = os.path.join(st.session_state.paths['prj_dir'], 'fl')
        utilio.upload_multiple_files(out_dir=fl_out_dir)

    with tab_csv:
        try:
            df = generate_template_csv()
            st.download_button(
                "⬇ Download template CSV",
                df.to_csv(index=False).encode('utf-8'),
                'participants.csv',
                'text/csv',
                use_container_width=True
            )
        except Exception:
            st.caption("Upload MRI scans first to auto-generate a template.")

        with st.form(key='form_participants_csv', clear_on_submit=True, border=False):
            csv_file = st.file_uploader("Participants CSV", type=['csv'], label_visibility='collapsed')
            if st.form_submit_button("Upload", use_container_width=True):
                if csv_file:
                    dest = os.path.join(st.session_state.paths['prj_dir'], 'participants', 'participants.csv')
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with open(dest, 'wb') as f:
                        f.write(csv_file.getbuffer())
                    st.toast("Participants CSV uploaded!")
                else:
                    st.warning("Please select a CSV file first.")

def panel_view_files():
    '''
    Show files in data folder
    '''
    logger.debug('    Function: panel_view_files')
    utilmisc._show_curr_prj()

    col1, col2 = st.columns([1, 1])

    with col1:
        with st.container(border=None, height=400):
            tree_items, list_paths = utildv.build_folder_tree(
                st.session_state.paths['prj_dir'],
                st.session_state.out_dirs,
                None,
                3,
                ['_upload', '_working']
            )
            selected = sac.tree(
                items=tree_items,
                index=None,
                align='left', size='xl', icon='table',
                checkbox=False,
                #checkbox_strict = True,
                open_all = True,
                return_index = True
                #height=400
            )

    with col2:
        if selected:
            if isinstance(selected, list):
                selected = selected[0]
            fpath = list_paths[selected]
            fname = os.path.basename(fpath)
            if fpath.endswith('.csv'):
                try:
                    df_tmp = pd.read_csv(fpath)
                    st.dataframe(df_tmp, hide_index=True)

                    with st.container(horizontal=True, horizontal_alignment="center"):
                        if st.button('Edit'):
                            edit_participants(fpath)
                except:
                    st.warning(f'Could not read csv file: {fname}')

            if fpath.endswith(('.nii.gz','.nii')):
                utilmri.panel_view_mri_simple(fpath)

def panel_data():
    tab_prj, tab_upload, tab_review, tab_help = st.tabs(
        [":material/folder: Select Project",
            ":material/upload: Upload Data",
            ":material/fact_check: Review",
            ":material/help: Help"],
        on_change='rerun',
    )

    if tab_prj.open:
        with tab_prj:
            panel_project_folder()

    if tab_upload.open:
        with tab_upload:
            if st.session_state.workflow == 'single_subject':
                panel_upload_single_subject()
            else:
                panel_upload_multi_subject()

    if tab_review.open:
        with tab_review:
            panel_view_files()

    if tab_help.open:
        with tab_help:
            my_help()
    


