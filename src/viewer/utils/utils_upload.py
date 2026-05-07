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

##############################################################
## Functions to consolidate data


def my_help():

    tab1, tab2, tab3 = st.tabs(
        ["Overview", "Upload Files", "Review Files"],
        on_change = 'rerun'
    )

    if tab1.open:
        with tab1:
            st.write(
                """
                - All processing steps are performed inside a project folder.
                - By default, NiChart will create and use a current project folder for you.
                - You may also create a new project folder using any name you choose.
                - If needed, you can reset the current project folder (this will remove all files inside it, but keep the folder itself), allowing you to start fresh.
                - You may also switch to an existing project folder.

                **Note:** If you are using the cloud version, stored files will be removed periodically, so previously used project folders might not remain available.
                """
            )

    if tab2.open:
        with tab2:
            if st.session_state.workflow == 'single_subject':
                st.write(
                    """
                    - You may upload MRI scans in any of the following formats:
                        - **NIfTI:** .nii or .nii.gz
                        - **DICOM (compressed):** a single .zip file containing the DICOM series
                        - **DICOM (individual files):** multiple .dcm files

                        *(Note: uploading a folder directly is not currently supported)*

                    - If you have multiple imaging modalities (e.g., T1, FLAIR), upload them one at a time.

                    - Once uploaded, NiChart will automatically:
                        - Organize the files into the standard input structure
                        - Create a subject list based on the uploaded MRI data

                    - You may open and edit the subject list (e.g., to add age, sex, or other metadata needed for analysis).

                    - You can also upload non-imaging data (e.g., clinical or cognitive measures) as a CSV file.

                    - The CSV must include an MRID column with values that match the subject IDs in the subject list, so the data can be merged correctly.
                    """
                )

        if st.session_state.workflow == 'multi_subject':
                st.write(
                    """
                    - MRI Scans (NIfTI format):
                        - Upload one or more .nii / .nii.gz files or
                        - Upload a .zip file containing multiple NIfTI files

                        *(Note: uploading a folder directly is not currently supported)*

                    - If your dataset includes multiple imaging modalities (e.g., T1, FLAIR), upload each modality separately.

                    - Wait for the whole batch to upload before proceeding. (You'll see a few upload bars -- use the arrows to scroll through the batch.)

                    - For many pipelines, a participant CSV is required, containing at least one column:

                        **MRID** → subject ID that matches the scan filenames

                    - Filename Format Requirement:

                        {MRID}_common_suffix.nii.gz

                        Example: SUB001_T1.nii.gz

                    - After upload, NiChart will automatically:

                        - Organize scans into the standard input directory structure

                        - Check consistency between the participants CSV and the uploaded scans

                    - You may view and edit participants CSV after upload.

                    - Optional: Upload non-imaging data (e.g., clinical or cognitive variables) as an additional CSV (should include the MRID column).
                    """
                )

    if tab3.open:
        with tab3:
            st.write(
                """
                - View files stored in the project folder.

                - Click on a file name to:

                    - View a scan (.nii.gz, .nii)

                    - View/edit a list (.csv)
                """
            )

def view_mri(fname):
    """
    FIXME : move to mri utils
    Panel for viewing a nifti scan
    """
    with st.spinner("Wait for it..."):
        try:
            # Prepare final 3d matrix to display
            img = utilmri.prep_image(fname)

            # Detect mask bounds and center in each view
            img_bounds = utilmri.detect_img_bounds(img)

            # Show images
            ind_view = utilmri.img_views.index('axial')
            size_auto = True
            utilmri.show_img_slices(
                img,
                ind_view,
                img_bounds[ind_view, :],
                'axial',
                500
            )
        except:
            st.warning(
                ":material/thumb_down: Image parsing failed. Please confirm that the image file represents a 3D volume using an external tool."
            )


def copy_test_folders():
    '''
    Copy demo folders into user folders as needed
    '''
    if st.session_state.has_cloud_session:
        # Copy demo dirs to user folder (TODO: make this less hardcoded)
        demo_dir_paths = [
            os.path.join(
                st.session_state.paths["root"],
                "output_folder",
                "NiChart_Demo1",
            ),
            os.path.join(
                st.session_state.paths["root"],
                "output_folder",
                "NiChart_Demo2",
            ),
        ]
        for demo in demo_dir_paths:
            demo_name = os.path.basename(demo)
            destination_path = os.path.join(
                st.session_state.paths["out_dir"], demo_name
            )
            if os.path.exists(destination_path):
                shutil.rmtree(destination_path)
            shutil.copytree(demo, destination_path, dirs_exist_ok=True)

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
            st.success(f'Updated participants file: {fname}')
            time.sleep(0.6)
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

    st.success(f'CSV file uploaded ...')
    time.sleep(0.6)
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
    
    in_dir = os.path.join(st.session_state.paths['prj_dir'], '_upload')
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
    
    if consolidate_nifti():
        st.success(f'Nifti file uploaded ...')
        time.sleep(0.6)
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
        st.success(f'Nifti files uploaded ...')
        time.sleep(0.6)
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
            st.warning(e)
            time.sleep(3)

        st.session_state.curr_scan = st.session_state.participant['mrid'] + '.nii.gz'
        utilss.init_dicoms()
    
    if consolidate_nifti():
        st.success(f'Nifti file uploaded ...')
        time.sleep(0.6)
        st.rerun()   

def upload_file_multi_subject(in_file):
    '''
    Copy file to output folder
    '''
    logger.debug(f'    Function: upload_file({in_file})')
    if in_file is None:
        st.warning('Please select input file')
        time.sleep(3)
        return
    
    tmp_dir = os.path.join(st.session_state.paths['prj_dir'], '_upload')
    os.makedirs(tmp_dir, exist_ok=True)

    fname = in_file.name
    f_out = os.path.join(tmp_dir, fname)
    if not os.path.exists(f_out):
        with open(f_out, "wb") as f:
            f.write(in_file.getbuffer())

    st.toast(f'Uploaded file ...')

    if fname.endswith('.zip'):
        d_out = os.path.join(tmp_dir, 'nifti')
        utilio.unzip_zip_file(f_out, d_out)
        dialog_consolidate_nifti_multiple()

    elif fname.endswith('.csv'):
        st.write('Hello')
        consolidate_user_csv(fname)

    else:
        st.warning('Input file type mismatch: should be .zip or .csv')

def upload_nifti(in_file):
    '''
    Copy input nifti image to output folder
    '''
    logger.debug('    Function: Upload_nifti')

    if in_file is None:
        return False
    
    up_dir = os.path.join(st.session_state.paths['prj_dir'], '_upload')
    os.makedirs(up_dir, exist_ok=True)

    fname = in_file.name
    if not fname.endswith(('.nii.gz', '.nii')):
        return False

    f_out = os.path.join(up_dir, fname)
    if not os.path.exists(f_out):
        with open(f_out, "wb") as f:
            f.write(in_file.getbuffer())

    st.session_state.curr_scan = fname
    dialog_consolidate_nifti()

def upload_csv(in_file):
    '''
    Upload csv file
    '''
    logger.debug('    Function: Upload_csv')

    if in_file is None:
        return
    
    up_dir = os.path.join(st.session_state.paths['prj_dir'], '_upload')
    os.makedirs(up_dir, exist_ok=True)

    fname = in_file.name
    if not fname.endswith(('.csv')):
        return
    f_out = os.path.join(up_dir, fname)
    if not os.path.exists(f_out):
        with open(f_out, "wb") as f:
            f.write(in_file.getbuffer())

    consolidate_user_csv(fname)

def upload_dicom_zipped(in_file):
    '''
    Copy zipped dicms to output folder
    '''
    logger.debug('    Function: Upload_dicom_zipped')

    if in_file is None:
        return
    
    ni_dir = os.path.join(st.session_state.paths['prj_dir'], '_upload')
    up_dir = os.path.join(st.session_state.paths['prj_dir'], '_upload', 'dicoms')

    os.makedirs(up_dir, exist_ok=True)
    fname = in_file.name
    if not fname.endswith(('.zip')):
        return

    f_out = os.path.join(up_dir, fname)
    if not os.path.exists(f_out):
        with open(f_out, "wb") as f:
            f.write(in_file.getbuffer())
    st.toast(f'Uploaded file')

    # Unzip file
    utilio.unzip_zip_files(up_dir)

    # Extract dicoms
    tmp_dir = os.path.join(st.session_state.paths['prj_dir'], '_staging')
    dialog_extract_dicoms(up_dir, ni_dir)
    
def upload_dicom_folder(in_files):
    '''
    Copy zipped dicms to output folder
    '''
    logger.debug('    Function: Upload_dicom_zipped')

    if in_files is None or len(in_files)==0:
        return
    
    up_dir = os.path.join(st.session_state.paths['prj_dir'], '_upload', 'dicoms')
    ni_dir = os.path.join(st.session_state.paths['prj_dir'], '_upload')

    os.makedirs(up_dir, exist_ok=True)

    for in_file in in_files:
        fname = os.path.basename(in_file.name.replace("\\", "/"))
        f_out = os.path.join(up_dir, fname)
        if not os.path.exists(f_out):
            with open(f_out, "wb") as f:
                f.write(in_file.getbuffer())

    dialog_extract_dicoms(up_dir, ni_dir)
      
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

            elif action == "Switch to existing project":
                list_projects = utilio.get_subfolders(st.session_state.paths['out_dir'])
                if list_projects:
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

                else:
                    st.caption("No existing projects found.")

            elif action == "Reset project folder":
                st.warning(":material/warning: This will delete all files in the project folder. This cannot be undone.")
                flag_confirm = st.checkbox("I understand and want to delete all files")
                if st.button("Delete", disabled=not flag_confirm):
                    utilio.clear_folder(st.session_state.paths['prj_dir'])
                    st.toast(f"Files in '{st.session_state.prj_name}' deleted.")
                    prj_name = st.session_state.prj_name
                    st.session_state.prj_name = None
                    utilss.init_project(prj_name)
                    st.success('The project folder has been reset')
                    time.sleep(0.6)
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
        ['Nifti', 'Dicom (single .zip)', 'Dicom (folder)', '.csv'],
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
                upload_nifti(f)
            elif dtype == 'Dicom (single .zip)':
                upload_dicom_zipped(f)
            elif dtype == 'Dicom (folder)':
                upload_dicom_folder(f)
            elif dtype == '.csv':
                upload_csv(f)

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
    sac.divider(key='_p2_div4')
    st.markdown("##### 📤 Upload Data")

    tab_t1, tab_fl, tab_csv = st.tabs(["T1 Scans", "FLAIR Scans", "Participants CSV"])

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
                view_mri(fpath)

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
    


