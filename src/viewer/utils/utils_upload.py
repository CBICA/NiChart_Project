import streamlit as st
import utils.utils_dicoms as utildcm
import utils.utils_session as utilss
import utils.utils_data_view as utildv
import utils.utils_io as utilio
import utils.utils_mriview as utilmri

import os
import pandas as pd
import numpy as np
import zipfile
import streamlit_antd_components as sac
from NiChart_common_utils.nifti_parser import NiftiMRIDParser
import shutil
import time
from typing import Any, BinaryIO, List, Optional

from utils.utils_logger import setup_logger
logger = setup_logger()

##############################################################
## Functions to consolidate data

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
    in_dir = os.path.join(st.session_state.paths['prj_dir'], 'user_upload')
    in_fpath = os.path.join(in_dir, fname)

    with st.form(key='_form_csv_info'):
        sel_opt = st.selectbox('Use the file as participants file:', ['Yes', 'No'])

        submitted = st.form_submit_button("Submit")
        flag_submit = False
        if submitted:
            flag_submit = True
        
    if flag_submit:
        if sel_opt == 'Yes':
            # Move csv to consolidated path
            out_dir = os.path.join(st.session_state.paths['prj_dir'], 'participants')
            out_fpath = os.path.join(out_dir, 'participants.csv')
            os.makedirs(out_dir, exist_ok=True)
            if os.path.exists(out_fpath):
                st.warning('CSV file exists, will be overwritten!')
            shutil.move(in_fpath, out_fpath)        
            
        else:
            # Move csv to consolidated path
            out_dir = os.path.join(st.session_state.paths['prj_dir'], 'user_data')
            out_fpath = os.path.join(out_dir, fname)
            os.makedirs(out_dir, exist_ok=True)
            if os.path.exists(out_fpath):
                st.warning('CSV file exists, will be overwritten!')
            shutil.move(in_fpath, out_fpath)
        utilio.clear_folder(in_dir)

        st.toast(f'CSV file consolidated ...')
        st.rerun()   

def consolidate_nifti():
    logger.debug(f'    Function: consolidate_nifti')
    
    # Get full name for the current file
    # Input image file is kept in a temporary upload folder
    in_fname = st.session_state.curr_scan
    if in_fname is None:
        return False
    in_dir = os.path.join(st.session_state.paths['prj_dir'], 'user_upload')
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
        st.success('Updated participant info!')

        # Move scan to consolidated path
        out_dir = os.path.join(st.session_state.paths['prj_dir'], mod)
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
    logger.debug('    Function: dialog_consolidate_nifti')
    # Detect mrid
    mrid = st.session_state.participant['mrid']
    if mrid is None:
        mrid = st.session_state.curr_scan
        for suffix in ['.nii.gz', '.nii', '_T1', '_t1', '_FL', '_fl']:
            mrid = mrid.replace(suffix, '')
        st.session_state.participant['mrid'] = mrid
    
    if consolidate_nifti():
        st.toast(f'Nifti file consolidated ...')
        st.rerun()   

def detect_common_suffix(files):
    reversed_names = [f[::-1] for f in files]
    common_suffix = os.path.commonprefix(reversed_names)[::-1]
    return common_suffix

def consolidate_nifti_multi():
    logger.debug(f'    Function: consolidate_nifti')
    
    # Detect common suffix
    in_dir = os.path.join(st.session_state.paths['prj_dir'], 'user_upload', 'nifti')
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
        st.toast(f'Nifti files consolidated ...')
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
        utilss.reset_dicoms()
    
    if consolidate_nifti():
        st.rerun()    

def upload_file_single_subject(in_file):
    '''
    Copy file to output folder
    '''
    logger.debug(f'    Function: upload_file({in_file})')
    if in_file is None:
        st.warning('Please select input file')
        time.sleep(3)
        return
    
    tmp_dir = os.path.join(st.session_state.paths['prj_dir'], 'user_upload')
    os.makedirs(tmp_dir, exist_ok=True)

    fname = in_file.name
    f_out = os.path.join(tmp_dir, fname)
    if not os.path.exists(f_out):
        with open(f_out, "wb") as f:
            f.write(in_file.getbuffer())

    st.toast(f'Uploaded file ...')

    if fname.endswith(('.nii.gz', '.nii')):
        st.session_state.curr_scan = fname
        dialog_consolidate_nifti()
        
    elif fname.endswith('.csv'):
        consolidate_user_csv(fname)

    elif fname.endswith('.zip'):
        d_out = os.path.join(tmp_dir, 'unzipped')
        utilio.unzip_zip_file(f_out, d_out)
        dialog_extract_dicoms(d_out, tmp_dir)
        
    else:
        st.warning('Input file type mismatch: should be one of .nii.gz, .nii, .csv or .zip')

def upload_file_multi_subject(in_file):
    '''
    Copy file to output folder
    '''
    logger.debug(f'    Function: upload_file({in_file})')
    if in_file is None:
        st.warning('Please select input file')
        time.sleep(3)
        return
    
    tmp_dir = os.path.join(st.session_state.paths['prj_dir'], 'user_upload')
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

def upload_files_single_subject(in_files):
    '''
    Copy files to output folder
    '''
    logger.debug('    Function: Upload_files')

    if len(in_files) == 0:
        return
    
    tmp_dir = os.path.join(st.session_state.paths['prj_dir'], 'user_upload')
    d_out = os.path.join(tmp_dir, 'dicoms')
    
    os.makedirs(d_out, exist_ok=True)

    for in_file in in_files:
        f_out = os.path.join(d_out, in_file.name)
        if not os.path.exists(f_out):
            with open(f_out, "wb") as f:
                f.write(in_file.getbuffer())

    st.toast(f'Uploaded files ...')

    dialog_extract_dicoms(d_out, tmp_dir)

def upload_files_multi_subject(in_files):
    '''
    Copy files to output folder
    '''
    logger.debug('    Function: upload_files_multi_subject')

    if len(in_files) == 0:
        return
    
    tmp_dir = os.path.join(st.session_state.paths['prj_dir'], 'user_upload')
    d_out = os.path.join(tmp_dir, 'nifti')
    
    os.makedirs(d_out, exist_ok=True)

    for in_file in in_files:
        f_out = os.path.join(d_out, in_file.name)
        if not os.path.exists(f_out):
            with open(f_out, "wb") as f:
                f.write(in_file.getbuffer())

    st.toast(f'Uploaded files ...')

    ## Multiple nifti images
    dialog_consolidate_nifti_multiple()
       
def view_mri(fname):
    """
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
            )
        except:
            st.warning(
                ":material/thumb_down: Image parsing failed. Please confirm that the image file represents a 3D volume using an external tool."
            )

##############################################################
## Main panels

def panel_project_folder():
    '''
    Panel to select project folder
    '''
    logger.debug('    Function: panel_project_folder')
    sac.divider(key='_p1_div1')
    
    with st.container(horizontal=True, horizontal_alignment="left"):
        st.markdown("##### Project Folder: ", width='content')

    placeholder = st.empty()
    placeholder.markdown(f"##### 📁 `{st.session_state.prj_name}`", width='content')

    sel_opt = st.selectbox(
        'Select an action',
        ["I'm Good — Move On", 'Create new project folder', 'Switch to existing project', 'Reset project folder'],
        label_visibility='collapsed',
        index=None
    )

    if sel_opt == "I'm Good — Move On":
        st.success('Great! Please upload your data!')

    if sel_opt == 'Create new project folder':
        sel_prj = st.text_input(
            "Project name:",
            None,
            placeholder="user_new_study",
            label_visibility = 'collapsed'
        )

        with st.container(horizontal=True, horizontal_alignment="center"):
            if st.button("Select"):
                utilss.update_project(sel_prj)
                placeholder.markdown(f"##### 📃 `{st.session_state.prj_name}`", width='content')

    if sel_opt == 'Switch to existing project':
        list_projects = utilio.get_subfolders(st.session_state.paths['out_dir'])
        if len(list_projects) > 0:
            sel_ind = list_projects.index(st.session_state.prj_name)
            sel_prj = sac.chip(
                list_projects,
                label='', index=None, align='left', size='sm', radius='sm',
                multiple=False, color='cyan', description='Projects in output folder'
            )
            
            with st.container(horizontal=True, horizontal_alignment="center"):
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
                utilio.clear_folder(st.session_state.paths['prj_dir'])
                st.toast(f"Files in project {st.session_state.prj_name} have been successfully deleted.")
                utilss.update_project(st.session_state.prj_name)
        
def panel_upload_single_subject():
    '''
    Upload user data to target folder
    '''
    logger.debug('    Function: panel_upload_single_subject')

    sac.divider(key='_p2_div1')
    
    with st.container(horizontal=True, horizontal_alignment="left"):
        st.markdown("##### Upload File(s): ", width='content')
        with st.popover("❓", width='content'):
            st.write(
                """
                **Data Upload Guide**
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
            
    # Upload data
    sel_opt = sac.chip(
        ['Single (.nii.gz, .nii, .zip, .csv)', 'Multiple (dicom files)'],
        label='', index=0, align='left', size='sm', radius='sm', multiple=False, 
        color='cyan', return_index = True
    )
    flag_multi=False
    if sel_opt == 1:
        flag_multi=True
        
    logger.debug(f'**** flag multi set to : {flag_multi}')
        
    with st.form(key='my_form', clear_on_submit=True, border=False):
                
        sel_files = st.file_uploader(
            "Input files or folders",
            key="_uploaded_input",
            accept_multiple_files=flag_multi,
            label_visibility="collapsed"
        )
        
        flag_submit = False
        with st.container(horizontal=True, horizontal_alignment="center"):
            submitted = st.form_submit_button("Submit")
            if submitted:
                flag_submit = True
        
    logger.debug(f'**** flag submitted set to : {flag_submit}')
    logger.debug(f'**** sel files : {sel_files}')
    if flag_submit == True:
        if flag_multi == False:
            upload_file(sel_files)
        else:
            upload_files(sel_files)

def generate_template_csv():
    mod_dirs = {mod: os.path.join(st.session_state.paths['project'], mod) for mod in ['T1', 'T2', 'FL', 'DTI', 'FMRI']}
    dir_dict = {'T1': mod_dirs['T1'],
                            'T2': mod_dirs['T2'],
                            'FLAIR': mod_dirs['FL'],
                            'DTI': mod_dirs['DTI'],
                            'FMRI': mod_dirs['FMRI'],
                            }
    nifti_parser = NiftiMRIDParser()
    heuristic_df = nifti_parser.create_master_csv(dir_dict, os.path.join(st.session_state.paths['project'], 'inferred_data_paths.csv'))
    
    
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

    sac.divider(key='_p2_div1')
    
    with st.container(horizontal=True, horizontal_alignment="left"):
        st.markdown("##### Upload File(s): ", width='content')
        with st.popover("❓", width='content'):
            st.write(
                """
                **Data Upload Guide**
                - Here, upload the data you have available. (In the next step we'll automatically determine which pipelines you can run based on this.)

                - You may upload MRI scans in any of the following formats:
                  - **NIfTI:** one or multiple .nii or .nii.gz files 
                  - **DICOM (compressed):** a single .zip file containing the DICOM series
                  - **DICOM (individual files):** multiple .dcm files
                  
                    *(Note: uploading a folder directly is not currently supported)*
                    
                - If you have multiple imaging modalities (e.g., T1, FLAIR), upload only one modality batch at a time. First click the modality on the list, then drag-and-drop your images onto the box.
                
                - Once uploaded, NiChart will automatically:
                  - Organize the files into the standard input structure
                  - Create a subject list based on the uploaded MRI data
                  
                - You may open and edit the subject list (e.g., to add age, sex, or other metadata needed for analysis).
                
                - You can also upload non-imaging data (e.g., clinical or cognitive measures) as a CSV file (required for harmonization and many analytical pipelines).
                
                - The CSV must include an MRID column with values that match the subject IDs in the subject list, so the data can be merged correctly.

                - When you go to select a pipeline in the next step, if you select a pipeline which needs more fields, we'll tell you.
                """
            )
            
    # Upload data
    #sel_opt = sac.chip(
    #    ['Single (.nii.gz, .nii, .zip, .csv)', 'Multiple (dicom files)'],
    #    label='', index=0, align='left', size='sm', radius='sm', multiple=False, 
    #    color='cyan', return_index = True
    #)
    #sel_opt = sac.chip(
    #    ['T1 scans (.nii.gz, .nii, .zip)', 'FLAIR scans (.nii.gz, .nii, .zip)',
    #     'DICOM images (.dcm, .zip)', 'Participants CSV (.csv)'],
    #     label='', index=0, align='left', size='sm', radius='sm', multiple=False,
    #     color='cyan', return_index = True
    #)

    with st.popover("T1 Scans"):
        t1_out_dir = os.path.join(st.session_state.paths['prj_dir'], 't1')
        utilio.upload_multiple_files(out_dir=t1_out_dir)
    with st.popover("FLAIR Scans"):
        fl_out_dir = os.path.join(st.session_state.paths['prj_dir'], 'fl')
        utilio.upload_multiple_files(out_dir=fl_out_dir)
    #with st.popover("DICOM images"):
    #    pass
    with st.popover("Participants CSV"):
        st.markdown("## Participants CSV Upload")
        st.info("Most pipelines require some clinical or demographic information about participants. Below you can download a template CSV file to fill in. If you don't have information for a certain column, feel free to delete it. When ready, upload it with the file uploader and hit 'Submit'.")
        ## TODO: explain all columns here (expandable)
        with st.expander(label="Explanation of columns", expanded=False):
            st.write("Age must be in years. Sex must be M or F.")
            st.write("Batch is just used to identify particpants from a related source and is used for harmonization (not needed otherwise). We auto-generate a batch name assuming all scans are from a single population.")
            st.write("IsCN is used to denote cognitively normal patients (1 for CN, 0 otherwise). This is used for filtering during harmonization and not needed otherwise.")
        try:
            autogenerated_csv = generate_template_csv()
            autogenerated_csv_data = autogenerated_csv.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download template CSV",
                data=autogenerated_csv_data,
                file_name='participants.csv',
                mime="text/csv"
            )
        except Exception as e:
            st.warning("We couldn't seem to generate your template CSV. Please go back and ensure you uploaded scans first.")
        
        csv_file = st.file_uploader(
            "Input participants CSV",
            key="_uploaded_csv_input",
            accept_multiple_files=False,
            label_visibility="collapsed"
        )
        flag_csv_submit = False
        with st.container(horizontal=True, horizontal_alignment="center"):
            csv_submitted = st.button("Submit")
            if csv_submitted:
                flag_csv_submit = True
    
        logger.debug(f'**** flag csv submitted set to : {flag_csv_submit}')
        logger.debug(f'**** sel csv file : {csv_file}')
        if flag_csv_submit == True:
            try:
                dest_path = os.path.join(st.session_state.paths['prj_dir'], 'participants', 'participants.csv')
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                with open(dest_path, 'wb') as f:
                    f.write(csv_file.getbuffer())
                st.toast("Uploaded your CSV file successfully!")
            except:
                st.toast("Failed to upload CSV file.")
            #upload_file(csv_file)

    #flag_multi=True
    #    
    #logger.debug(f'**** flag multi set to : {flag_multi}')

def panel_view_files():
    '''
    Show files in data folder
    '''
    logger.debug('    Function: panel_view_files')
    
    sac.divider(key='_p3_div1')
    
    with st.container(horizontal=True, horizontal_alignment="left"):
        st.markdown("##### Review File(s): ", width='content')
            
    placeholder = st.empty()
    placeholder.markdown(f"##### 📁 `{st.session_state.prj_name}`", width='content')

    with st.container(border = None, height = 400):
        tree_items, list_paths = utildv.build_folder_tree(
            st.session_state.paths['prj_dir'],
            st.session_state.out_dirs,
            None,
            3,
            ['user_upload']
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

    if selected:
        if isinstance(selected, list):
            selected = selected[0]
        fpath = list_paths[selected]
        fname = os.path.basename(fpath)
        if fpath.endswith('.csv'):
            try:
                df_tmp = pd.read_csv(fpath)
                st.info(f'Data file: {fname}')
                st.dataframe(df_tmp, hide_index=True)

                with st.container(horizontal=True, horizontal_alignment="center"):
                    if st.button('Edit'):
                        edit_participants(fpath)
            except:
                st.warning(f'Could not read csv file: {fname}')

        if fpath.endswith(('.nii.gz','.nii')):
            view_mri(fpath)
            
                              
        


