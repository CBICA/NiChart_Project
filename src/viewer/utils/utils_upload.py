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
        st.success(f"Removed dir: {folder}")
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

def unzip_zip_file(f_in, d_out):
    '''
    Unzips a ZIP file to a new folder and deletes the zip file
    '''
    os.makedirs(d_out, exist_ok=True)
    with zipfile.ZipFile(f_in, "r") as zip_ref:
        zip_ref.extractall(d_out)
    os.remove(f_in)
    st.toast(f'Unzipped zip file ...')

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
        use_container_width=True
    )
    if st.button('Save'):
        df_user.to_csv(in_file, index=False)
        st.success(f'Updated participants file: {fname}')
        
        st.rerun()


def select_project():
    """
    Panel for selecting a project
    """


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

def consolidate_nifti():
    
    # Get full name for the current file
    # Input image file is kept in a temporary upload folder
    in_fname = st.session_state.curr_scan
    if in_fname is None:
        return False
    in_dir = os.path.join(st.session_state.paths['prj_dir'], 'user_upload')
    in_fpath = os.path.join(in_dir, in_fname)

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
        clear_folder(in_dir)
        return True

    return False

@st.dialog("Scan/Participant Info", width='medium')
def dialog_consolidate_nifti():
    # Detect mrid
    mrid = st.session_state.participant['mrid']
    if mrid is None:
        mrid = st.session_state.curr_scan
        for suffix in ['.nii.gz', '.nii', '_T1', '_t1', '_FL', '_fl']:
            mrid = mrid.replace(suffix, '')
        st.session_state.participant['mrid'] = mrid
    
    if consolidate_nifti():
        st.rerun()    
            
import utils.utils_dicoms as utildcm
        
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
    
    st.info(f'Fname: {st.session_state.curr_scan}')

    if consolidate_nifti():
        st.rerun()    

def upload_file(in_file):
    '''
    Copy file to output folder
    '''
    if in_file is None:
        st.warning('Please select input file(s)')
        return
    
    st.toast(f'Uploading file ...')

    tmp_dir = os.path.join(st.session_state.paths['prj_dir'], 'user_upload')
    os.makedirs(tmp_dir, exist_ok=True)

    fname = in_file.name
    f_out = os.path.join(tmp_dir, fname)
    if not os.path.exists(f_out):
        with open(f_out, "wb") as f:
            f.write(in_file.getbuffer())

    if fname.endswith(('.nii.gz', '.nii')):
        st.session_state.curr_scan = fname
        dialog_consolidate_nifti()

    elif fname.endswith('.csv'):
        consolidate_csv(fname)

    elif fname.endswith('.zip'):
        d_out = os.path.join(tmp_dir, 'unzipped')
        unzip_zip_file(f_out, d_out)
        dialog_extract_dicoms(d_out, tmp_dir)
        
    else:
        st.warning('Input file type mismatch: should be one of .nii.gz, .nii, .csv or .zip')
        
    # Remove temp folder
    #shutil.rmtree(tmp_dir)

def upload_files(in_files):
    '''
    Copy files to output folder
    '''
    if len(in_files) == 0:
        return
    
    st.toast(f'Uploading files ...')

    tmp_dir = os.path.join(st.session_state.paths['prj_dir'], 'user_upload')
    os.makedirs(tmp_dir, exist_ok=True)

    for in_file in in_files:
        f_out = os.path.join(tmp_dir, in_file.name)
        if not os.path.exists(f_out):
            with open(f_out, "wb") as f:
                f.write(in_file.getbuffer())

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
        
    with st.form(key='my_form', clear_on_submit=True, border=False):
                
        sel_files = st.file_uploader(
            "Input files or folders",
            key="_uploaded_input",
            accept_multiple_files=flag_multi,
            label_visibility="collapsed"
        )
        
        flag_submit = False
        submitted = st.form_submit_button("Submit")
        if submitted:
            flag_submit = True
        
    if flag_submit == True:
        if flag_multi == False:
            upload_file(sel_files)
        else:
            upload_files(sel_files)
       
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
                  - Click on a file name to:
                  
                    - View a scan (.nii.gz, .nii)
                    
                    - View/edit a list (.csv)
                """
            )
            
    placeholder = st.empty()
    placeholder.markdown(f"##### 📁 `{st.session_state.prj_name}`", width='content')

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
                
                if st.button('Edit'):
                    edit_participants(fpath)                
            except:
                st.warning(f'Could not read csv file: {fname}')

        if fpath.endswith(('.nii.gz','.nii')):
            view_mri(fpath)
            
                              
        


