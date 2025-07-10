import streamlit as st
import utils.utils_dicoms as utildcm
import utils.utils_io as utilio
import utils.utils_session as utilss
import os
import pandas as pd
import streamlit_antd_components as sac
import shutil

from utils.utils_logger import setup_logger
logger = setup_logger()

def select_project(out_dir, curr_project):
    """
    Panel for creating/selecting project (to keep all data for the current project)
    """
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
        list_projects = utilio.get_subfolders(out_dir)
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
            
def disp_folder_tree(root_path):
    '''
    View contents of folder
    '''
    if not os.path.isdir(root_path):
        st.error("Invalid root path.")
        return

    st.markdown(f"##### 📂 `{root_path}`")

    folders = sorted([f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))])
    if not folders:
        st.info("Project folder is empty")
        return

    sel_folder = st.pills(
        'Select folder',
        folders,
        default = None,
        selection_mode = 'single',
        label_visibility = 'collapsed',
    )
    
    if sel_folder is None:
        return

    selected_path = os.path.join(root_path, sel_folder)
    st.markdown(f"---\n##### 📄 Files in `{sel_folder}`")

    try:
        files = sorted([
            f for f in os.listdir(selected_path)
            if os.path.isfile(os.path.join(selected_path, f))
        ])
        if not files:
            st.info("No files in this folder.")
        else:
            # Scrollable container
            data = [{
                "Filename": f,
            } for f in files]

            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

        csv_files = [f for f in files if f.lower().endswith(".csv")]
        
        if len(csv_files) > 0:
            st.markdown("##### 📊 Preview data files")
            sel_file = st.pills(
                'Select csv file',
                csv_files,
                default = None,
                selection_mode = 'single',
                label_visibility = 'collapsed',
            )        
            if sel_file is None:
                return

            csv_path = os.path.join(root_path, sel_folder, sel_file)
            try:
                df_csv = pd.read_csv(csv_path)
                st.success(f"Showing preview of `{sel_file}`")
                st.dataframe(df_csv, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to read `{sel_file}`: {e}")            
            
    except Exception as e:
        st.error(f"Error reading files: {e}")


def copy_uploaded_to_dir():
    '''
    Copies files to local storage
    '''
    if len(st.session_state['_uploaded_input']) > 0:
        utilio.copy_and_unzip_uploaded_files(
            st.session_state['_uploaded_input'], st.session_state.paths["target"]
        )

#def delete_folder():
    ## Delete folder if user wants to reload
    #if st.button("Reset", f"key_btn_reset_{dtype}"):
        #try:
            #if os.path.islink(out_dir):
                #os.unlink(out_dir)
            #else:
                #shutil.rmtree(out_dir)
            #st.session_state.flags[dtype] = False
            #st.success(f"Removed dir: {out_dir}")
            #time.sleep(4)
        #except:
            #st.error(f"Could not delete folder: {out_dir}")
        #st.rerun()

def upload_folder(out_dir, title_txt):
    '''
    Upload user data to target folder
    Input data may be a folder, multiple files, or a zip file (unzip the zip file if so)
    '''
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    st.session_state.paths['target'] = out_dir

    # Upload data
    st.file_uploader(
        title_txt,
        key='_uploaded_input',
        accept_multiple_files=True,
        on_change = copy_uploaded_to_dir,
    )

def upload_multi(out_dir):
    """
    Panel for uploading multiple input files or folder(s)
    """
    # Check if data exists
    if st.session_state.app_type == "cloud":
        # Upload data
        upload_folder(
            st.session_state.paths["dicoms"],
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
    fcount = utilio.get_file_count(st.session_state.paths[dtype])
    if fcount > 0:
        st.session_state.flags[dtype] = True
        p_dicom = st.session_state.paths[dtype]
        st.success(
            f" Uploaded data: ({p_dicom}, {fcount} files)",
            icon=":material/thumb_up:",
        )
        time.sleep(4)

        st.rerun()

        
def load_dicoms():
    tab = sac.tabs(
        items=[
            sac.TabsItem(label='Load Data'),
            sac.TabsItem(label='Detect Series'),
            sac.TabsItem(label='Extract Scans'),
            sac.TabsItem(label='View Scans'),
        ],
        size='lg',
        align='left'
    )
    
    if tab == "Load Data":
        out_dir = os.path.join(
            st.session_state.paths['project'], 'dicoms'
        )
        upload_multi(out_dir)
        
    elif tab == "Detect Series":
        panel_detect()
        
    elif tab == "Extract Scans":
        panel_extract()
        
    elif tab == "View Scans":
        panel_view('T1')

def load_nifti():
    sel_mod = sac.tabs(
        items=st.session_state.list_mods,
        size='lg',
        align='left'
    )
    
    if sel_mod is None:
        return

    folder_path = os.path.join(
        st.session_state.paths['project'], sel_mod.lower()
    )
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    lists_path = os.path.join(
        st.session_state.paths['project'], 'lists'
    )
    if not os.path.exists(lists_path):
        os.makedirs(lists_path)
    
    if st.button("Upload"):
        # Upload data
        utilio.upload_multiple_files(sel_mod.lower())
        
        # Create list of scans
        df = utilio.create_img_list(sel_mod.lower())
        if df is not None:
            out_file = os.path.join(
                lists_path, 'list_nifti.csv'
            )
            df.to_csv(out_file, index=False)
            
    if st.button("Reset"):
        utilio.remove_dir(sel_mod.lower())
    
    fcount = utilio.get_file_count(folder_path, ['.nii', '.nii.gz'])
    if fcount > 0:
        st.success(
            f" Input data available: ({fcount} nifti image files)",
            icon=":material/thumb_up:",
        )

def load_csv():
    """
    Panel for uploading covariates
    """    
    #Check out files
    file_path = os.path.join(
        st.session_state.paths['project'], 'lists', 'covars.csv'
    )
    
    list_options = ['Upload', 'Enter Manually', 'Reset']
    sel_step = st.pills(
        "Select Step",
        list_options,
        selection_mode="single",
        label_visibility="collapsed",
        default = None,
        key = '_key_sel_input_covar'
    )       
    if sel_step == "Upload":
        utilio.upload_single_file('lists', 'demog.csv', '.csv')

    elif sel_step == 'Enter Manually':
        id_list = os.path.join(
            st.session_state.paths['project'], 'lists', 'list_nifti.csv'
        )
        try:
            df = pd.read_csv(id_list)
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
            df_user.to_csv(file_path, index=False)
            st.success(f'Created demographic file: {file_path}')
        

    elif sel_step == "View":
        if not os.path.exists(file_path):
            st.warning('Covariate file not found!')
            return
        try:
            df_cov = pd.read_csv(file_path)
            st.dataframe(df_cov)
        except:
            st.warning(f'Could not load covariate file: {file_path}')
        
    elif sel_step == "Reset":
        utilio.remove_dir('lists')

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
            #st.markdown(
                #"""
                #***NIfTI Images***
                #- Upload NIfTI images
                #"""
            #)
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
