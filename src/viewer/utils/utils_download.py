import streamlit as st
import utils.utils_io as utilio
import streamlit_antd_components as sac
import os

def panel_info():
    with st.container(border=True):
        st.markdown(
            '''
            - NiChart Reference Dataset is a large and diverse collection from multiple MRI studies, created as part of the ISTAGING project to develop a system for identifying imaging biomarkers of aging and neurodegenerative diseases.

            - The dataset includes multi-modal MRI data, as well as carefully curated demographic, clinical, and cognitive variables from participants with a variety of health conditions.

            - The reference dataset is used for training machine learning models and for creating reference distributions of imaging measures and signatures

            - Users can compare their values to normative or disease-related reference distributions.            '''
        )
        st.image(
            os.path.join(
                st.session_state.paths['resources'], 'images', 'nichart_data.png'
            ),
            width=1200
        )

def prepare_data_for_download(prj_dir, sel_opt, out_zip):
    utilio.zip_folders(prj_dir, sel_opt, out_zip)
    with open(out_zip, "rb") as f:
        file_download = f.read()
    st.toast('Created zip file with selected folders')
    os.remove(out_zip)
    return file_download

def panel_download():
    '''
    Panel to download results
    '''
    if st.session_state.workflow == 'ref_data':
        st.info('Reference data download is not available at this time.')
        return
    
    with st.container(horizontal=True):

        st.markdown(f"###### 📁 Project Folder:   `{st.session_state.prj_name}`", width='content')
    
        prj_dir = st.session_state.paths['prj_dir']
        list_dirs = utilio.get_subfolders(prj_dir)
        for folder_name in ['download', 'downloads', 'user_upload']:
            if folder_name in list_dirs:
                list_dirs.remove(folder_name)
        
        if len(list_dirs) == 0:
            return
        
        sel_opt = sac.checkbox(
            list_dirs,
            label='Folder(s) to download:', 
            color='#aaeeaa', size='xl',
            check_all='Select all'
        )

        with st.container(horizontal=True):
            flag_disabled1 = True
            flag_disabled2 = True
            data_zip = ''
            if sel_opt is not None and len(sel_opt)>0:
                flag_disabled1 = False
            
            if st.button('Prepare Data', disabled = flag_disabled1):
                out_dir = os.path.join(prj_dir, 'downloads')
                os.makedirs(out_dir, exist_ok=True)
                out_zip = os.path.join(out_dir, 'nichart_results.zip')
                data_zip = prepare_data_for_download(prj_dir, sel_opt, out_zip)
                flag_disabled2 = False

            #st.download_button(
                #label = f"Download",
                #data = prepare_data_for_download(prj_dir, sel_opt, out_zip),
                #file_name = 'nichart_results.zip',
                #on_click = 'ignore'
            #)

            st.download_button(
                label = f"Download",
                data = data_zip,
                file_name = 'nichart_results.zip',
                disabled = flag_disabled2
            )