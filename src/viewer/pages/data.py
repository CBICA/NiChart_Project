import os
import shutil
import time
from typing import Any

import streamlit as st
import pandas as pd
import utils.utils_dicom as utildcm
import utils.utils_doc as utildoc
import utils.utils_io as utilio
import utils.utils_nifti as utilni
import utils.utils_pages as utilpg
#import utils.utils_panels as utilpn
import utils.utils_st as utilst
from stqdm import stqdm
import pandas as pd
import numpy as np

# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()

result_holder = st.empty()
def progress(p: int, i: int, decoded: Any) -> None:
    with result_holder.container():
        st.progress(p, f"Progress: Token position={i}")

def panel_detect() -> None:
    """
    Panel for detecting dicom series
    """
    
    dicom_folder = os.path.join(st.session_state.paths['task'], 'dicoms')
    
    with st.container(border=True):
        # Detect dicom series
        btn_detect = st.button("Detect Series")
        if btn_detect:
            with st.spinner("Wait for it..."):
                df_dicoms = utildcm.detect_series(dicom_folder)
                list_series = df_dicoms.SeriesDesc.unique()
                num_dicom_scans = (
                    df_dicoms[["PatientID", "StudyDate", "SeriesDesc"]]
                    .drop_duplicates()
                    .shape[0]
                )
                st.session_state.list_series = list_series
                st.session_state.num_dicom_scans = num_dicom_scans
                st.session_state.df_dicoms = df_dicoms

        if len(st.session_state.list_series) > 0:
            st.session_state.flags["dicoms_series"] = True
            st.success(
                f"Detected {st.session_state.num_dicom_scans} scans in {len(st.session_state.list_series)} series!",
                icon=":material/thumb_up:",
            )

        with st.expander("Show dicom metadata", expanded=False):
            st.dataframe(st.session_state.df_dicoms)

        #utilst.util_help_dialog(utildoc.title_dicoms_detect, utildoc.def_dicoms_detect)

def panel_extract() -> None:
    """
    Panel for extracting dicoms
    """
    sel_mod = "T1"

    dicom_folder = os.path.join(st.session_state.paths['task'], 'dicoms')
    out_folder = os.path.join(st.session_state.paths['task'], sel_mod.lower())
    
    with st.container(border=True):

        # Check if data exists
        if st.session_state.flags[sel_mod]:
            st.success(
                f"Data is ready: {out_folder}",
                icon=":material/thumb_up:",
            )

            df_files = utilio.get_file_names(out_folder, ".nii.gz")
            with st.expander("View NIFTI image list"):
                st.dataframe(df_files)

            # Delete folder if user wants to reload
            if st.button("Reset", key="reset_extraction"):
                try:
                    if os.path.islink(out_folder):
                        os.unlink(out_folder)
                    else:
                        shutil.rmtree(out_folder)
                    st.session_state.flags[sel_mod] = False
                    st.success(f"Removed dir: {out_folder}")
                except:
                    st.error(f"Could not delete folder: {out_folder}")
                time.sleep(4)
                st.rerun()

        else:
            # Create out dir
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)

            # Selection of dicom series
            st.session_state.sel_series = st.multiselect(
                "Select series for the T1 scan:", st.session_state.list_series, None
            )
            btn_convert = st.button("Convert Series")
            if btn_convert:
                with st.spinner("Wait for it..."):
                    try:
                        utildcm.convert_sel_series(
                            st.session_state.df_dicoms,
                            st.session_state.sel_series,
                            out_folder,
                            f"_{sel_mod}.nii.gz",
                        )
                        st.session_state.flags[sel_mod] = True
                        # if st.session_state.has_cloud_session:
                        #     utilcloud.update_stats_db(
                        #         st.session_state.cloud_user_id, "NIFTIfromDICOM", num_nifti
                        #     )

                    except:
                        st.warning(":material/thumb_down: NIfTI conversion failed!")

                time.sleep(1)
                st.rerun()

        #utilst.util_help_dialog(
            #utildoc.title_dicoms_extract, utildoc.def_dicoms_extract
        #)


def panel_view(dtype:str) -> None:
    """
    Panel for viewing nifti images
    """
    in_dir = os.path.join(
        st.session_state.paths['task'], dtype.lower()
    )
    
    with st.container(border=True):
        # Check init status
        
        list_nifti = utilio.get_file_list(in_dir, ".nii.gz")
        
        st.write(list_nifti)
        st.write(list_nifti)

        # Selection of image
        sel_img = utilst.user_input_select(
            "Select Image",
            "key_select_img",
            list_nifti,
            None,
            "FIXME: Help message",
            False,
        )
        if sel_img is None:
            return

        path_img = os.path.join(in_dir, sel_img)

        # Create a list of checkbox options
        list_orient = utilst.user_input_multiselect(
            "Select viewing planes:",
            "key_multiselect_viewplanes",
            utilni.img_views,
            utilni.img_views,
            "FIXME: Help message",
            flag_disabled=False,
        )

        if len(list_orient) == 0:
            return

        if not os.path.exists(path_img):
            return

        with st.spinner("Wait for it..."):

            try:
                # Prepare final 3d matrix to display
                img = utilni.prep_image(path_img)

                # Detect mask bounds and center in each view
                img_bounds = utilni.detect_img_bounds(img)

                # Show images
                blocks = st.columns(len(list_orient))
                for i, tmp_orient in stqdm(
                    enumerate(list_orient),
                    desc="Showing images ...",
                    total=len(list_orient),
                ):
                    with blocks[i]:
                        ind_view = utilni.img_views.index(tmp_orient)
                        size_auto = True
                        utilst.show_img3D(
                            img,
                            ind_view,
                            img_bounds[ind_view, :],
                            tmp_orient,
                            size_auto,
                        )
            except:
                st.warning(
                    ":material/thumb_down: Image parsing failed. Please confirm that the image file represents a 3D volume using an external tool."
                )

def panel_dicoms():
    list_opt = ["Upload", "Detect Series", "Extract Scans", "View", "Reset"]
    sel_step = st.pills(
        "Select Step", list_opt, selection_mode="single", label_visibility="collapsed"
    )
    if sel_step == "Upload":
        utilio.upload_multiple_files('dicoms')

    elif sel_step == "Detect Series":
        panel_detect()
        
    elif sel_step == "Extract Scans":
        panel_extract()
        
    elif sel_step == "View":
        panel_view('T1')
        
    elif sel_step == "Reset":
        utilio.remove_dir('dicoms')

def panel_nifti():
    sel_mod = st.pills(
        "Select Modality",
        st.session_state.list_mods,
        selection_mode="single",
        label_visibility="collapsed",
        default = None,
    )
    if sel_mod is None:
        return

    folder_path = os.path.join(
        st.session_state.paths['task'], sel_mod.lower()
    )
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    lists_path = os.path.join(
        st.session_state.paths['task'], 'lists'
    )
    if not os.path.exists(lists_path):
        os.makedirs(lists_path)
    
    list_options = ['Upload', 'Link', 'View', 'Reset']
    sel_step = st.pills(
        "Select Step",
        list_options,
        selection_mode="single",
        label_visibility="collapsed",
        default = None,
        key = '_key_sel_input_nifti'
    )       
    if sel_step == "Upload":
        # Upload data
        utilio.upload_multiple_files(sel_mod.lower())
        
        # Create list of scans
        df = utilio.create_img_list(sel_mod.lower())
        if df is not None:
            out_file = os.path.join(
                lists_path, 'list_nifti.csv'
            )
            df.to_csv(out_file, index=False)

    elif sel_step == "Link":
        st.write('!!! Not implemented yet !!!')
        
    elif sel_step == "View":
        fcount = utilio.get_file_count(folder_path, ['.nii', '.nii.gz'])
        if fcount == 0:
            st.warning('Input data not found!')
            return
        panel_view(sel_mod.lower())
            
    elif sel_step == "Reset":
        utilio.remove_dir(sel_mod.lower())
    
    fcount = utilio.get_file_count(folder_path, ['.nii', '.nii.gz'])
    if fcount > 0:
        st.success(
            f" Input data available: ({fcount} nifti image files)",
            icon=":material/thumb_up:",
        )

def panel_in_covars() -> None:
    """
    Panel for uploading covariates
    """    
    #Check out files
    file_path = os.path.join(
        st.session_state.paths['task'], 'lists', 'covars.csv'
    )
    
    list_options = ['Upload', 'Enter Manually', 'View', 'Reset']
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
            st.session_state.paths['task'], 'lists', 'list_nifti.csv'
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
    

st.info(
    """
    ### Upload Input Data
    - Panel to enter **:red[MRI image files] and/or **:red[CSV files] containing demographic and other information.
    - Supported file types and formats are indicated in each section to ensure smooth uploads.
    - Once your data is uploaded, you can proceed to apply pipelines tailored to your needs.
    """
)

list_opt = [
    "Image Data",
    "Covariate File",
]
sel_task = st.pills(
    "Select Task", list_opt, selection_mode="single", label_visibility="collapsed"
)

if sel_task == "Image Data":
    #list_opt_img = ["NIfTI", "DICOM", "BIDS", "PACS Server"]
    list_opt_img = ["NIfTI", "DICOM"]
    sel_task_img = st.pills(
        "Select Img Task",
        list_opt_img,
        selection_mode="single",
        label_visibility="collapsed",
        default=None,
        key='_sel_task_img'
    )

    if sel_task_img == "NIfTI":
        with st.container(border=True):
            st.markdown(
                """
                **NIfTI Images**
                - Upload NIfTI images
                """
            )
            panel_nifti()

    elif sel_task_img == "DICOM":
        with st.container(border=True):
            st.markdown(
                """
                **DICOM Files**
                
                - Upload a folder containing raw DICOM files
                - DICOM files will be converted to NIfTI scans
                """
            )
            panel_dicoms()
        
    elif sel_task_img == "BIDS Data":
        with st.container(border=True):
            st.markdown(
                """
                **BIDS Format**
                - Load a dataset structured according to the **:red[BIDS standard](https://bids.neuroimaging.io)**, where all imaging modalities and metadata are organized in a single directory.
                - This is the easiest option if your data is already standardized.
                """
            )
            st.warning('Work in progress ...')
            

    elif sel_task_img == "Connect to PACS Server":
        with st.container(border=True):
            st.markdown(
                """
                **Connect to PACS Server**
                - Query and fetch imaging data directly from a hospital PACS server using DICOM networking.
                - Requires PACS credentials and access permissions.
                """
            )
            st.warning('Work in progress ...')

elif sel_task == "Covariate File":
    with st.container(border=True):
        st.markdown(
            """
            **Covariate File**
            - Upload a **:red[csv file with covariate info]** (Age, Sex, DX, etc.)
            """
        )
        panel_in_covars()
