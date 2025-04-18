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
    with st.container(border=True):
        # Detect dicom series
        btn_detect = st.button("Detect Series")
        if btn_detect:
            with st.spinner("Wait for it..."):
                df_dicoms = utildcm.detect_series(st.session_state.paths["dicoms"])
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

        utilst.util_help_dialog(utildoc.title_dicoms_detect, utildoc.def_dicoms_detect)

def panel_extract() -> None:
    """
    Panel for extracting dicoms
    """
    with st.container(border=True):
        sel_mod = "T1"

        # Check if data exists
        dout = st.session_state.paths[sel_mod]
        if st.session_state.flags[sel_mod]:
            st.success(
                f"Data is ready: {dout}",
                icon=":material/thumb_up:",
            )

            df_files = utilio.get_file_names(st.session_state.paths[sel_mod], ".nii.gz")
            with st.expander("View NIFTI image list"):
                st.dataframe(df_files)

            # Delete folder if user wants to reload
            if st.button("Reset", key="reset_extraction"):
                try:
                    if os.path.islink(dout):
                        os.unlink(dout)
                    else:
                        shutil.rmtree(dout)
                    st.session_state.flags[sel_mod] = False
                    st.success(f"Removed dir: {dout}")
                except:
                    st.error(f"Could not delete folder: {dout}")
                time.sleep(4)
                st.rerun()

        else:
            # Create out dir
            if not os.path.exists(dout):
                os.makedirs(dout)

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
                            dout,
                            f"_{sel_mod}.nii.gz",
                        )
                        st.session_state.flags[sel_mod] = True
                        # if st.session_state.has_cloud_session:
                        #     utilcloud.update_stats_db(
                        #         st.session_state.cloud_user_id, "NIFTIfromDICOM", num_nifti
                        #     )

                    except:
                        st.warning(":material/thumb_down: Nifti conversion failed!")

                time.sleep(1)
                st.rerun()

        utilst.util_help_dialog(
            utildoc.title_dicoms_extract, utildoc.def_dicoms_extract
        )


def panel_view(dtype:str) -> None:
    """
    Panel for viewing nifti images
    """
    in_dir = os.path.join(
        st.session_state.paths['task'], dtype
    )
    
    with st.container(border=True):
        # Check init status
        
        list_nifti = utilio.get_file_list(in_dir, ".nii.gz")

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
    list_opt = ["Load Data", "Detect Series", "Extract Scans", "View Scans"]
    sel_step = st.pills(
        "Select Step", list_opt, selection_mode="single", label_visibility="collapsed"
    )
    if sel_step == "Load Data":
        utilpn.util_panel_input_multi("dicoms", True)
    elif sel_step == "Detect Series":
        panel_detect()
    elif sel_step == "Extract Scans":
        panel_extract()
    elif sel_step == "View Scans":
        panel_view('T1')
        
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
    
    # Check out files
    folder_path = os.path.join(
        st.session_state.paths['task'], sel_mod.lower()
    )
    fcount = utilio.get_file_count(folder_path, ['.nii', '.nii.gz'])
    if fcount > 0:
        st.success(
            f" Input data available: ({fcount} nifti image files)",
            icon=":material/thumb_up:",
        )
        list_options = ["View", "Reset"]
        sel_step = st.pills(
            "Select Step",
            list_options,
            selection_mode="single",
            label_visibility="collapsed",
            default = None,
            key = '_sel_view_reset'
        )
        if sel_step == "View":
            panel_view(sel_mod.lower())
            
        if sel_step == "Reset":
            utilio.remove_dir(sel_mod.lower())
            if '_sel_view_reset' in st.session_state:
                del(st.session_state['_sel_view_reset'])
            st.rerun()

    else:
        list_options = ["Load"]
        sel_step = st.pills(
            "Select Step2",
            list_options,
            selection_mode="single",
            label_visibility="collapsed",
            default = None,
            key = '_key_sel_load'
        )       
        if sel_step == "Load":
            utilio.panel_input_multi(sel_mod.lower())
            st.rerun()


def panel_in_covars() -> None:
    """
    Panel for uploading covariates
    """
    
    st.write('hello')
    
    #Check out files
    file_path = os.path.join(
        st.session_state.paths['task'], 'lists', 'covars.csv'
    )
    if os.path.exists(file_path):
        st.success(
            f" Covariates file available: {file_path}",
            icon=":material/thumb_up:",
        )
        list_opt = ["View"]
        sel_step = st.pills(
            "Select Step",
            list_opt,
            selection_mode="single",
            label_visibility="collapsed",
            default = []
        )
        if sel_step == "View":
            try:
                df_cov = pd.read_csv(file_path)
                st.dataframe(df_cov)
            except:
                st.warning(f'Could not load dataframe: {file_path}')

        if st.button("Reset", key = '_btn_reset_covar'):
            utilio.remove_file(file_path)
                #st.rerun()
    
    else:
        sel_mod = st.pills(
            "Covar data type",
            ['Load File', 'Enter Manually'],
            selection_mode="single",
            label_visibility="collapsed"
        )
        if sel_mod is None:
            return
    
        if sel_mod == 'Load File':
            if st.session_state.app_type == "cloud":
                utilio.upload_file(
                    file_path,
                    "Demographics csv",
                    "uploaded_demog_file",
                )

            else:  # st.session_state.app_type == 'desktop'
                utilio.util_select_file(
                    "selected_demog_file",
                    "Demographics csv",
                    file_path,
                    st.session_state.paths["file_search_dir"],
                )
                
            #if os.path.exists(file_path):
                #st.rerun()


        elif sel_mod == 'Enter Manually':
            st.info("Please enter values for your sample")
            df_rois = pd.read_csv(st.session_state.paths["dlmuse_csv"])
            df_tmp = pd.DataFrame({"MRID": df_rois["MRID"], "Age": None, "Sex": None})
            df_user = st.data_editor(df_tmp)

            if st.button("Save data"):
                if not os.path.exists(
                    os.path.dirname(st.session_state.paths["demog_csv"])
                ):
                    os.makedirs(os.path.dirname(st.session_state.paths["demog_csv"]))

                df_user.to_csv(st.session_state.paths["demog_csv"], index=False)
                st.success(f"Data saved to {st.session_state.paths['demog_csv']}")


st.markdown(
    """
    ### Load input data
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
    list_opt_img = [
        "Nifti Images",
        "Dicom Files",
        "BIDS Data",
        "Connect to PACS Server",
    ]
    sel_task_img = st.pills(
        "Select Img Task",
        list_opt_img,
        selection_mode="single",
        label_visibility="collapsed",
        default=None,
        key='_sel_task_img'
    )

    if sel_task_img == "Nifti Images":
        with st.container(border=True):
            st.markdown(
                """
                **Nifti Images**
                Upload a folder containing Nifti images
                """
            )
            panel_nifti()

    elif sel_task_img == "Dicom Files":
        with st.container(border=True):
            st.markdown(
                """
                **Raw DICOM Files**
                Upload a folder containing unprocessed DICOM images (as exported by MRI scanners).
                The tool can convert DICOM to NIfTI internally or as part of a preprocessing pipeline step.
                This option is ideal if your data has not yet been converted or organized.
                """
            )
            panel_dicoms()
        
    elif sel_task_img == "BIDS Data":
        with st.container(border=True):
            st.markdown(
                """
                **BIDS Format**
                Load a dataset structured according to the [BIDS standard](https://bids.neuroimaging.io/), where all imaging modalities and metadata are organized in a single directory.
                This is the easiest option if your data is already standardized.
                """
            )
            st.warning('Work in progress ...')
            

    elif sel_task_img == "Connect to PACS Server":
        with st.container(border=True):
            st.markdown(
                """
                **Connect to PACS Server**
                Query and fetch imaging data directly from a hospital PACS server using DICOM networking.
                Requires PACS credentials and access permissions.
                """
            )
            st.warning('Work in progress ...')

elif sel_task == "Covariate File":
    del st.session_state['_sel_task_img']
    with st.container(border=True):
        st.markdown(
            """
            **Covariate File**
            Upload a csv file with covariate info (Age, Sex, DX, etc.)
            """
        )
        panel_in_covars()
