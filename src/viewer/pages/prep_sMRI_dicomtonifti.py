import os
from typing import Any

import streamlit as st
import utils.utils_cloud as utilcloud
import utils.utils_dicom as utildcm
import utils.utils_io as utilio
import utils.utils_menu as utilmenu
import utils.utils_nifti as utilni
import utils.utils_session as utilss
import utils.utils_st as utilst
import utils.utils_doc as utildoc
import utils.utils_panels as utilpn
from stqdm import stqdm
import time
import shutil

# Page config should be called for each page
utilss.config_page()

result_holder = st.empty()

utilmenu.menu()

st.write("# Dicom to Nifti Conversion")

def progress(p: int, i: int, decoded: Any) -> None:
    with result_holder.container():
        st.progress(p, f"Progress: Token position={i}")

def panel_detect(status) -> None:
    """
    Panel for detecting dicom series
    """
    with st.container(border=True):
        # Check init status
        if not status:
            st.warning('Please check previous step!')
            return
            flag_disabled = not st.session_state.flags["dicoms"]

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


def panel_extract(status) -> None:
    """
    Panel for extracting dicoms
    """
    with st.container(border=True):
        # Check init status
        if not status:
            st.warning('Please check previous step!')
            return

        sel_mod = 'T1'

        # Check if data exists
        dout = st.session_state.paths[sel_mod]
        if st.session_state.flags[sel_mod]:
            st.success(
                f"Data is ready: {dout}",
                icon=":material/thumb_up:",
            )

            df_files = utilio.get_file_names(
                st.session_state.paths[sel_mod], ".nii.gz"
            )
            with st.expander("View NIFTI image list"):
                st.dataframe(df_files)

            # Delete folder if user wants to reload
            if st.button('Reset', key='reset_extraction'):
                try:
                    if os.path.islink(dout):
                        os.unlink(dout)
                    else:
                        shutil.rmtree(dout)
                    st.session_state.flags[sel_mod] = False
                    st.success(f'Removed dir: {dout}')
                except:
                    st.error(f'Could not delete folder: {dout}')
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

        utilst.util_help_dialog(utildoc.title_dicoms_extract, utildoc.def_dicoms_extract)


def panel_view(status) -> None:
    """
    Panel for viewing extracted nifti images
    """
    with st.container(border=True):
        # Check init status
        if not status:
            st.warning('Please check previous step!')
            return

        sel_mod = 'T1'
        list_nifti = utilio.get_file_list(
            st.session_state.paths[sel_mod], ".nii.gz"
        )

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

        path_img = os.path.join(
            st.session_state.paths[sel_mod], sel_img
        )

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

st.markdown(
    """
    - Extracts raw DICOM files to NIFTI format.
    - Automatically identifies different imaging series.
    - Allows users to select specific series for extraction.
    - Generates consistently named NIFTI files based on DICOM information.
    - Provides a visual review of extracted images.
    """
)

# Call all steps
if st.session_state.app_type == "cloud":
    t1, t2, t3, t4, t5, t6 =  st.tabs(
        ['Experiment Name', 'Input Data', 'Detect Series', 'Extract Scans', 'View Scans', 'Download']
    )
else:
    t1, t2, t3, t4, t5 =  st.tabs(
        ['Experiment Name', 'Input Data', 'Detect Series', 'Extract Scans', 'View Scans']
    )

with t1:
    utilpn.util_panel_experiment()
with t2:
    # panel_indicoms()
    status = st.session_state.flags['experiment']
    utilpn.util_panel_input_multi('dicoms', status)
with t3:
    status = st.session_state.flags['dicoms']
    panel_detect(status)
with t4:
    status = st.session_state.flags['dicoms_series']
    panel_extract(status)
with t5:
    status = st.session_state.flags['T1']
    panel_view(status)
if st.session_state.app_type == "cloud":
    with t6:
        status = st.session_state.flags['dicoms']
        utilpn.util_panel_download('T1', status)

# FIXME: For DEBUG
utilst.add_debug_panel()
