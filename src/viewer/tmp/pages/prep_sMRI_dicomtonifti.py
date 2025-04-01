import os
from typing import Any

import streamlit as st
import utils.utils_dicom as utildcm
import utils.utils_io as utilio
import utils.utils_nifti as utilni
import utils.utils_st as utilst
from menu import menu
from stqdm import stqdm

result_holder = st.empty()

menu()

def progress(p: int, i: int, decoded: Any) -> None:
    with result_holder.container():
        st.progress(p, f"Progress: Token position={i}")

def panel_wdir():
    '''
    Panel for selecting the working dir
    '''
    icon = st.session_state.icon_thumb[st.session_state.flags['dir_out']]
    show_panel_wdir = st.checkbox(
        f":material/folder_shared: Working Directory {icon}",
        value = False
    )
    if not show_panel_wdir:
        return

    with st.container(border=True):
        utilst.util_panel_workingdir(st.session_state.app_type)
        if os.path.exists(st.session_state.paths["dset"]):
            st.success(
                f"All results will be saved to: {st.session_state.paths['dset']}",
                icon=":material/thumb_up:",
            )
            st.session_state.flags["dir_out"] = True

def panel_indicoms():
    '''
    Panel for selecting input dicoms
    '''
    msg = st.session_state.app_config[st.session_state.app_type]['msg_infile']
    icon = st.session_state.icon_thumb[st.session_state.flags['dir_dicom']]
    show_panel_indicoms = st.checkbox(
        f":material/upload: {msg} Dicoms {icon}",
        disabled = not st.session_state.flags['dir_out'],
        value = False
    )
    if not show_panel_indicoms:
        return

    with st.container(border=True):
        if st.session_state.app_type == "cloud":
            # Upload data
            utilst.util_upload_folder(
                st.session_state.paths["dicom"],
                "Dicom files or folders",
                False,
                "Raw dicom files can be uploaded as a folder, multiple files, or a single zip file"
            )
            fcount = utilio.get_file_count(st.session_state.paths["dicom"])
            if fcount > 0:
                st.session_state.flags['dir_dicom'] = True
                st.success(
                    f"Data is ready ({st.session_state.paths["dicom"]}, {fcount} files)",
                    icon=":material/thumb_up:"
                )

        else:  # st.session_state.app_type == 'desktop'
            utilst.util_select_folder(
                "selected_dicom_folder",
                "Dicom folder",
                st.session_state.paths["dicom"],
                st.session_state.paths["last_in_dir"],
                False,
            )
            fcount = utilio.get_file_count(st.session_state.paths["dicom"])
            if fcount > 0:
                st.session_state.flags['dir_dicom'] = True
                st.success(
                    f"Data is ready ({st.session_state.paths["dicom"]}, {fcount} files)",
                    icon=":material/thumb_up:"
                )

def panel_detect():
    '''
    Panel for detecting dicom series
    '''
    icon = st.session_state.icon_thumb[st.session_state.flags['dicom_series']]
    show_panel_detect = st.checkbox(
        f":material/manage_search: Detect Dicom Series {icon}",
        disabled = not st.session_state.flags['dir_dicom'],
        value = False
    )
    if not show_panel_detect:
        return

    with st.container(border=True):
        flag_disabled = not st.session_state.flags['dir_dicom']

        # Detect dicom series
        num_scans = 0
        btn_detect = st.button("Detect Series", disabled=flag_disabled)
        if btn_detect:
            with st.spinner("Wait for it..."):
                df_dicoms = utildcm.detect_series(st.session_state.paths["dicom"])
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
            st.session_state.flags['dicom_series'] = True
            st.success(
                f"Detected {st.session_state.num_dicom_scans} scans in {len(st.session_state.list_series)} series!",
                icon=":material/thumb_up:"
            )

def panel_extract():
    '''
    Panel for extracting dicoms
    '''
    icon = st.session_state.icon_thumb[st.session_state.flags['dir_nifti']]
    show_panel_extract = st.checkbox(
        f":material/auto_awesome_motion: Extract Scans {icon}",
        disabled = not st.session_state.flags['dicom_series'],
        value = False
    )
    if not show_panel_extract:
        return

    with st.container(border=True):

        flag_disabled = not st.session_state.flags['dicom_series']

        # Selection of img modality
        sel_mod = utilst.user_input_select(
            "Image Modality",
            'key_select_modality',
            st.session_state.list_mods,
            None,
            "Modality of the extracted images",
            flag_disabled
        )
        if sel_mod is not None:
            st.session_state.sel_mod = sel_mod
            if not os.path.exists(st.session_state.paths[st.session_state.sel_mod]):
                os.makedirs(st.session_state.paths[st.session_state.sel_mod])

        # Selection of dicom series
        st.session_state.sel_series = utilst.user_input_multiselect(
            "Select series:",
            "key_multiselect_dseries",
            st.session_state.list_series,
            [],
            '',
            flag_disabled=flag_disabled
        )

        btn_convert = st.button("Convert Series", disabled=flag_disabled)
        if btn_convert:
            with st.spinner("Wait for it..."):
                utildcm.convert_sel_series(
                    st.session_state.df_dicoms,
                    st.session_state.sel_series,
                    st.session_state.paths[st.session_state.sel_mod],
                    f"_{st.session_state.sel_mod}.nii.gz",
                )

        num_nifti = utilio.get_file_count(
            st.session_state.paths[st.session_state.sel_mod],
            '.nii.gz'
        )
        if num_nifti > 0:
            st.session_state.flags['dir_nifti'] = True
            st.session_state.flags[st.session_state.sel_mod] = True
            st.success(
                f"Nifti images are ready ({st.session_state.paths[st.session_state.sel_mod]}, {num_nifti} scan(s))",
                icon=":material/thumb_up:",
            )

def panel_view():
    '''
    Panel for viewing extracted nifti images
    '''

    show_panel_view = st.checkbox(
        f":material/visibility: View Scans",
        disabled = not st.session_state.flags['dir_nifti'],
        value = False
    )
    if not show_panel_view:
        return

    with st.container(border=True):

        # Selection of img modality
        sel_mod = utilst.user_input_select(
            "Image Modality",
            'key_selbox_modality_viewer',
            st.session_state.list_mods,
            None,
            "Modality of the images to view",
            False
        )

        list_nifti = []
        if sel_mod is None:
            return

        st.session_state.sel_mod = sel_mod
        list_nifti = utilio.get_file_list(
            st.session_state.paths[st.session_state.sel_mod], '.nii.gz'
        )

        # Selection of image
        sel_img = utilst.user_input_select(
            "Select Image",
            "key_select_img",
            list_nifti,
            None,
            'FIXME: Help message',
            False
        )
        if sel_img is None:
            return

        st.session_state.paths["sel_img"] = os.path.join(
            st.session_state.paths[st.session_state.sel_mod], sel_img
        )

        # Create a list of checkbox options
        list_orient = utilst.user_input_multiselect(
            "Select viewing planes:",
            "key_multiselect_viewplanes",
            utilni.img_views,
            utilni.img_views,
            'FIXME: Help message',
            flag_disabled=False
        )

        if len(list_orient) == 0:
            return

        if not os.path.exists(st.session_state.paths["sel_img"]):
            return

        with st.spinner("Wait for it..."):

            # Prepare final 3d matrix to display
            img = utilni.prep_image(st.session_state.paths["sel_img"])

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
                    utilst.show_img3D(
                        img, ind_view, img_bounds[ind_view, :], tmp_orient
                    )

def panel_download():
    '''
    Panel for viewing extracted nifti images
    '''
    show_panel_view = st.checkbox(
        f":material/download: Download Scans",
        disabled = not st.session_state.flags['dir_nifti'],
        value = False
    )
    if not show_panel_view:
        return

    with st.container(border=True):
        # Selection of img modality
        sel_mod = utilst.user_input_select(
            "Image Modality",
            'key_selbox_modality_download',
            ['All'] + st.session_state.list_mods,
            None,
            "Modality of the images to download",
            False
        )

        if sel_mod is not None:
            st.session_state.sel_mod = sel_mod
            if st.session_state.sel_mod == 'All':
                st.session_state.sel_mod = 'nifti'

        out_zip = bytes()
        if not os.path.exists(st.session_state.paths["download"]):
            os.makedirs(st.session_state.paths["download"])
        f_tmp = os.path.join(
            st.session_state.paths["download"],
            f"{st.session_state.sel_mod}"
        )
        out_zip = utilio.zip_folder(st.session_state.paths[st.session_state.sel_mod], f_tmp)

        st.download_button(
            "Download Extracted Scans",
            out_zip,
            file_name=f"{st.session_state.dset}_{st.session_state.sel_mod}.zip",
            disabled = False
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
st.divider()
panel_wdir()
panel_indicoms()
panel_detect()
panel_extract()
panel_view()
if st.session_state.app_type == "cloud":
    panel_download()

# FIXME: For DEBUG
utilst.add_debug_panel()
