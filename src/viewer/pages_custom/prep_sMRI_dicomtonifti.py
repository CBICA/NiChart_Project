import os
from typing import Any

import streamlit as st
import utils.utils_dicom as utildcm
import utils.utils_io as utilio
import utils.utils_nifti as utilni
import utils.utils_st as utilst
from stqdm import stqdm

result_holder = st.empty()


def progress(p: int, i: int, decoded: Any) -> None:
    with result_holder.container():
        st.progress(p, f"Progress: Token position={i}")


st.markdown(
    """
    - Extracts raw DICOM files to NIFTI format.
    - Automatically identifies and separates different imaging series.
    - Allows users to select specific series for extraction.
    - Generates consistently named NIFTI files based on DICOM information.
    - Provides a visual review of extracted images.
    """
)

# Panel for output (dataset name + out_dir)
utilst.util_panel_workingdir(st.session_state.app_type)

# Panel for selecting input dicom files
flag_disabled = not st.session_state.flags['dset']

if st.session_state.app_type == "CLOUD":
    with st.expander(":material/upload: Upload data", expanded=False):  # type:ignore
        utilst.util_upload_folder(
            st.session_state.paths["Dicoms"],
            "Dicom files or folders",
            flag_disabled,
            "Raw dicom files can be uploaded as a folder, multiple files, or a single zip file",
        )

else:  # st.session_state.app_type == 'DESKTOP'
    with st.expander(":material/upload: Select data", expanded=False):  # type:ignore
        utilst.util_select_folder(
            "selected_dicom_folder",
            "Dicom folder",
            st.session_state.paths["Dicoms"],
            st.session_state.paths["last_in_dir"],
            flag_disabled,
        )

# Panel for detecting dicom series
with st.expander(":material/manage_search: Detect dicom series", expanded=False):

    flag_disabled = not st.session_state.flags['Dicoms']

    # Detect dicom series
    num_scans = 0
    btn_detect = st.button("Detect Series", disabled=flag_disabled)
    if btn_detect:
        with st.spinner("Wait for it..."):
            df_dicoms = utildcm.detect_series(st.session_state.paths["Dicoms"])
            list_series = df_dicoms.SeriesDesc.unique()
            num_dicom_scans = (
                df_dicoms[["PatientID", "StudyDate", "SeriesDesc"]]
                .drop_duplicates()
                .shape[0]
            )
            st.session_state.list_series = list_series
            st.session_state.num_dicom_scans = num_dicom_scans
            st.session_state.df_dicoms = df_dicoms
            if len(list_series) == 0:
                st.error("Could not detect any dicom series!")
    if len(st.session_state.list_series) > 0:
        st.success(
            f"Detected {st.session_state.num_dicom_scans} scans in {len(st.session_state.list_series)} series!",
            icon=":material/thumb_up:",
        )
        st.session_state.flags['dicom_series'] = True

# Panel for selecting and extracting dicom series
with st.expander(":material/auto_awesome_motion: Extract scans", expanded=False):

    flag_disabled = not st.session_state.flags['dicom_series']

    # Selection of img modality
    helpmsg = "Modality of the extracted images"
    sel_mod = utilst.user_input_select(
        "Image Modality",
        st.session_state.list_mods,
        'key_selbox_modality',
        helpmsg,
        flag_disabled
    )
    if sel_mod is not None:
        st.session_state.sel_mod = sel_mod
        
    # Selection of dicom series
    st.session_state.sel_series = st.multiselect(
        "Select series:", st.session_state.list_series, []
    )
    # Create out folder for the selected modality
    if len(st.session_state.sel_series) > 0:
        if not os.path.exists(st.session_state.paths[st.session_state.sel_mod]):
            os.makedirs(st.session_state.paths[st.session_state.sel_mod])

    btn_convert = st.button("Convert Series", disabled=flag_disabled)
    if btn_convert:
        with st.spinner("Wait for it..."):
            utildcm.convert_sel_series(
                st.session_state.df_dicoms,
                st.session_state.sel_series,
                st.session_state.paths[st.session_state.sel_mod],
                f"_{st.session_state.sel_mod}.nii.gz",
            )
            st.session_state.list_input_nifti = [
                f
                for f in os.listdir(st.session_state.paths[st.session_state.sel_mod])
                if f.endswith("nii.gz")
            ]
            if len(st.session_state.list_input_nifti) == 0:
                st.error("Could not extract any nifti images")
            else:
                st.session_state.flags[st.session_state.sel_mod] = True

    if st.session_state.flags[st.session_state.sel_mod]:
        st.success(
            f"Nifti images are ready ({st.session_state.paths[st.session_state.sel_mod]}, {len(st.session_state.list_input_nifti)} scan(s)",
            icon=":material/thumb_up:",
        )

# Panel for viewing extracted nifti images
with st.expander(":material/visibility: View images", expanded=False):

    flag_disabled = not st.session_state.flags[st.session_state.sel_mod]

    # Selection of MRID
    sel_img = st.selectbox(
        "Select Image",
        st.session_state.list_input_nifti,
        key="selbox_images",
        index=None,
        disabled=flag_disabled
    )

    if sel_img is not None:
        st.session_state.paths["sel_img"] = os.path.join(
            st.session_state.paths[st.session_state.sel_mod], sel_img
        )
        if os.path.exists(st.session_state.paths["sel_img"]):
            st.session_state.flags['sel_img'] = True

    # Create a list of checkbox options
    flag_img = st.session_state.flags['sel_img']
    list_orient = st.multiselect(
        "Select viewing planes:",
        utilni.img_views,
        utilni.img_views,
        disabled=not flag_img
    )

    if flag_img:
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

# Panel for downloading results
if st.session_state.app_type == "CLOUD":
    with st.expander(":material/download: Download Results", expanded=False):

        flag_disabled = not st.session_state.flags[st.session_state.sel_mod]

        out_zip = bytes()
        if not flag_disabled:
            if not os.path.exists(st.session_state.paths["out_zipped"]):
                os.makedirs(st.session_state.paths["out_zipped"])
            f_tmp = os.path.join(st.session_state.paths["out_zipped"], f"{st.session_state.sel_mod}.zip")
            out_zip = utilio.zip_folder(st.session_state.paths[st.session_state.sel_mod], f_tmp)

        st.download_button(
            "Download Extracted Scans",
            out_zip,
            file_name=f"{st.session_state.sel_mod}.zip",
            disabled = flag_disabled
        )

if st.session_state.debug_show_state:
    with st.expander("DEBUG: Session state - all variables"):
        st.write(st.session_state)

if st.session_state.debug_show_paths:
    with st.expander("DEBUG: Session state - paths"):
        st.write(st.session_state.paths)

if st.session_state.debug_show_flags:
    with st.expander("DEBUG: Session state - flags"):
        st.write(st.session_state.flags)
