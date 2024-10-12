import os
from typing import Any

import streamlit as st
import utils.utils_dicom as utildcm
import utils.utils_io as utilio
import utils.utils_nifti as utilni
import utils.utils_st as utilst

result_holder = st.empty()


def progress(p: int, i: int, decoded: Any) -> None:
    with result_holder.container():
        st.progress(p, f"Progress: Token position={i}")

def save_and_unzip_files():
    # Save files to local storage
    if len(st.session_state['uploaded_dicoms']) > 0:
        utilio.save_uploaded_files(st.session_state['uploaded_dicoms'], st.session_state.paths["Dicoms"])

st.markdown(
    """
    - Enables users to extract raw DICOM files in the input directory to NIFTI format.
    - The application automatically identifies different imaging series within the folder.
    - Users can choose specific series they want to extract.
    - The extracted Nifti files are consistently named using dicom information ({participant_id}\\_{scan_date}\\_{modality}.nii.gz)
    - Extracted images can be viewed to review them visually.
    """
)

# Panel for output (dataset name + out_dir)
utilst.util_panel_workingdir(st.session_state.app_type)

# Panel for selecting input data
with st.expander("Select or upload input data", expanded=False):

    # Create Dicoms folder
    if os.path.exists(st.session_state.paths["dset"]):
        if not os.path.exists(st.session_state.paths["Dicoms"]):
            os.makedirs(st.session_state.paths["Dicoms"])

    if st.session_state.app_type == "CLOUD":

        # Create Dicoms folder
        flag_uploader = os.path.exists(st.session_state.paths["Dicoms"])

        # Upload dicom files
        in_files = st.file_uploader(
            "Upload dicom file(s)",
            key = 'uploaded_dicoms',
            accept_multiple_files=True,
            on_change = save_and_unzip_files
        )

    else:  # st.session_state.app_type == 'DESKTOP':

        # Input dicom image folder
        helpmsg = "Input folder with dicom files (.dcm).\n\nChoose the path by typing it into the text field or using the file browser to browse and select it"
        st.session_state.paths["Dicoms"] = utilst.user_input_folder(
            "Select folder",
            "btn_indir_dicom",
            "Input dicom folder",
            st.session_state.paths["last_sel"],
            st.session_state.paths["Dicoms"],
            helpmsg,
        )

# Panel for detecting dicom series
with st.expander("Detect dicom series", expanded=False):

    flag_btn = False
    if os.path.exists(st.session_state.paths["Dicoms"]):
        flag_btn = len(os.listdir(st.session_state.paths["Dicoms"])) > 0

    # Detect dicom series
    btn_detect = st.button("Detect Series", disabled=not flag_btn)
    if btn_detect:
        with st.spinner("Wait for it..."):
            df_dicoms = utildcm.detect_series(st.session_state.paths["Dicoms"])
            list_series = df_dicoms.SeriesDesc.unique()
            num_scans = (
                df_dicoms[["PatientID", "StudyDate", "SeriesDesc"]]
                .drop_duplicates()
                .shape[0]
            )
            if len(list_series) == 0:
                st.warning("Could not detect any dicom series!")
            else:
                st.success(
                    f"Detected {num_scans} scans in {len(list_series)} series!",
                    icon=":material/thumb_up:",
                )
            st.session_state.list_series = list_series
            st.session_state.df_dicoms = df_dicoms

# Panel for selecting and extracting dicom series
with st.expander("Select dicom series", expanded=False):

    # Selection of img modality
    helpmsg = "Modality of the extracted images"
    st.session_state.sel_mod = utilst.user_input_select(
        "Image Modality",
        st.session_state.list_mods,
        st.session_state.list_mods[0],
        helpmsg,
    )
    # Selection of dicom series
    st.session_state.sel_series = st.multiselect(
        "Select series:", st.session_state.list_series, []
    )
    # Create out folder for the selected modality
    if len(st.session_state.sel_series) > 0:
        if not os.path.exists(st.session_state.paths[st.session_state.sel_mod]):
            os.makedirs(st.session_state.paths[st.session_state.sel_mod])

    # Button for extraction
    flag_btn = (
        st.session_state.df_dicoms.shape[0] > 0 and len(st.session_state.sel_series) > 0
    )
    btn_convert = st.button("Convert Series", disabled=not flag_btn)
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
                st.warning("Could not extract any nifti images")
            else:
                st.success(
                    f"Extracted {len(st.session_state.list_input_nifti)} nifti images",
                    icon=":material/thumb_up:",
                )

# Panel for viewing extracted nifti images
with st.expander("View images", expanded=False):
    # Selection of MRID
    sel_img = st.selectbox(
        "Select Image",
        st.session_state.list_input_nifti,
        key="selbox_images",
        index=None,
    )

    if sel_img is not None:
        st.session_state.paths["sel_img"] = os.path.join(
            st.session_state.paths[st.session_state.sel_mod], sel_img
        )

    flag_btn = os.path.exists(st.session_state.paths["sel_img"])

    # Create a list of checkbox options
    list_orient = st.multiselect("Select viewing planes:", utilni.VIEWS, utilni.VIEWS)

    if flag_btn:

        with st.spinner("Wait for it..."):

            # Prepare final 3d matrix to display
            img = utilni.prep_image(st.session_state.paths["sel_img"])

            # Detect mask bounds and center in each view
            img_bounds = utilni.detect_img_bounds(img)

            # Show images
            blocks = st.columns(len(list_orient))
            for i, tmp_orient in enumerate(list_orient):
                with blocks[i]:
                    ind_view = utilni.VIEWS.index(tmp_orient)
                    utilst.show_img3D(
                        img, ind_view, img_bounds[ind_view, :], tmp_orient
                    )

# Panel for downloading results
if st.session_state.app_type == "CLOUD":
    with st.expander("Download Results", expanded=False):
        # Zip results
        flag_btn = os.path.exists(st.session_state.paths[st.session_state.sel_mod])

        out_zip = bytes()
        if flag_btn:
            if not os.path.exists(st.session_state.paths["OutZipped"]):
                os.makedirs(st.session_state.paths["OutZipped"])
            f_tmp = os.path.join(st.session_state.paths["OutZipped"], "T1.zip")
            out_zip = utilio.zip_folder(st.session_state.paths["T1"], f_tmp)

        st.download_button(
            "Download Extracted Scans",
            out_zip,
            file_name=f"{st.session_state.sel_mod}.zip",
            disabled=not flag_btn,
        )


with st.expander("TMP: session vars"):
    st.write(st.session_state)
with st.expander("TMP: session vars - paths"):
    st.write(st.session_state.paths)