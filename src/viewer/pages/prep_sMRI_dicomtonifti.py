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
from stqdm import stqdm

# Page config should be called for each page
utilss.config_page()

result_holder = st.empty()

utilmenu.menu()

st.write("# Dicom to Nifti Conversion")

def progress(p: int, i: int, decoded: Any) -> None:
    with result_holder.container():
        st.progress(p, f"Progress: Token position={i}")

def panel_wdir() -> None:
    """
    Panel for selecting output dir
    """
    with st.container(border=True):
        curr_dir = st.session_state.paths["out_dir"]
        sel_dir = utilst.util_select_dir(curr_dir, 'sel_out_dir')
        if sel_dir is not None and sel_dir != curr_dir:
            st.session_state.paths["outdir"] = sel_dir
            
    with st.container(border=True):
        if os.path.exists(st.session_state.paths["out_dir"]):
            st.success(
                f"Output directory: {st.session_state.paths['out_dir']}",
                icon=":material/thumb_up:",
            )
            st.session_state.flags["out_dir"] = True

def panel_indicoms() -> None:
    """
    Panel for selecting input dicoms
    """
    msg = st.session_state.app_config[st.session_state.app_type]["msg_infile"]
    icon = st.session_state.icon_thumb[st.session_state.flags["dir_dicom"]]
    st.checkbox(
        f":material/upload: {msg} Dicoms {icon}",
        disabled=not st.session_state.flags["out_dir"],
        key="_check_dicoms_in",
        value=st.session_state.checkbox["dicoms_in"],
    )
    if not st.session_state._check_dicoms_in:
        return

    with st.container(border=True):
        if st.session_state.app_type == "cloud":
            # Upload data
            utilst.util_upload_folder(
                st.session_state.paths["dicom"],
                "Dicom files or folders",
                False,
                "Raw dicom files can be uploaded as a folder, multiple files, or a single zip file",
            )
            fcount = utilio.get_file_count(st.session_state.paths["dicom"])
            if fcount > 0:
                st.session_state.flags["dir_dicom"] = True
                p_dicom = st.session_state.paths["dicom"]
                st.success(
                    f"Data is ready ({p_dicom}, {fcount} files)",
                    icon=":material/thumb_up:",
                )

        else:  # st.session_state.app_type == 'desktop'
            utilst.util_select_folder(
                "selected_dicom_folder",
                "Dicom folder",
                st.session_state.paths["dicom"],
                st.session_state.paths["file_search_dir"],
                False,
            )
            fcount = utilio.get_file_count(st.session_state.paths["dicom"])
            if fcount > 0:
                st.session_state.flags["dir_dicom"] = True
                p_dicom = st.session_state.paths["dicom"]
                st.success(
                    f"Data is ready ({p_dicom}, {fcount} files)",
                    icon=":material/thumb_up:",
                )

        s_title = "DICOM Data"
        s_text = """
        - Upload or select the input DICOM folder containing all DICOM files. Nested folders are supported.

        - On the desktop app, a symbolic link named **"Dicoms"** will be created in the **working directory**, pointing to your input DICOM folder.

        - On the cloud platform, you can directly drag and drop your DICOM files or folders and they will be uploaded to the **"Dicoms"** folder within the **working directory**.

        - On the cloud, **we strongly recommend** compressing your DICOM data into a single ZIP archive before uploading. The system will automatically extract the contents of the ZIP file into the **"Dicoms"** folder upon upload.
        """
        utilst.util_get_help(s_title, s_text)

def panel_detect() -> None:
    """
    Panel for detecting dicom series
    """
    icon = st.session_state.icon_thumb[st.session_state.flags["dicom_series"]]
    st.checkbox(
        f":material/manage_search: Detect Dicom Series {icon}",
        disabled=not st.session_state.flags["dir_dicom"],
        key="_check_dicoms_series",
        value=st.session_state.checkbox["dicoms_series"],
    )
    if not st.session_state._check_dicoms_series:
        return

    with st.container(border=True):
        flag_disabled = not st.session_state.flags["dir_dicom"]

        # Detect dicom series
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
            st.session_state.flags["dicom_series"] = True
            st.success(
                f"Detected {st.session_state.num_dicom_scans} scans in {len(st.session_state.list_series)} series!",
                icon=":material/thumb_up:",
            )

        with st.expander("Show dicom metadata", expanded=False):
            st.dataframe(st.session_state.df_dicoms)

        s_title = "DICOM Series"
        s_text = """
        - The system verifies all files within the DICOM folder.
        - Valid DICOM files are processed to extract the DICOM header information, which is used to identify and group images into their respective series
        - The DICOM field **"SeriesDesc"** is used to identify series
        """
        utilst.util_get_help(s_title, s_text)


def panel_extract() -> None:
    """
    Panel for extracting dicoms
    """
    icon = st.session_state.icon_thumb[st.session_state.flags["dir_nifti"]]
    st.checkbox(
        f":material/auto_awesome_motion: Extract Scans {icon}",
        disabled=not st.session_state.flags["dicom_series"],
        key="_check_dicoms_run",
        value=st.session_state.checkbox["dicoms_run"],
    )
    if not st.session_state._check_dicoms_run:
        return

    with st.container(border=True):

        flag_disabled = not st.session_state.flags["dicom_series"]

        # Selection of img modality
        sel_mod = utilst.user_input_select(
            "Image Modality",
            "key_select_modality",
            st.session_state.list_mods,
            None,
            "Modality of the extracted images",
            flag_disabled,
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
            "",
            flag_disabled=flag_disabled,
        )

        btn_convert = st.button("Convert Series", disabled=flag_disabled)
        if btn_convert:
            with st.spinner("Wait for it..."):
                try:
                    utildcm.convert_sel_series(
                        st.session_state.df_dicoms,
                        st.session_state.sel_series,
                        st.session_state.paths[st.session_state.sel_mod],
                        f"_{st.session_state.sel_mod}.nii.gz",
                    )
                except:
                    st.warning(":material/thumb_down: Nifti conversion failed!")

            num_nifti = utilio.get_file_count(
                st.session_state.paths[st.session_state.sel_mod], ".nii.gz"
            )
            if num_nifti == 0:
                st.warning(
                    ":material/thumb_down: The extraction process did not produce any Nifti images!"
                )
            else:
                if st.session_state.has_cloud_session:
                    utilcloud.update_stats_db(
                        st.session_state.cloud_user_id, "NIFTIfromDICOM", num_nifti
                    )

        df_files = utilio.get_file_names(
            st.session_state.paths[st.session_state.sel_mod], ".nii.gz"
        )
        num_nifti = df_files.shape[0]

        if num_nifti > 0:
            st.session_state.flags["dir_nifti"] = True
            st.session_state.flags[st.session_state.sel_mod] = True
            st.success(
                f"Nifti images are ready ({st.session_state.paths[st.session_state.sel_mod]}, {num_nifti} scan(s))",
                icon=":material/thumb_up:",
            )

            with st.expander("View NIFTI image list"):
                st.dataframe(df_files)

        s_title = "Nifti Conversion"
        s_text = """
        - The user specifies the desired modality and selects the associated series.
        - Selected series are converted into Nifti image format.
        - Nifti images are renamed with the following format: **{PatientID}_{StudyDate}_{modality}.nii.gz**
        """
        utilst.util_get_help(s_title, s_text)


def panel_view() -> None:
    """
    Panel for viewing extracted nifti images
    """
    st.checkbox(
        ":material/visibility: View Scans",
        disabled=not st.session_state.flags["dir_nifti"],
        key="_check_dicoms_view",
        value=st.session_state.checkbox["dicoms_view"],
    )
    if not st.session_state._check_dicoms_view:
        return

    with st.container(border=True):

        # Selection of img modality
        sel_mod = utilst.user_input_select(
            "Image Modality",
            "key_selbox_modality_viewer",
            st.session_state.list_mods,
            None,
            "Modality of the images to view",
            False,
        )

        list_nifti = []
        if sel_mod is None:
            return

        st.session_state.sel_mod = sel_mod
        list_nifti = utilio.get_file_list(
            st.session_state.paths[st.session_state.sel_mod], ".nii.gz"
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

        st.session_state.paths["sel_img"] = os.path.join(
            st.session_state.paths[st.session_state.sel_mod], sel_img
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

        if not os.path.exists(st.session_state.paths["sel_img"]):
            return

        with st.spinner("Wait for it..."):

            try:
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


def panel_download() -> None:
    """
    Panel for downloading extracted nifti images
    """
    st.checkbox(
        ":material/download: Download Scans",
        disabled=not st.session_state.flags["dir_nifti"],
        key="_check_dicoms_download",
        value=st.session_state.checkbox["dicoms_download"],
    )
    if not st.session_state._check_dicoms_download:
        return

    with st.container(border=True):
        # Selection of img modality
        sel_mod = utilst.user_input_select(
            "Image Modality",
            "key_selbox_modality_download",
            ["All"] + st.session_state.list_mods,
            None,
            "Modality of the images to download",
            False,
        )

        if sel_mod is not None:
            st.session_state.sel_mod = sel_mod
            if st.session_state.sel_mod == "All":
                st.session_state.sel_mod = "nifti"

        out_zip = bytes()
        if not os.path.exists(st.session_state.paths["download"]):
            os.makedirs(st.session_state.paths["download"])
        f_tmp = os.path.join(
            st.session_state.paths["download"], f"{st.session_state.sel_mod}"
        )
        out_zip = utilio.zip_folder(
            st.session_state.paths[st.session_state.sel_mod], f_tmp
        )

        if os.path.exists(st.session_state.paths[st.session_state.sel_mod]):
            st.download_button(
                "Download Extracted Scans",
                out_zip,
                file_name=f"{st.session_state.dset}_{st.session_state.sel_mod}.zip",
                disabled=False,
            )
        else:
            st.warning(":material/thumb_down: No images found for download!")


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
t1, t2, t3, t4, t5 =  st.tabs(
    ['Working Dir', 'Input Data', 'Detect Series', 'Extract Scans', 'View Scans']
)
if st.session_state.app_type == "cloud":
    t1, t2, t3, t4, t5, t6 =  st.tabs(
        ['Working Dir', 'Input Data', 'Detect Series', 'Extract Scans', 'View Scans', 'Download']
    )

with t1:
    panel_wdir()
with t2:
    panel_indicoms()
with t3:
    panel_detect()
with t4:
    panel_extract()
with t5:
    panel_view()
if st.session_state.app_type == "cloud":
    with t6:
        panel_download()

# FIXME: For DEBUG
utilst.add_debug_panel()
