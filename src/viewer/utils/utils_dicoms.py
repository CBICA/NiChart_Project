import gc
import os
import re
import traceback
import unicodedata
from typing import Any

import streamlit as st
import dicom2nifti.common as common
import dicom2nifti.convert_dicom as convert_dicom
import dicom2nifti.settings
import pandas as pd
from pydicom import dcmread
from pydicom.tag import Tag
from stqdm import stqdm

# Useful links
# https://github.com/rordenlab/dcm2niix/blob/master/FILENAMING.md
# https://stackoverflow.com/questions/71042522/conversion-not-working-properly-using-dicom2nifti
# https://pypi.org/project/dicom2nifti/#history
# https://pycad.medium.com/mvp-online-dicom-nifti-viewer-with-python-0da8b3aceadd
# https://github.com/angelomenezes/dicom-labeling-tool


# Adapted from dicom2nifti
def _is_valid_imaging_dicom(dicom_header: Any) -> bool:
    """
    Function will do some basic checks to see if this is a valid imaging dicom
    """
    # if it is philips and multiframe dicom then we assume it is ok
    try:
        if common.is_multiframe_dicom([dicom_header]):
            return True

        if "SeriesInstanceUID" not in dicom_header:
            return False

        if "InstanceNumber" not in dicom_header:
            return False

        if (
            "ImageOrientationPatient" not in dicom_header
            or len(dicom_header.ImageOrientationPatient) < 6
        ):
            return False

        if (
            "ImagePositionPatient" not in dicom_header
            or len(dicom_header.ImagePositionPatient) < 3
        ):
            return False

        # for all others if there is image position patient we assume it is ok
        if Tag(0x0020, 0x0037) not in dicom_header:
            return False

        return True
    except (KeyError, AttributeError):
        return False

# Adapted from dicom2nifti
def _remove_accents_(unicode_filename: str) -> str:
    """
    Function that will try to remove accents from a unicode string to be used in a filename.
    input filename should be either an ascii or unicode string
    """
    valid_characters = bytes(b"-_.() 1234567890abcdefghijklmnopqrstuvwxyz")
    cleaned_filename = unicodedata.normalize("NFKD", unicode_filename).encode(
        "ASCII", "ignore"
    )

    new_filename = ""

    for char_int in bytes(cleaned_filename):
        char_byte = bytes([char_int])
        if char_byte in valid_characters:
            new_filename += char_byte.decode()

    return new_filename

# Adapted from dicom2nifti
def detect_series(in_dir: str) -> Any:
    """
    This function selects dicom files that match the selection keywords
    Selection is done using the "SeriesDescription"
    """
    # Sort dicom files and detect series
    list_files = []
    for root, _, files in os.walk(in_dir):
        for dicom_file in files:
            list_files.append(os.path.join(root, dicom_file))
    list_dfields = []
    for _, file_path in stqdm(
        enumerate(list_files),
        desc="Detecting series in dicom files ...",
        total=len(list_files),
    ):
        # noinspection PyBroadException
        dfields = []
        try:
            if common.is_dicom_file(file_path):
                # read only dicom header without pixel data
                dicom_headers = dcmread(
                    file_path,
                    defer_size="1 KB",
                    stop_before_pixels=True,
                    force=dicom2nifti.settings.pydicom_read_force,
                )
                if _is_valid_imaging_dicom(dicom_headers):
                    dfields = [
                        file_path,
                        dicom_headers.PatientID,
                        dicom_headers.StudyDate,
                        dicom_headers.SeriesDescription,
                    ]
            list_dfields.append(dfields)

        # Explicitly capturing all errors here to be able to continue processing all the rest
        except:
            print(f"Unable to read: {file_path}")

    # Create dataframe with file name and dicom series description
    df_dicoms = pd.DataFrame(
        data=list_dfields, columns=["fname", "PatientID", "StudyDate", "SeriesDesc"]
    )

    return df_dicoms

def select_series(df_dicoms: pd.DataFrame, dict_series: pd.Series) -> Any:
    # Select dicom files for which series desc. contains user keywords
    df_sel_list = []
    dict_out = {}
    for key, value in dict_series.items():
        df_sel = df_dicoms[df_dicoms.SeriesDesc.str.contains(value)]
        dict_out[key] = df_sel.SeriesDesc.unique().tolist()
        if df_sel.shape[0] > 0:
            df_sel_list.append(df_sel)
            print(f" Detected dicoms: {key} , {df_sel.shape[0]}")
        else:
            print(f" WARNING: No matching dicoms for {key}")

    # Return selected files, series descriptions, and all series in the folder
    return df_sel_list, dict_out

def convert_single_series(
    list_files: list,
    out_dir: str,
    out_suff: str,
    compression: bool = True,
    reorient: bool = True,
) -> None:
    """
    This function will extract dicom files given in the list to nifti
    """
    # Sort dicom files by series uid
    dicom_series = {}  # type: ignore
    for file_path in list_files:
        try:
            dicom_headers = dcmread(
                file_path,
                defer_size="1 KB",
                stop_before_pixels=False,
                force=dicom2nifti.settings.pydicom_read_force,
            )
            if not _is_valid_imaging_dicom(dicom_headers):
                print(f"Skipping: {file_path}")
                continue
            print(f"Organizing: {file_path}")
            if dicom_headers.SeriesInstanceUID not in dicom_series:
                dicom_series[dicom_headers.SeriesInstanceUID] = []
            dicom_series[dicom_headers.SeriesInstanceUID].append(dicom_headers)
        except:  # Explicitly capturing all errors here to be able to continue processing all the rest
            print("Unable to read: %s" % file_path)

    # Start converting one by one
    for series_id, dicom_input in stqdm(
        dicom_series.items(), desc="    Converting scans...", total=len(dicom_series)
    ):
        base_filename = ""
        try:
            # construct the filename for the nifti
            base_filename = ""
            if "PatientID" in dicom_input[0]:
                base_filename = _remove_accents("%s" % dicom_input[0].PatientID)

            # FIXME: Check also "AcquisitionDate"
            if "StudyDate" in dicom_input[0]:
                base_filename = _remove_accents(
                    f"{base_filename}_{dicom_input[0].StudyDate}"
                )

            # if 'SeriesDescription' in dicom_input[0]:
            # base_filename = _remove_accents(f'{base_filename}_{dicom_input[0].SeriesDescription}')

            else:
                base_filename = _remove_accents(dicom_input[0].SeriesInstanceUID)

            print("--------------------------------------------")
            print(f"Start converting {base_filename}")
            if compression:
                nifti_file = os.path.join(out_dir, base_filename + out_suff)
            else:
                nifti_file = os.path.join(out_dir, base_filename + out_suff)
            convert_dicom.dicom_array_to_nifti(dicom_input, nifti_file, reorient)
            gc.collect()
        except:  # Explicitly capturing app exceptions here to be able to continue processing
            print(f"Unable to convert: {base_filename}")
            traceback.print_exc()


def convert_sel_series(
    df_dicoms: pd.DataFrame, sel_series: pd.Series, out_dir: str, out_suff: str
):
    # Convert all images for each selected series
    for _, stmp in stqdm(
        enumerate(sel_series), desc="Sorting series...", total=len(sel_series)
    ):
        print(f"Converting series: {stmp}")
        list_files = df_dicoms[df_dicoms.SeriesDesc == stmp].fname.tolist()
        print(list_files)

        convert_single_series(
            list_files, out_dir, out_suff, compression=True, reorient=True
        )


def panel_extract_dicoms() -> None:
    """
    Panel for extracting dicoms
    """
    sel_mod = "T1"

    dicom_folder = os.path.join(st.session_state.paths['project'], 'dicoms')
    out_folder = os.path.join(st.session_state.paths['project'], sel_mod.lower())
    
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

                    except:
                        st.warning(":material/thumb_down: NIfTI conversion failed!")

                time.sleep(1)
                st.rerun()


def panel_detect_dicom_series(in_dir) -> None:
    """
    Panel for detecting dicom series
    """
    # Detect dicom series
    btn_detect = st.button("Detect Series")
    if btn_detect:
        with st.spinner("Wait for it..."):
            df_dicoms = detect_series(in_dir)
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


