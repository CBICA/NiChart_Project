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
import time
from pydicom import dcmread
from pydicom.tag import Tag
from stqdm import stqdm
import utils.utils_io as utilio
import utils.utils_cloud as utilcloud

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
def _remove_accents(unicode_filename: str) -> str:
    """
    Function that will try to remove accents from a unicode string to be used in a filename.
    input filename should be either an ascii or unicode string
    """
    unicode_filename = unicode_filename.replace(' ','_')
    valid_characters = bytes(
        b"-_.()1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )    
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
    '''
    This function selects dicom files that match the selection keywords
    Selection is done using the "SeriesDescription"
    '''
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
                        dicom_headers.PatientAge,
                        dicom_headers.PatientSex
                    ]
            list_dfields.append(dfields)

        # Explicitly capturing all errors here to be able to continue processing all the rest
        except:
            print(f"Unable to read: {file_path}")

    # Create dataframe with file name and dicom series description
    df_dicoms = pd.DataFrame(
        data=list_dfields,
        columns=["fname", "PatientID", "StudyDate", "SeriesDesc","Age", "Sex"]
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
                print(dicom_input[0].PatientID)
                print(base_filename)

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
            
            # Create demog info csv from dicoms
            page = None
            if "PatientAge" in dicom_input[0]:
                page = dicom_input[0].PatientAge
                page = page.replace('Y','')
            psex = None
            if "PatientSex" in dicom_input[0]:
                psex = dicom_input[0].PatientSex
            df_demog = pd.DataFrame({'MRID': [base_filename], 'Age': [page], 'Sex': [psex]})
            csv_file = os.path.join(
                out_dir, f'{base_filename}{out_suff.replace('.nii.gz', '').replace('.nii','')}.csv'
            )
            df_demog.to_csv(csv_file, index=False)
            
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

def panel_detect_dicom_series(in_dir) -> None:
    '''
    Panel for detecting dicom series
    '''
    # Detect dicom series
    if st.button("Detect Series"):
        with st.spinner("Detecting series ..."):
            df_dicoms = detect_series(in_dir)
            list_series = df_dicoms.SeriesDesc.unique()
            num_dicom_scans = (
                df_dicoms[["PatientID", "StudyDate", "SeriesDesc"]]
                .drop_duplicates()
                .shape[0]
            )
            st.session_state.dicoms['list_series'] = list_series
            st.session_state.dicoms['num_dicom_scans'] = num_dicom_scans
            st.session_state.dicoms['df_dicoms'] = df_dicoms

    if st.session_state.dicoms['list_series'] is None:
        return

    if len(st.session_state.dicoms['list_series']) == 0:
        return

    st.success(
        f"Detected {st.session_state.dicoms['num_dicom_scans']} scans in {len(st.session_state.dicoms['list_series'])} series!",
        icon=":material/thumb_up:",
    )

    with st.expander("Show dicom metadata", expanded=False):
        st.dataframe(st.session_state.dicoms['df_dicoms'])


def panel_extract_nifti(out_dir):
    """
    Panel for extracting dicoms
    """
    # Selection of img modality
    sel_mod = st.selectbox(
        "Image Modality",
        st.session_state.list_mods,
        key = "key_select_modality",
    )
    
    if sel_mod is not None:
        st.session_state.sel_mod = sel_mod
        dout = os.path.join(
            out_dir, sel_mod.lower()
        )
        if not os.path.exists(dout):
            os.makedirs(dout)

    # Selection of dicom series
    st.session_state.dicoms['sel_series'] = st.multiselect(
        "Select series:",
        st.session_state.dicoms['list_series'],
        key = "key_multiselect_dseries",
    )

    btn_convert = st.button("Convert Series")
    if btn_convert:
        with st.spinner("Wait for it..."):
            try:
                convert_sel_series(
                    st.session_state.dicoms['df_dicoms'],
                    st.session_state.dicoms['sel_series'],
                    dout,
                    f"_{st.session_state.sel_mod}.nii.gz",
                )
            except:
                st.warning(":material/thumb_down: Nifti conversion failed!")

        num_nifti = utilio.get_file_count(
            dout, ".nii.gz"
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
        dout, ".nii.gz"
    )
    num_nifti = df_files.shape[0]

    if num_nifti > 0:
        st.success(
            f"Nifti images are ready ({dout}, {num_nifti} scan(s))",
            icon=":material/thumb_up:",
        )

        with st.expander("View NIFTI image list"):
            st.dataframe(df_files)

