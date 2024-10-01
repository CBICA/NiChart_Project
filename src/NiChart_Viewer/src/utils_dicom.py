import gc
import logging
import os
import re
import traceback
import unicodedata
import time
import streamlit as st
from stqdm import stqdm

import pydicom
from pydicom import dcmread
from pydicom.tag import Tag

import dicom2nifti.common as common
import dicom2nifti.convert_dicom as convert_dicom
import dicom2nifti.settings

import pandas as pd

from glob import glob


## Useful links
# https://github.com/rordenlab/dcm2niix/blob/master/FILENAMING.md
# https://stackoverflow.com/questions/71042522/conversion-not-working-properly-using-dicom2nifti
# https://pypi.org/project/dicom2nifti/#history
# https://pycad.medium.com/mvp-online-dicom-nifti-viewer-with-python-0da8b3aceadd
# https://github.com/angelomenezes/dicom-labeling-tool

## Adapted from dicom2nifti
def _is_valid_imaging_dicom(dicom_header):
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

        if "ImageOrientationPatient" not in dicom_header or len(dicom_header.ImageOrientationPatient) < 6:
            return False

        if "ImagePositionPatient" not in dicom_header or len(dicom_header.ImagePositionPatient) < 3:
            return False

        # for all others if there is image position patient we assume it is ok
        if Tag(0x0020, 0x0037) not in dicom_header:
            return False

        return True
    except (KeyError, AttributeError):
        return False

## Adapted from dicom2nifti
def _remove_accents(unicode_filename):
    """
    Function that will try to remove accents from a unicode string to be used in a filename.
    input filename should be either an ascii or unicode string
    """
    # noinspection PyBroadException
    try:
        unicode_filename = unicode_filename.replace(" ", "_")
        cleaned_filename = unicodedata.normalize('NFKD', unicode_filename).encode('ASCII', 'ignore').decode('ASCII')

        cleaned_filename = re.sub(r'[^\w\s-]', '', cleaned_filename.strip().lower())
        cleaned_filename = re.sub(r'[-\s]+', '-', cleaned_filename)

        return cleaned_filename
    except:
        traceback.print_exc()
        return unicode_filename

## Adapted from dicom2nifti
def _remove_accents_(unicode_filename):
    """
    Function that will try to remove accents from a unicode string to be used in a filename.
    input filename should be either an ascii or unicode string
    """
    valid_characters = bytes(b'-_.() 1234567890abcdefghijklmnopqrstuvwxyz')
    cleaned_filename = unicodedata.normalize('NFKD', unicode_filename).encode('ASCII', 'ignore')

    new_filename = ""

    for char_int in bytes(cleaned_filename):
        char_byte = bytes([char_int])
        if char_byte in valid_characters:
            new_filename += char_byte.decode()

    return new_filename

## Adapted from dicom2nifti
def detect_series(in_dir):
    """
    This function selects dicom files that match the selection keywords
    Selection is done using the "SeriesDescription"
    """

    # Sort dicom files and detect series
    list_files = []
    for root, _, files in os.walk(in_dir):
        for dicom_file in files:
            list_files.append(os.path.join(root, dicom_file))
    df_dicoms = pd.DataFrame(data = list_files, columns = ['fname'])
    df_dicoms = df_dicoms.reindex(columns = ['fname', 'dtype'])
    list_dtypes = []
    for (_, file_path) in stqdm(enumerate(df_dicoms.fname.tolist()), desc="Detecting series in dicom files ...", total=len(df_dicoms.fname.tolist())):
        # noinspection PyBroadException
        dtype = ''
        try:
            if common.is_dicom_file(file_path):
                # read only dicom header without pixel data
                dicom_headers = dcmread(file_path,
                                        defer_size="1 KB",
                                        stop_before_pixels=True,
                                        force=dicom2nifti.settings.pydicom_read_force)
                if _is_valid_imaging_dicom(dicom_headers):
                    dtype = dicom_headers.SeriesDescription

        # Explicitly capturing all errors here to be able to continue processing all the rest
        except:
            print(f"Unable to read: {file_path}")

        list_dtypes.append(dtype)

    # Create dataframe with file name and dicom series description
    df_dicoms['dtype'] = list_dtypes

    # Detect all unique series
    list_series = df_dicoms.dtype.unique()

    return df_dicoms, list_series

def select_series(df_dicoms, dict_series):

    # Select dicom files for which series desc. contains user keywords
    df_sel_list = []
    dict_out = {}
    for key, value in dict_series.items():
        df_sel = df_dicoms[df_dicoms.dtype.str.contains(value)]
        dict_out[key] = df_sel.dtype.unique().tolist()
        if df_sel.shape[0] > 0:
            df_sel_list.append(df_sel)
            print(f' Detected dicoms: {key} , {df_sel.shape[0]}')
        else:
            print(f' WARNING: No matching dicoms for {key}')

    # Return selected files, series descriptions, and all series in the folder
    return df_sel_list, dict_out


def convert_single_series(list_files, out_dir, out_suff, compression=True, reorient=True):
    """
    This function will extract dicom files given in the list to nifti
    """
    # Sort dicom files by series uid
    dicom_series = {}
    for file_path in list_files:
        try:
            dicom_headers = dcmread(file_path,
                                    defer_size = "1 KB",
                                    stop_before_pixels = False,
                                    force = dicom2nifti.settings.pydicom_read_force)
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
    for series_id, dicom_input in stqdm(dicom_series.items(), desc="    Converting scans...", total=len(dicom_series)):
        base_filename = ""
        try:
            # construct the filename for the nifti
            base_filename = ""
            if 'PatientID' in dicom_input[0]:
                base_filename = _remove_accents('%s' % dicom_input[0].PatientID)

            ## FIXME: Check also "AcquisitionDate"
            if 'StudyDate' in dicom_input[0]:
                base_filename = _remove_accents(f'{base_filename}_{dicom_input[0].StudyDate}')

            if 'SeriesDescription' in dicom_input[0]:
                base_filename = _remove_accents(f'{base_filename}_{dicom_input[0].SeriesDescription}')

            else:
                base_filename = _remove_accents(dicom_input[0].SeriesInstanceUID)

            print('--------------------------------------------')
            print(f'Start converting {base_filename}')
            if compression:
                nifti_file = os.path.join(out_dir, base_filename + out_suff)
            else:
                nifti_file = os.path.join(out_dir, base_filename + out_suff)
            convert_dicom.dicom_array_to_nifti(dicom_input, nifti_file, reorient)
            gc.collect()
        except:  # Explicitly capturing app exceptions here to be able to continue processing
            print(f'Unable to convert: {base_filename}')
            traceback.print_exc()


def convert_sel_series(df_dicoms, sel_series, out_dir, out_suff):
    # Convert all images for each selected series
    for (_, stmp) in stqdm(enumerate(sel_series), desc="Converting series...", total=len(sel_series)):
        print(f'Converting series: {stmp}')
        list_files = df_dicoms[df_dicoms.dtype == stmp].fname.tolist()
        print(list_files)

        convert_single_series(list_files, out_dir, out_suff, compression=True, reorient=True)
#def convert_dicoms_to_nifti(in_dir, out_dir):
    ## Detect files
    #filesandirs = glob(os.path.join(in_dir, '**', '*'), recursive=True)
    #files = [f for f in filesandirs if os.path.isfile(f)]
    ## Read dicom meta data
    #dicoms = [pydicom.dcmread(f, stop_before_pixels=True) for f in files]
