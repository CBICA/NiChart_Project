# -*- coding: utf-8 -*-
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from pathlib import Path

def dir_has_any_files_with_suffix(directory: Path, suffix: str):
    dir_path = Path(directory)
    return any(file for file in dir_path.glob(f"*{suffix}") if file.is_file())

def file_exists(filepath: Path):
    file_path = Path(filepath)
    return file_path.is_file()

def verify_data_dlmuse():
    flag_data = True
    t1_dir = Path(st.session_state.paths['project']) / "t1"
    demog_file = Path(st.session_state.paths['project']) / "participants" / "participants.csv"
    if not file_exists(demog_file):
        st.warning("Participants file missing.")
        flag_data = False
    if not dir_has_any_files_with_suffix(t1_dir, ".nii.gz"):
        st.error("No T1 images detected. Please upload data.")
        flag_data = False
    return flag_data

def verify_data_dlwmls():
    flag_data = True
    fl_dir = Path(st.session_state.paths['project']) / "fl"
    if not dir_has_any_files_with_suffix(fl_dir, ".nii.gz"):
        st.error("No FLAIR images detected. Please upload data.")
        flag_data = False
    return flag_data

def verify_data_cclnmf():
    flag_data = True
    t1_dir = Path(st.session_state.paths['project']) / "t1"
    if not dir_has_any_files_with_suffix(t1_dir, ".nii.gz"):
        st.error("No T1 images detected. Please upload data.")
        flag_data = False
    return flag_data

def verify_data_surrealgan():
    flag_data = True
    t1_dir = Path(st.session_state.paths['project']) / "t1"
    if not dir_has_any_files_with_suffix(t1_dir, ".nii.gz"):
        st.error("No T1 images detected. Please upload data.")
        flag_data = False
    return flag_data
def verify_data_spare():
    flag_data = True
    t1_dir = Path(st.session_state.paths['project']) / "t1"
    demog_file = Path(st.session_state.paths['project']) / "participants" / "participants.csv"
    if not file_exists(demog_file):
        st.warning("Participants file missing. Only needed for harmonized SPARE.")
        flag_data = False
    if not dir_has_any_files_with_suffix(t1_dir, ".nii.gz"):
        st.error("No T1 images detected. Please upload data.")
        flag_data = False
    return flag_data
def verify_data_dlspare():
    flag_data = True
    t1_dir = Path(st.session_state.paths['project']) / "t1"
    if not dir_has_any_files_with_suffix(t1_dir, ".nii.gz"):
        st.error("No T1 images detected. Please upload data.")
        flag_data = False
    return flag_data

def verify_data(method):
    flag_data = False
    if method == 'dlmuse':
        flag_data = verify_data_dlmuse()
    if method == 'dlwmls':
        flag_data = verify_data_dlwmls()
    if method == 'cclnmf':
        flag_data = verify_data_cclnmf()
    if method == 'surrealgan':
        flag_data = verify_data_surrealgan()
    if method in ['spare-ba', 'spare-ad', 'spare-depression', 'spare-obesity',
                   'spare-psychosis', 'spare-diabetes', 'spare-hypertension',
                   'spare-smoking']:
        flag_data = verify_data_spare()
    if method == 'spare-ba-image':
        flag_data = verify_data_dlspare()
    if method == 'dlmuse-dlwmls':
        flag_data = verify_data_dlmuse() and verify_data_dlwmls()
    if method == 'ravens':
        flag_data = verify_data_dlmuse()

    
    return flag_data
        
    









