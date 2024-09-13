import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import plotly.express as px
from math import ceil
import os
from tempfile import NamedTemporaryFile

import tkinter as tk
from tkinter import filedialog

def browse_file(init_dir):
    '''
    File selector
    Returns the file name selected by the user and the parent folder
    '''
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    out_path = filedialog.askopenfilename(initialdir = init_dir)
    out_dir = os.path.dirname(out_path)
    root.destroy()
    return out_path, out_dir

def browse_folder(init_dir):
    '''
    Folder selector
    Returns the folder name selected by the user
    '''
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    out_path = filedialog.askdirectory(initialdir = init_dir)
    root.destroy()
    return out_path

st.markdown(
        """
    NiChart sMRI ML pipeline using COMBAT harmonization, SPARE
    scores, and SurrealGAN indices.
    - Input data is a csv file with the DLMUSE ROI volumes and
    a csv file with demographic info (Age, Sex, DX, Site).

    ### Want to learn more?
    - Visit [SPARE GitHub](https://github.com/CBICA/spare_scores)
        """
)

with st.container(border=True):


    # Dataset name: Used to create a main folder for all outputs
    dset_name = st.text_input("Give a name to your dataset", value = st.session_state.study_name)
    st.session_state.study_name = dset_name

    # Roi file name
    tmpcol = st.columns((1,8))
    with tmpcol[0]:
        if st.button("Select ROI file"):
            st.session_state.in_csv_MUSE, st.session_state.init_dir = browse_file(st.session_state.init_dir)
    with tmpcol[1]:
        input_rois = st.text_input("Enter the name of the ROI csv file:",
                                   value = st.session_state.in_csv_MUSE,
                                   label_visibility="collapsed")

    # Demog file name
    tmpcol = st.columns((1,8))
    with tmpcol[0]:
        if st.button("Select demog file"):
            st.session_state.in_csv_Demog, st.session_state.init_dir = browse_file(st.session_state.init_dir)
    with tmpcol[1]:
        input_demog = st.text_input("Enter the name of the demog csv file:",
                                    value = st.session_state.in_csv_Demog,
                                    label_visibility="collapsed")

    # Out folder name
    tmpcol = st.columns((1,8))
    with tmpcol[0]:
        if st.button("Select output folder"):
            st.session_state.out_dir = browse_folder(st.session_state.dir_root)
    with tmpcol[1]:
        dir_output = st.text_input("Enter the name of the output folder:",
                                   value = st.session_state.out_dir,
                                   label_visibility="collapsed")

    # Check input files
    flag_files = 1
    if not os.path.exists(input_rois):
        st.warning("Path to input DLMUSE csv doesn't exist")
        flag_files = 0

    if not os.path.exists(input_demog):
        st.warning("Path to input demographic csv doesn't exist")
        flag_files = 0

    run_dir='../../workflow/workflows/w_sMRI'

    # Run workflow
    if flag_files == 1:
        if st.button("Run w_sMRI"):
            st.write("Pipeline is running, please wait!")
            os.system(f"cd {run_dir}")
            cmd = f"python3 {run_dir}/call_snakefile.py --run_dir {run_dir} --dset_name {dset_name} --input_rois {input_rois} --input_demog {input_demog} --dir_output {dir_output}"
            os.system(cmd)
            st.write("Run completed!")

    # Set the output file as the input for the related viewers
    out_csv = f"{dir_output}/out_combined/{dset_name}_All.csv"
    if os.path.exists(out_csv):
        st.session_state.in_csv_sMRI = out_csv

# FIXME: this is for debugging; will be removed
with st.expander('session_state: Plots'):
    st.session_state.plots

with st.expander('session_state: All'):
    st.write(st.session_state)

