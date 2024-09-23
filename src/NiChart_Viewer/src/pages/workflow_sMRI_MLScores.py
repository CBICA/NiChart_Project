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

def browse_file(path_input):
    '''
    File selector
    Returns the file name selected by the user and the parent folder
    '''
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    out_path = filedialog.askopenfilename(initialdir = path_input)
    path_out = os.path.dirname(out_path)
    root.destroy()
    return out_path, path_out

def browse_folder(path_input):
    '''
    Folder selector
    Returns the folder name selected by the user
    '''
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    out_path = filedialog.askdirectory(initialdir = path_input)
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
    tmpcol = st.columns((1,8))
    with tmpcol[0]:
        dset_name = st.text_input("Dataset name", value = st.session_state.study_name,
                                  help = "Each dataset's results are organized in a dedicated folder named after the dataset")
        st.session_state.study_name = dset_name

    # ROI file name
    tmpcol = st.columns((8,1))
    with tmpcol[1]:
        if st.button("Select the ROI file"):
            st.session_state.path_csv_dlmuse, st.session_state.path_last_sel = browse_file(st.session_state.path_last_sel)
    with tmpcol[0]:
        csv_dlmuse = st.text_input("DLMUSE csv file", value = st.session_state.path_csv_dlmuse,
                                  help = 'Input csv file with DLMUSE ROI volumes.\n\nChoose the file by typing it into the text field or using the file browser to browse and select it')
        if os.path.exists(csv_dlmuse):
            st.session_state.path_csv_dlmuse = csv_dlmuse

    # Demog file name
    tmpcol = st.columns((8,1))
    with tmpcol[1]:
        if st.button("Select the demographics file"):
            st.session_state.path_csv_demog, st.session_state.path_last_sel = browse_file(st.session_state.path_last_sel)
    with tmpcol[0]:
        csv_demog = st.text_input("Demog csv file", value = st.session_state.path_csv_demog,
                                  help = 'Input csv file with demographic values.\n\nChoose the file by typing it into the text field or using the file browser to browse and select it')
        if os.path.exists(csv_dlmuse):
            st.session_state.path_csv_demog = csv_demog

    # Out folder name
    tmpcol = st.columns((8,1))
    with tmpcol[1]:
        if st.button("Select the output folder", help = 'Choose the path by typing it into the text field or using the file browser to browse and select it'):
            st.session_state.path_out = browse_folder(st.session_state.path_last_sel)
    with tmpcol[0]:
        dir_out = st.text_input("Output folder",
                                value = st.session_state.path_out,
                                help = 'Results will be saved into the output folder, in a subfolder named "MLScores".\n\nThe results will include harmonized ROIs, SPARE scores and SurrealGAN scores.\n\nChoose the path by typing it into the text field or using the file browser to browse and select it')
        if os.path.exists(dir_out):
            st.session_state.path_out = dir_out

    # Check input files
    flag_files = 1
    if not os.path.exists(csv_dlmuse):
        flag_files = 0

    if not os.path.exists(csv_demog):
        flag_files = 0

    run_dir = os.path.join(st.session_state.path_root, 'src', 'workflow', 'workflows', 'w_sMRI')

    # Run workflow
    if flag_files == 1:
        if st.button("Run w_sMRI"):

            import time
            dir_out_mlscores = os.path.join(dir_out, dset_name, 'MLScores')
            st.info(f"Running: mlscores_workflow ", icon = ":material/manufacturing:")
            with st.spinner('Wait for it...'):
                time.sleep(15)
                os.system(f"cd {run_dir}")
                cmd = f"python3 {run_dir}/call_snakefile.py --run_dir {run_dir} --dset_name {dset_name} --input_rois {csv_dlmuse} --input_demog {csv_demog} --dir_out {dir_out_mlscores}"
                os.system(cmd)
                st.success("Run completed!", icon = ":material/thumb_up:")

                # Set the output file as the input for the related viewers
                csv_mlscores = f"{dir_out_mlscores}/{dset_name}_DLMUSE+MLScores.csv"
                if os.path.exists(csv_mlscores):
                    st.session_state.path_csv_mlscores = csv_mlscores

                st.success(f"Out file: {csv_mlscores}")


# FIXME: this is for debugging; will be removed
with st.expander('session_state: Plots'):
    st.session_state.plots

with st.expander('session_state: All'):
    st.write(st.session_state)

