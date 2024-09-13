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

def browse_file_folder(is_file, init_dir):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    if is_file == True:
        out_path = filedialog.askopenfilename(initialdir = init_dir)
    else:
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

    # Get path to root folder
    dir_root = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))

    # Default file names
    # FIXME: The init values for the input fields are set here to run tests quickly
    #        - they will be replaced in the future with values used to link I/O
    #          between modules (i.e if DLMUSE pipeline was run, it's out file will be set as input'
    default_study_name = 'Study1'
    default_roi_name = f'{dir_root}/test/test_input/test2_rois/Study1/Study1_DLMUSE.csv'
    default_demog_name = f'{dir_root}/test/test_input/test2_rois/Study1/Study1_Demog.csv'
    default_out_name = f'{dir_root}/test/test_output/test2_rois/Study1'

    # Dset name: We use this to name all output for a study
    dset_name = st.text_input("Give a name to your dataset", value = default_study_name)

    # Roi file name (user can enter either using the file browser or type  full path)
    tmpcol = st.columns((1,8))
    fname_rois = default_roi_name
    with tmpcol[0]:
        if st.button("Select ROI file"):
            fname_rois = browse_file_folder(True, dir_root)
    with tmpcol[1]:
        input_rois = st.text_input("Enter the name of the ROI csv file:", value = fname_rois,
                                   label_visibility="collapsed")

    # Demog file name (user can enter either using the file browser or type  full path)
    tmpcol = st.columns((1,8))
    fname_demog = default_demog_name
    with tmpcol[0]:
        if st.button("Select demog file"):
            fname_demog = browse_file_folder(True, dir_root)
    with tmpcol[1]:
        input_demog = st.text_input("Enter the name of the demog csv file:", value = fname_demog,
                                   label_visibility="collapsed")

    # Out folder name (user can enter either using the file browser or type  full path)
    tmpcol = st.columns((1,8))
    dname_out = default_out_name
    with tmpcol[0]:
        if st.button("Select output folder"):
            dname_out = browse_file_folder(False, dir_root)
    with tmpcol[1]:
        dir_output = st.text_input("Enter the name of the output folder:", value = dname_out,
                                   label_visibility="collapsed")

    flag_files = 1

    # Check input files
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
