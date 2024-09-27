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

import utils_st as utilst

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

    # Dataset name: All results will be saved in a main folder named by the dataset name 
    helpmsg = "Each dataset's results are organized in a dedicated folder named after the dataset"
    dset_name = utilst.user_input_text("Dataset name", 
                                        st.session_state.dset_name, 
                                        helpmsg)
    st.session_state.dset_name = dset_name

    # DLMUSE file name
    helpmsg = 'Input csv file with DLMUSE ROI volumes.\n\nChoose the file by typing it into the text field or using the file browser to browse and select it'
    csv_dlmuse, csv_path = utilst.user_input_file("Select file",
                                                  'btn_input_dlmuse',
                                                  "DLMUSE ROI file",
                                                  st.session_state.path_last_sel,
                                                  st.session_state.path_csv_dlmuse,
                                                  helpmsg)
    if os.path.exists(csv_dlmuse):
        st.session_state.path_csv_dlmuse = csv_dlmuse
        st.session_state.path_last_sel = csv_path

    # Demog file name
    helpmsg = 'Input csv file with demographic values.\n\nChoose the file by typing it into the text field or using the file browser to browse and select it'
    csv_demog, csv_path = utilst.user_input_file("Select file",
                                                  'btn_input_demog',
                                                  "Demographics file",
                                                  st.session_state.path_last_sel,
                                                  st.session_state.path_csv_demog,
                                                  helpmsg)
    if os.path.exists(csv_demog):
        st.session_state.path_csv_demog = csv_demog
        st.session_state.path_last_sel = csv_path

    # Out folder name
    helpmsg = 'Results will be saved into the output folder, in a subfolder named "MLScores".\n\nThe results will include harmonized ROIs, SPARE scores and SurrealGAN scores.\n\nChoose the path by typing it into the text field or using the file browser to browse and select it'
    path_out = utilst.user_input_folder("Select folder",
                                        'btn_out_dir',
                                        "Output folder",
                                        st.session_state.path_last_sel,
                                        st.session_state.path_out,
                                        helpmsg)
    st.session_state.path_out = path_out

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
            path_out_mlscores = os.path.join(path_out, dset_name, 'MLScores')
            st.info(f"Running: mlscores_workflow ", icon = ":material/manufacturing:")
            with st.spinner('Wait for it...'):
                time.sleep(15)
                os.system(f"cd {run_dir}")
                cmd = f"python3 {run_dir}/call_snakefile.py --run_dir {run_dir} --dset_name {dset_name} --input_rois {csv_dlmuse} --input_demog {csv_demog} --dir_out {path_out_mlscores}"
                os.system(cmd)
                st.success("Run completed!", icon = ":material/thumb_up:")

                # Set the output file as the input for the related viewers
                csv_mlscores = f"{path_out_mlscores}/{dset_name}_DLMUSE+MLScores.csv"
                if os.path.exists(csv_mlscores):
                    st.session_state.path_csv_mlscores = csv_mlscores

                st.success(f"Out file: {csv_mlscores}")


# FIXME: this is for debugging; will be removed
with st.expander('session_state: Plots'):
    st.session_state.plots

with st.expander('session_state: All'):
    st.write(st.session_state)
