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

    # Input text boxes
    # FIXME: The init values for the input fields are set here to run tests quickly
    #        they will be removed or replaced in the future
    dset_name = st.text_input("Give a name to your dataset", value = 'Study1')

    # input_rois = st.text_input("Enter the input rois file name:", key = 'input_rois')
    # upload_rois = st.file_uploader("Upload the file:", key = 'upload_rois')

    input_rois = st.text_input("path to DLMUSE csv file", value = '/home/gurayerus/GitHub/CBICA/NiChart_Project/test/test_input/test2_rois/Study1/Study1_DLMUSE.csv')
    input_demog = st.text_input("path to demographic csv file", value = '/home/gurayerus/GitHub/CBICA/NiChart_Project/test/test_input/test2_rois/Study1/Study1_Demog.csv')
    dir_output = st.text_input("path to output folder", value = '/home/gurayerus/GitHub/CBICA/NiChart_Project/test/test_output/test2_rois')

    # Check input files
    if not os.path.exists(input_rois):
        st.warning("Path to input DLMUSE csv doesn't exist")
    if not os.path.exists(input_rois):
        st.warning("Path to input demographic csv doesn't exist")
    if not os.path.exists(dir_output):
        st.warning("Path to output folder doesn't exist")

    run_dir='../../workflow/workflows/w_sMRI'

    # Run workflow
    if st.button("Run w_sMRI"):
        st.write("Pipeline is running, please wait!")
        os.system(f"cd {run_dir}")
        cmd = f"python3 {run_dir}/call_snakefile.py --run_dir {run_dir} --dset_name {dset_name} --input_rois {input_rois} --input_demog {input_demog} --dir_output {dir_output}"
        os.system(cmd)
        st.write("Run completed!")
