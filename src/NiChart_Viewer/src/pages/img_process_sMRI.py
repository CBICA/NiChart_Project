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
    NiChart sMRI segmentation pipeline using DLMUSE.
    - Please create an input folder with your images to run
    DLMUSE on them.
    - DLMUSE segments a raw T1 image into 145 regions of
    interest.
    - DLMUSE image variables are used as input in subsequent
    NiChart data analytics steps.

    ### Want to learn more?
    - Visit [DLMUSE GitHub](https://github.com/CBICA/NiChart_DLMUSE)
        """
)

with st.container(border=True):
    input_folder = st.text_input("path to input folder")
    output_folder = st.text_input("path to output folder")
    studies = st.text_input("total studies")
    cores = st.text_input("total cores")

    if not os.path.exists(input_folder):
        st.warning("Path to input folder doesn't exist")
    if not os.path.exists(output_folder):
        st.warning("Path to output folder doesn't exist")

    if st.button("Run w_DLMUSE"):
        st.write("Pipeline is running, please wait!")
        os.system("cd ../../NiCHart_Project/src/workflow")
        os.system(f"cd ../../NiChart_Project && python3 run.py --dir_input {input_folder} --dir_output {output_folder} --studies {studies} --cores {cores} --conda 1")
        st.write("Run completed!")
