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
    dlmuse_csv = st.text_input("path to DLMUSE csv file")
    demog_csv = st.text_input("path to demographic csv file")
    output_folder = st.text_input("path to output folder")

    if not os.path.exists(dlmuse_csv):
        st.warning("Path to input DLMUSE csv doesn't exist")
    if not os.path.exists(dlmuse_csv):
        st.warning("Path to input demographic csv doesn't exist")
    if not os.path.exists(output_folder):
        st.warning("Path to output folder doesn't exist")

    if st.button("Run w_sMRI"):
        st.write("Pipeline is running, please wait!")
        os.system("cd ../../NiCHart_Project/src/workflow")
        os.system(f"cd ../../NiChart_Project && python3 run.py --input_dlmuse {dlmuse_csv} --input_demog {demog_csv} --dir_output {output_folder} --conda 1")
        st.write("Run completed!")
