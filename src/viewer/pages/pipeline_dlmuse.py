import streamlit as st
import utils.utils_menu as utilmenu

utilmenu.menu()

st.write("# Structural MRI Biomarkers")

st.markdown(
    """
    Neuroimaging pipeline for deriving machine learning (ML)-based biomarkers of disease and aging using imaging features extracted from T1-weighted image segmentation. The pipeline includes:
    - Segmentation of T1-weighted scans using DLMUSE (a DL based segmentation method)
    - COMBAT harmonization to reference data
    - SPARE scores for AD and Aging indices of brain atrophy patterns
    - SPARE scores for cardio-vascular disease (CVD)-related indices of brain atrophy patterns
    - SurrealGAN scores for image based subtype scores of brain aging
    """
)
