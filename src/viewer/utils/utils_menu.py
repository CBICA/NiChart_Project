from typing import Any

import streamlit as st


def menu() -> Any:
    if st.session_state.pipeline == "Home":
        st.sidebar.page_link("pages/home.py", label="Home")

    if st.session_state.pipeline == "sMRI Biomarkers (T1)":
        st.sidebar.page_link("pages/home.py", label="Home")
        st.sidebar.page_link(
            "pages/pipeline_dlmuse.py", label=":material/arrow_forward: Pipeline Overview"
        )
        st.sidebar.page_link(
            "pages/prep_sMRI_dicomtonifti.py",
            label=":material/arrow_forward: Dicom to Nifti",
        )
        st.sidebar.page_link(
            "pages/process_sMRI_DLMUSE.py",
            label=":material/arrow_forward: DLMUSE Segmentation",
        )
        st.sidebar.page_link(
            "pages/workflow_sMRI_MLScores.py",
            label=":material/arrow_forward: Machine Learning Biomarkers",
        )
        st.sidebar.page_link(
            "pages/plot_sMRI_vars_study.py", label=":material/arrow_forward: View Data"
        )

    if st.session_state.pipeline == "WM Lesion Segmentation (FL)":
        st.sidebar.page_link("pages/home.py", label="Home")
        st.sidebar.page_link(
            "pages/pipeline_dlwmls.py", label=":material/arrow_forward: Pipeline Overview"
        )
        st.sidebar.page_link(
            "pages/prep_sMRI_dicomtonifti.py",
            label=":material/arrow_forward: Dicom to Nifti",
        )
        st.sidebar.page_link(
            "pages/process_sMRI_DLWMLS.py", label=":material/arrow_forward: DLWMLS"
        )

    if st.session_state.pipeline == "DTI Biomarkers (DTI)":
        st.sidebar.page_link("pages/home.py", label="Home")
        st.sidebar.page_link(
            "pages/pipeline_dti.py", label=":material/arrow_forward: Pipeline Overview"
        )

    if st.session_state.pipeline == "Resting State fMRI Biomarkers (rsfMRI)":
        st.sidebar.page_link("pages/home.py", label="Home")
        st.sidebar.page_link(
            "pages/pipeline_fmri.py", label=":material/arrow_forward: Pipeline Overview"
        )


def menu_selection() -> Any:
    if "pipeline" not in st.session_state or st.session_state.pipeline is None:
        st.switch_page("pages/home.py")
    menu()
