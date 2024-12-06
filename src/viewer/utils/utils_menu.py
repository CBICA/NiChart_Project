from typing import Any

import streamlit as st
from streamlitextras.webutils import stxs_javascript
from typing import NoReturn

def redirect(url: str) -> NoReturn:
    stxs_javascript(f"window.location.replace('{url}');")

def menu() -> Any:
    ## Force redirect to the home page if anything is not properly instantiated.
    if 'instantiated' not in st.session_state:
        print("Redirected to home page as a required instantiation variable was missing.")
        st.switch_page('pages/home.py')
    if st.session_state.has_cloud_session:
        email = st.session_state.cloud_session_token['email']
        logout_url = "https://cbica-nichart.auth.us-east-1.amazoncognito.com/logout?client_id=4shr6mm2h0p0i4o9uleqpu33fj&response_type=code&scope=email+openid+phone&redirect_uri=https://cbica-nichart-alb-272274500.us-east-1.elb.amazonaws.com/oauth2/idpresponse"
        st.sidebar.info(f"Logged in as {email}.")
        ## TODO: Make this button also delete user data automatically
        st.sidebar.button("Log out", on_click=redirect, args=(logout_url,))
    if st.session_state.pipeline == "Home":
        st.sidebar.page_link("pages/home.py", label="Home")

    if st.session_state.pipeline == "sMRI Biomarkers (T1)":
        st.sidebar.page_link("pages/home.py", label="Home")
        st.sidebar.page_link(
            "pages/pipeline_dlmuse.py",
            label=":material/arrow_forward: Pipeline Overview",
        )
        st.sidebar.page_link(
            "pages/tutorial_dlmuse.py",
            label=":material/arrow_forward: Tutorial",
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
            "pages/pipeline_dlwmls.py",
            label=":material/arrow_forward: Pipeline Overview",
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
