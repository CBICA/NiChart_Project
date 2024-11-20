import streamlit as st
   
def menu():
    if st.session_state.pipeline == 'Home':
        st.sidebar.page_link("pages/home.py", label="Home")
        
    if st.session_state.pipeline == 'DLMUSE':
        st.sidebar.page_link(
            "pages/home.py",
            label="Home"
        )
        st.sidebar.page_link(
            "pages/prep_sMRI_dicomtonifti.py",
            label=":material/arrow_forward: Dicom to Nifti"
        )
        st.sidebar.page_link(
            "pages/process_sMRI_DLMUSE.py",
            label=":material/arrow_forward: DLMUSE"
        )
        st.sidebar.page_link(
            "pages/workflow_sMRI_MLScores.py",
            label=":material/arrow_forward: MLScores"
        )
        st.sidebar.page_link(
            "pages/plot_sMRI_vars_study.py",
            label=":material/arrow_forward: View"
        )
        
    if st.session_state.pipeline == 'DLWMLS':
        st.sidebar.page_link(
            "pages/home.py",
            label="Home"
        )
        st.sidebar.page_link(
            "pages/prep_sMRI_dicomtonifti.py",
            label=":material/arrow_forward: Dicom to Nifti"
        )
        st.sidebar.page_link(
            "pages/process_sMRI_DLWMLS.py",
            label=":material/arrow_forward: DLWMLS"
        )
        st.sidebar.page_link(
            "pages/plot_sMRI_vars_study.py",
            label=":material/arrow_forward: View"
        )

def menu_selection():
    if "pipeline" not in st.session_state or st.session_state.pipeline is None:
        st.switch_page("pages/home.py")
    menu()
