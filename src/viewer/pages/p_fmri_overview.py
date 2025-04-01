import streamlit as st
import utils.utils_menu as utilmenu
import utils.utils_session as utilss

# Page config should be called for each page
utilss.config_page()

utilmenu.menu()

st.write("# Resting State Functional MRI Biomarkers")

st.markdown(
    """
    :open_hands: Integration of rsfMRI processing pipelines [fMRIPrep](https://fmriprep.readthedocs.io), [XCP-D](https://xcp-d.readthedocs.io), [PNet](https://github.com/YuncongMa/pNet), and ML analytics for biomarker extraction is **Work in Progress**
    """
)
