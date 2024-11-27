import streamlit as st
import utils.utils_menu as utilmenu
import utils.utils_session as utilss

# Page config should be called for each page
utilss.config_page()

utilmenu.menu()

st.write("# Diffusion-Weighted MRI Biomarkers")

st.markdown(
    """
    :open_hands: Integration of DTI processing pipeline [QSIPrep](https://qsiprep.readthedocs.io) and ML analytics for biomarker extraction is **Work in Progress**
    """
)
