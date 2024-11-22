import streamlit as st
import utils.utils_menu as utilmenu
import utils.utils_session as utilss

# Page config should be called for each page
utilss.config_page()

utilmenu.menu()

st.write("# White Matter Lesion Segmentation")

st.markdown(
    """
    Neuroimaging pipeline for segmentation of white matter lesions on FL scans
    """
)
