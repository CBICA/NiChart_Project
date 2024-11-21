import streamlit as st
import utils.utils_menu as utilmenu

utilmenu.menu()

st.write("# White Matter Lesion Segmentation")

st.markdown(
    """
    Neuroimaging pipeline for segmentation of white matter lesions on FL scans
    """
)
