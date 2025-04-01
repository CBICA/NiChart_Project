import os

import streamlit as st
import utils.utils_menu as utilmenu
import utils.utils_pages as utilpg
import utils.utils_session as utilss

# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()

st.markdown(
    """
    # White Matter Lesion Segmentation
    ### Neuroimaging pipeline for segmentation of white matter lesions on FL scans
    """
)

st.divider()

st.markdown(
    """
    - [DLWMLS](https://neuroimagingchart.com/components/#Image%20Processing): Rapid and accurate segmentation of ***white matter lesions**
    """
)
f_img = os.path.join(
    st.session_state.paths["root"], "resources", "images", "dlwmls_segmask.png"
)
st.image(f_img, width=300)

st.divider()


st.markdown(
    """
    """
)
