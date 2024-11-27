import streamlit as st
import utils.utils_menu as utilmenu
import utils.utils_session as utilss

# Page config should be called for each page
utilss.config_page()

utilmenu.menu()

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
st.image('/home/guraylab/Desktop/images/dlwmls_segmask.png', width=300)

st.divider()


st.markdown(
    """
    """
)
