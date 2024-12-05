import os

import streamlit as st
import utils.utils_menu as utilmenu
import utils.utils_session as utilss

# Page config should be called for each page
utilss.config_page()

utilmenu.menu()

st.markdown(
    """
    # Structural MRI Biomarkers
    ### Neuroimaging pipeline for computing AI biomarkers of disease and aging from T1-weighted MRI scans
    """
)

st.divider()

st.markdown(
    """
    - [DLMUSE](https://neuroimagingchart.com/components/#Image%20Processing): Rapid and accurate **brain anatomy segmentation**
    """
)
f_img = os.path.join(
    st.session_state.paths["root"], "resources", "images", "dlicv+dlmuse_segmask.png"
)
st.image(f_img, width=400)

st.divider()

st.markdown(
    """
    - [COMBAT](https://neuroimagingchart.com/components/#Harmonization): **Statistical data harmonization** to [reference data](https://neuroimagingchart.com/components/#Reference%20Dataset)
    """
)
f_img = os.path.join(
    st.session_state.paths["root"], "resources", "images", "combat_agetrend.png"
)
st.image(f_img, width=800)

st.divider()

st.markdown(
    """
    - [SPAREAD and SPAREAge indices](https://neuroimagingchart.com/components/#Machine%20Learning%20Models): AI biomarkers of **Alzheimer's Disease and Aging** related brain atrophy patterns
    """
)
f_img = os.path.join(
    st.session_state.paths["root"], "resources", "images", "sparead+age.png"
)
st.image(f_img, width=800, caption="Habes et. al. Alzheimer's & Dementia 2021")

st.divider()

st.markdown(
    """
    - [SPARECVR indices](https://alz-journals.onlinelibrary.wiley.com/doi/abs/10.1002/alz.067709): AI biomarkers of brain atrophy patterns associated with **Cardio-Vascular Risk Factors**
    """
)
f_img = os.path.join(
    st.session_state.paths["root"], "resources", "images", "sparecvr.png"
)
st.image(
    f_img, width=800, caption="Govindarajan, S.T., et. al., Nature Communications, 2024"
)

st.divider()

st.markdown(
    """
    - [SurrealGAN indices](https://www.nature.com/articles/d41586-024-02692-z): Data-driven phenotyping of brain aging, **5 Brain Aging Subtypes**
    """
)
f_img = os.path.join(st.session_state.paths["root"], "resources", "images", "sgan1.jpg")
st.image(f_img, width=800, caption="Zhijian Yang et. al. Nature Medicine 2023")
