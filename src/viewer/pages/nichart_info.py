
import utils.utils_pages as utilpg
import os
import streamlit as st
from utils.utils_styles import inject_global_css
from utils.utils_logger import setup_logger

logger = setup_logger()
logger.debug("--- STARTING: Info ---")

inject_global_css()

utilpg.set_global_style()

st.set_page_config(page_title="NiChart", layout="wide")

imgdir = os.path.join(st.session_state.paths['resources'], 'images', 'nichart_logo')

def imgfile_to_data(filepath):
    import base64
    with open(filepath, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data)
    return "data:image/png;base64," + encoded.decode("utf-8")

tabA, tabB, tabC = st.tabs(
    ["Methods", "Data", "Quick Start"],
    on_change='rerun',
)

if tabA.open:
    with tabA:

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["NiChart", "MRI Segmentation", "AI Biomarkers", "Brain Aging Dimensions", "Abnormality Maps"],
            on_change='rerun',
        )

        if tab1.open:
            with tab1:
                st.markdown("<h5 style='color:#3a3a88;'>NiChart Overview</h5>", unsafe_allow_html=True)
                with st.container(horizontal=True, border=False):
                    st.image(imgfile_to_data(os.path.join(imgdir, 'nichart_logo_v2_img1_v2.png')))
                    st.markdown(
                        """
                        **NeuroImaging Chart of AI-based Imaging Biomarkers**

                        A framework to:

                        - Process MRI images
                        - Harmonize scans to reference datasets
                        - Apply and contribute machine learning models
                        - Derive individualized neuroimaging biomarkers
                        """
                    )

        if tab2.open:
            with tab2:
                st.markdown("<h5 style='color:#3a3a88;'>MRI Segmentation</h5>", unsafe_allow_html=True)
                with st.container(horizontal=True, border=False):
                    st.image(imgfile_to_data(os.path.join(imgdir, 'nichart_logo_v2_img4_v2.png')))
                    st.markdown(
                        """
                        **Segmentation of Brain Anatomy**

                        NiChart integrates DL-based models to calculate:

                        - **DLICV:** Intra-cranial volume estimation
                        - **DLMUSE:** Region of interest segmentation https://pubmed.ncbi.nlm.nih.gov/26679328
                        - **DLWMLS:** WM lesion segmentation https://pubmed.ncbi.nlm.nih.gov/26679328
                        """
                    )

        if tab3.open:
            with tab3:
                st.markdown("<h5 style='color:#3a3a88;'>AI Biomarkers</h5>", unsafe_allow_html=True)
                with st.container(horizontal=True, border=False):
                    st.image(imgfile_to_data(os.path.join(imgdir, 'nichart_logo_v2_img3_v2.png')))

                    st.markdown(
                        """
                        **Imaging biomarkers of brain aging and disease**

                        NiChart uses raw T1 images and/or derived features to compute a set of predictive biomarkers (SPARE scores - Spatial Patterns of Abnormalities reflect structural variability in the brain associated with a given task)

                        - **SPARE-BA:** An individualized index reflecting the brain age

                        - **DeepSPARE-BA:** An individualized index reflecting the brain age and derived directly from raw T1 scan

                        - **SPARE-AD:** An individualized index quantifying the presence and severity of Alzheimer's disease (AD)-like patterns of atrophy in the brain (https://pubmed.ncbi.nlm.nih.gov/19416949/)

                        - **SPARE-CVMs:** The cardiometabolic risk models (smoking, obesity, hypertension, and diabetes) https://www.nature.com/articles/s41467-025-57867-7

                        - Other SPARE disease models reflect the specific conditions for depression (**SPARE-Depression**) and psychosis (**SPARE-Psychosis**)
                        """
                    )

        if tab4.open:
            with tab4:
                st.markdown("<h5 style='color:#3a3a88;'>Brain Aging Dimensions</h5>", unsafe_allow_html=True)
                with st.container(horizontal=True, border=False):
                    st.image(imgfile_to_data(os.path.join(imgdir, 'nichart_logo_v2_img5_v2.png')))
                    st.markdown(
                        """
                        **Semi-supervised ML models of brain aging heterogeneity**

                        Brain aging dimensions reflect continuous latent representations of structural patterns associated with aging.

                        - **Surreal-GAN R-indices:** https://pubmed.ncbi.nlm.nih.gov/39147830/

                        - **CCLNMF indices:** Coupled Cross-Sectional and Longitudinal Non-Negative Matrix Factorization 

                        **Note:** Surreal-GAN and CCL-NMF indices in NiChart were obtained using a knowledge distillation method to train a tabular transformer with four encoder layers to predict the original indices
                        """
                    )

        if tab5.open:
            with tab5:
                st.markdown("<h5 style='color:#3a3a88;'>Abnormality Maps</h5>", unsafe_allow_html=True)
                with st.container(horizontal=True, border=False):
                    st.image(imgfile_to_data(os.path.join(imgdir, 'nichart_logo_v2_img6_v2.png')))
                    st.markdown(
                        """
                        **Voxelwise CSF abnormality maps to quantify regional brain atrophy***

                        Voxelwise abnormality maps quantify how much each brain region deviates from a normative aging model, highlighting localized tissue loss or expansion

                        - Abnormality maps were derived using mass-preserving tissue density measures (**RAVENS maps**), enabling precise regional comparisons of gray matter, white matter, and CSF volumes.

                        - Combining RAVENS with CSF-based abnormality maps yields a spatial fingerprint of structural vulnerability, showing where tissue density differs from healthy controls at the voxel level.

                        - These maps allow subject-level interpretation, enabling visualization of individual neuroanatomical abnormalities, not just group averages.
                        """
                    )

# Home button
utilpg.navig_home()
