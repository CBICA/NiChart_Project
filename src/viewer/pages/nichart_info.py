
import utils.utils_pages as utilpg
# Page config should be called for each page
#utilpg.config_page()

import os
import numpy as np
import pandas as pd
import streamlit as st
import utils.utils_misc as utilmisc
import utils.utils_plots as utilpl
import utils.utils_session as utilses
import utils.utils_mriview as utilmri
import utils.utils_alerts as utils_alerts
import utils.utils_survey as utils_survey
from streamlit_image_select import image_select
import logging
from stqdm import stqdm
from utils.utils_logger import setup_logger
import gui.utils_navig as utilnav

import streamlit_antd_components as sac
import streamlit as st
from streamlit_card import card

#utilpg.config_page()
utilpg.set_global_style()

st.set_page_config(page_title="NiChart", layout="wide")

def imgfile_to_data(filepath):
    import base64

    with open(filepath, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data)
    data = "data:image/png;base64," + encoded.decode("utf-8")    
    return data
    

imgdir =  os.path.join(st.session_state.paths['resources'], 'images', 'nichart_logo')

def show_short_desc(title):
    if title == "NiChart":
        st.markdown("Neuroimaging Chart of **AI-based** imaging biomarkers")

        pass
    if title == "MRI Segmentation":
        st.markdown("Fast **deep-learning** segmentation of **healthy** and **pathological** anatomy")
        pass
    if title == "AI Biomarkers":
        st.markdown("AI-based **imaging biomarkers** quantifying **brain aging** and **neurodegeneration**")
        pass
    if title == "Brain Aging Dimensions":
        st.markdown("**Data-driven** brain aging indices capturing **distinct patterns** of **aging-related atrophy**")
        pass
    if title == "Abnormality Maps":
        st.markdown("Voxelwise CSF abnormality maps quantifying regional brain atrophy.")
        pass

def show_full_desc(title):
    if title == "NiChart":
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
    if title == "MRI Segmentation":
        st.markdown(
            """
            **Segmentation of Brain Anatomy**
            
            NiChart integrates DL-based models to calculate:
            
            - **DLICV:** Intra-cranial volume estimation 
            - **DLMUSE:** Region of interest segmentation https://pubmed.ncbi.nlm.nih.gov/26679328
            - **DLWMLS:** WM lesion segmentation https://pubmed.ncbi.nlm.nih.gov/26679328
            """
        )
            
    if title == "AI Biomarkers":
        st.markdown(
            """
            **Supervised ML models of brain aging and disease**
            
            NiChart uses raw T1 images and/or derived features to compute a set of predictive biomarkers (SPARE scores - Spatial Patterns of Abnormalities reflect structural variability in the brain associated with a given task)
            
            - **SPARE-BA:** An individualized index reflecting the brain age
            
            - **DeepSPARE-BA:** An individualized index reflecting the brain age and derived directly from raw T1 scan

            - **SPARE-AD:** An individualized index quantifying the presence and severity of Alzheimer’s disease (AD)-like patterns of atrophy in the brain (https://pubmed.ncbi.nlm.nih.gov/19416949/
            
            - **SPARE-CVMs:** The cardiometabolic risk models (smoking, obesity, hypertension, and diabetes) https://www.nature.com/articles/s41467-025-57867-7

            - Other SPARE disease models reflect the specific conditions for depression (**SPARE-Depression**) and psychosis (**SPARE-Psychosis**)
            """
        )

    if title == "Brain Aging Dimensions":
        st.markdown(
            """
            **Semi-supervised ML models of brain aging heterogeneity**
            
            Brain aging dimensions reflect continuous latent representations of structural patterns associated with aging. 
            
            - **Surreal-GAN R-indices:** The R-index reflects the severity of individualized brain changes along multiple dimensions, potentially reflecting the stage of a mixture of underlying neuropathological and biological processes https://pubmed.ncbi.nlm.nih.gov/39147830/
            
                ***R1:*** subcortical atrophy, mainly concentrated in the caudate and putamen
            
                ***R2:*** focal medial temporal lobe (MTL) atrophy
            
                ***R3:*** parieto-temporal atrophy, including that in middle temporal gyrus, angular gyrus and middle occipital gyrus
            
                ***R4:*** diffuse cortical atrophy in medial and lateral frontal regions, as well as superior parietal and occipital regions
            
                ***R5:*** perisylvian atrophy centered around the insular cortex
            
            - **CCLNMF indices:** Coupled Cross-Sectional and Longitudinal Non-Negative Matrix Factorization identifies dominant patterns of brain aging by jointly modeling baseline (cross-sectional) and follow-up (longitudinal) MRI data.
            
                Cross-sectional maps capture cumulative aging differences across individuals, while longitudinal maps quantify subject-specific change rates. These maps are jointly decomposed via NMF into shared spatial components and individual loadings, providing continuous measures of distinct aging patterns.
            
            **Note:** Surreal-GAN and CCL-NMF indices in NiChart were obtained using a knowledge distillation method to train a tabular transformer with four encoder layers to predict the original indices


            """
        )

        st.markdown('''

''')
    if title == "Abnormality Maps":
        st.markdown(
            """
            **CSF Abnormalities** ***(work in progress)***
            
            Voxelwise abnormality maps quantify how much each brain region deviates from a normative aging model, highlighting localized tissue loss or expansion
            
            - Abnormality maps were derived using mass-preserving tissue density measures (**RAVENS maps**), enabling precise regional comparisons of gray matter, white matter, and CSF volumes.
            
            - Combining RAVENS with CSF-based abnormality maps yields a spatial fingerprint of structural vulnerability, showing where tissue density differs from healthy controls at the voxel level.
            
            - These maps allow subject-level interpretation, enabling visualization of individual neuroanatomical abnormalities, not just group averages.
            """
        )

def card(title, image_path):
    with st.container(border=True, horizontal_alignment = 'left'):
        st.image(imgfile_to_data(image_path))
        with st.container(border=False, height = 200, horizontal_alignment = 'center'):
            st.markdown(f"#### {title}")
            show_short_desc(title)
        with st.popover("See More"):
            show_full_desc(title)

with st.container(horizontal_alignment="center"):
    st.markdown("## What can I do with NiChart?")
with st.container(horizontal=True, horizontal_alignment="center"):
    cols = st.columns(5)

    with cols[0]: # NiChart
        card(title="NiChart",
             image_path=os.path.join(imgdir, 'nichart_logo_v2_img1_v2.png')
            )

    with cols[1]: # Segmentation
        card(title="MRI Segmentation",
             image_path=os.path.join(imgdir, 'nichart_logo_v2_img4_v2.png')
             )

    with cols[2]: # AI Biomarkers
        card(title="AI Biomarkers",
             image_path=os.path.join(imgdir, 'nichart_logo_v2_img3_v2.png')
             )

    with cols[3]: # Brain Aging Subtypes
        card(title="Brain Aging Dimensions",
             image_path=os.path.join(imgdir, 'nichart_logo_v2_img5_v2.png')
             )

    with cols[4]: # Voxelwise Abnormality Maps
        card(title="Abnormality Maps",
             image_path=os.path.join(imgdir, 'nichart_logo_v2_img6_v2.png')
             )

    

utilnav.main_navig(
    None, None,
    'Home', 'pages/nichart_home.py',
)
