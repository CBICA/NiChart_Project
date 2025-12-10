
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

# State
if 'clicked_cards' not in st.session_state:
    st.session_state.clicked_cards = [False for i in range(5)]

def click_callback(integer):
    def callback():
        new_range = [False for i in range(5)]
        new_range[integer] = True
        st.session_state.clicked_cards = new_range
        st.write(f"Callback from card {integer}. Contents: {new_range}")
    return callback

def dismiss_callback():
    st.session_state.dialog_tile = None
    return

@st.dialog("Details", on_dismiss=dismiss_callback)
def nichart_dialog():
    st.markdown('Hello NiChart')
    st.session_state.dialog_tile = None

@st.dialog("Details", on_dismiss=dismiss_callback)
def biomarker_dialog():
    st.markdown('Predictive scores for quantification of brain aging or neurodegeneration from MRI images')
    st.session_state.dialog_tile = None

@st.dialog("Details", on_dismiss=dismiss_callback)
def segmentation_dialog():
    st.markdown('Predictive scores for quantification of brain aging or neurodegeneration from MRI images')
    st.session_state.dialog_tile = None

@st.dialog("Details", on_dismiss=dismiss_callback)
def brainagingsubtypes_dialog():
    st.markdown('Brain aging subtypes')
    st.session_state.dialog_tile = None

@st.dialog("Details", on_dismiss=dismiss_callback)
def abnormalitymaps_dialog():
    st.markdown('Voxelwise abnormality maps')
    st.session_state.dialog_tile = None

def info_dialog(dialog_tile):
    pass

    st.write(st.session_state.dialog_tile)
   
    if dialog_tile == 'NiChart':
        nichart_dialog()
    
    elif dialog_tile == 'AI Biomarkers':
        biomarker_dialog()

    elif dialog_tile == 'DL Segmentation':
        segmentation_dialog()
    
    elif dialog_tile == 'Brain Aging Subtypes':
        brainagingsubtypes_dialog()
    
    elif dialog_tile == 'Voxelwise Abnormality Maps':
        abnormalitymaps_dialog()

    st.session_state.dialog_tile = None

def imgfile_to_data(filepath):
    import base64

    with open(filepath, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data)
    data = "data:image/png;base64," + encoded.decode("utf-8")    
    return data
    

imgdir =  os.path.join(st.session_state.paths['resources'], 'images', 'nichart_logo')

my_card = {
    "width": "95%",
    #"height": "100%",
    "margin": "2px",
    "border-radius": "1vmin",
    "box-shadow": "0 0 10px rgba(0,0,0,0.3)",
    "color": 'green',
}

my_filter = {
    "background-color": "rgba(0, 1, 0, 0)"
}

my_text = {
    "font-family": "'Source Sans', sans-serif;",
    "font-size": '1.5rem',
    "font-weight": "bold",
    "color": 'white',
    "background-color": "black",
    "padding": "2px 4px",  # optional: makes it look like real highlighting
    "border-radius": "3px",  # optional: rounds the highlight edges 
}

def show_short_desc(title):
    if title == "NiChart":
        st.markdown("NiChart analyzes neuroimaging data like never before.")
        pass
    if title == "AI Biomarkers":
        st.markdown("Extract AI-derived biomarkers from your imaging data.")
        pass
    if title == "DL Segmentation":
        st.markdown("Segment brain scans in seconds.")
        pass
    if title == "Brain Aging Subtypes":
        st.markdown("Calculate trajectory along subtypes of brain aging.")
        pass
    if title == "Voxelwise Abnormality Maps":
        st.markdown("View abnormalities in anatomical space.")
        pass

def show_full_desc(title):
    if title == "NiChart":
        st.markdown('''NeuroImaging Chart of AI-based Imaging Biomarkers. A framework to process MRI images, harmonize to reference data, apply and contribute machine learning models, and derive individualized neuroimaging biomarkers.''')
    if title == "AI Biomarkers":
        st.markdown('''
    NiChart leverages T1 images to compute several machine learning imaging predictions in the form of SPARE scores.


    SPARE - Spatial Patterns of Abnormalities reflect structural variability in the brain associated with a given task. These models use the DLMUSE ROIs, age (not used in SPARE-BA), sex, and DLICV as predictors and SVM classifiers.

    SPARE-AD: An individualized index reflecting the presence and severity of Alzheimer’s disease (AD)-like patterns of atrophy in the brain (https://pubmed.ncbi.nlm.nih.gov/19416949/)

    All other SPARE disease models reflect the specific conditions for depression (SPARE-Depression), psychosis (SPARE-Psychosis), hypertension (SPARE-Hypertension), diabetes (SPARE-Diabetes), obesity (SPARE-Obesity), and smoking status (SPARE-Smoking).

    The cardiometabolic risk models (smoking, obesity, hypertension, and diabetes) are a reflection of the previously published SPARE-CVMs (https://www.nature.com/articles/s41467-025-57867-7).

    DeepSPARE-BA - Reflects structural changes using the minimally processed T1 image directly.''')
    if title == "DL Segmentation":
        st.markdown('''NiChart performs three aspects of processing, deep learning-based ICV estimation (DLICV), segmentation into MUSE (https://pubmed.ncbi.nlm.nih.gov/26679328/) ROIs using a deep learning method (https://pubmed.ncbi.nlm.nih.gov/40960397/). Additionally, if FLAIR is available DLWMLS, a deep learning white matter lesion segmentation method, computes white matter hyperintensity presence within the DLMUSE ROIs.''')
    if title == "Brain Aging Subtypes":
        st.markdown('''
    Brain aging dimensions reflect continuous latent representations of structural patterns associated with aging. 


    Surreal-GAN R-indices: The R-index reflects the severity of individualized brain changes along multiple dimensions, potentially reflecting the stage of a mixture of underlying neuropathological and biological processes that induce deviations from the distribution of a reference brain structure.

The Surreal-GAN dimensions of aging were obtained using a knowledge distillation method to train a tabular transformer with four encoder layers to predict the original dimensions of aging from the full Surreal-GAN architecture:
    
    R1: subcortical atrophy, mainly concentrated in the caudate and putamen
    
    R2: focal medial temporal lobe (MTL) atrophy
    
    R3: parieto-temporal atrophy, including that in middle temporal gyrus, angular gyrus and middle occipital gyrus
    
    R4: diffuse cortical atrophy in medial and lateral frontal regions, as well as superior parietal and occipital regions
    
    R5: perisylvian atrophy centered around the insular cortex


    Coupled Cross-Sectional and Longitudinal Non-Negative Matrix Factorization (CCL-NMF) identifies dominant patterns of brain aging by jointly modeling baseline (cross-sectional) and follow-up (longitudinal) MRI data. Cross-sectional maps capture cumulative aging differences across individuals, while longitudinal maps quantify subject-specific change rates. These maps are jointly decomposed via NMF into shared spatial components and individual loadings, providing continuous measures of distinct aging patterns.

''')
    if title == "Voxelwise Abnormality Maps":
        st.markdown('''
    Voxelwise abnormality maps quantify how much each brain region deviates from a normative aging model, highlighting localized tissue loss or expansion

RAVENS maps provide mass-preserving tissue density measures, enabling precise regional comparisons of gray matter, white matter, and CSF volumes.
Combining RAVENS with CSF-based abnormality maps yields a spatial fingerprint of structural vulnerability, showing where tissue density differs from healthy controls at the voxel level.
These maps allow detection of subtle, early, and spatially specific changes tied to cardiometabolic risks, disease progression, or aging.
The approach supports subject-level interpretation, enabling visualization of individual neuroanatomical abnormalities, not just group averages.''')
    
    pass

def card(title, image_path):
    with st.container(border=True):
        st.image(imgfile_to_data(image_path))
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

    with cols[1]: # AI Biomarkers
        card(title="AI Biomarkers",
             image_path=os.path.join(imgdir, 'nichart_logo_v2_img3_v2.png')
             )

    with cols[2]: # DL Segmentation
        card(title="DL Segmentation",
             image_path=os.path.join(imgdir, 'nichart_logo_v2_img4_v2.png')
             )

    with cols[3]: # Brain Aging Subtypes
        card(title="Brain Aging Subtypes",
             image_path=os.path.join(imgdir, 'nichart_logo_v2_img5_v2.png')
             )

    with cols[4]: # Voxelwise Abnormality Maps
        card(title="Voxelwise Abnormality Maps",
             image_path=os.path.join(imgdir, 'nichart_logo_v2_img6_v2.png')
             )

    

utilnav.main_navig(
    None, None,
    'Home', 'pages/nichart_home.py',
)
