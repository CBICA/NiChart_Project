
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

import streamlit_antd_components as sac

#utilpg.config_page()
utilpg.set_global_style()

import streamlit as st

st.set_page_config(page_title="NiChart", layout="wide")

st.title("🧠 NiChart")
st.subheader("Neuroimaging Charting and AI-based Brain Analytics")
st.markdown("---")

# ---- Custom CSS for tiles ----
st.markdown("""
<style>
.tile {
    background-color: #f8f9fa;
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 8px 20px rgba(0,0,0,0.05);
    transition: transform 0.15s ease-in-out;
    cursor: pointer;
    height: 220px;
}
.tile:hover {
    transform: scale(1.02);
    background-color: #f1f3f5;
}
.tile-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 10px;
}
.tile-text {
    font-size: 15px;
    color: #444;
}
</style>
""", unsafe_allow_html=True)


# ---- Tiles content ----
tiles = [
    {
        "title": "NiChart",
        "text": "Interactive neuroimaging charts for visualizing brain structure, aging patterns, and regional abnormalities."
    },
    {
        "title": "AI Biomarkers",
        "text": "Submit your MRI scans and NiChart AI models will extract imaging scores of brain aging and AD-like brain atrophy."
    },
    {
        "title": "DL-based Segmentation",
        "text": "Deep learning models automatically segment brain regions and tissue classes from MRI scans."
    },
    {
        "title": "Aging Subtypes",
        "text": "Discover distinct patterns of brain aging by clustering individuals into biologically meaningful subtypes."
    },
    {
        "title": "Voxelwise Abnormality Maps",
        "text": "Generate voxel-level maps showing spatial patterns of abnormal brain tissue relative to normative reference data."
    },
]

# ---- Layout: 3 + 2 grid ----
row1 = st.columns(3)
row2 = st.columns(2)

cols = row1 + row2

# ---- Interactive tile display ----
for i, col in enumerate(cols):
    with col:
        tile = tiles[i]
        st.markdown(
            f"""
            <div class="tile">
                <div class="tile-title">{tile['title']}</div>
                <div class="tile-text">{tile['text']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
