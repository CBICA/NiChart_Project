
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
    st.markdown(f"##{title} Short Description")
    pass

def show_full_desc(title):
    st.markdown(f"##{title} Full Description")
    if title == "NiChart":
        pass
    if title == "AI Biomarkers":
        pass
    if title == "DL Segmentation":
        pass
    if title == "Brain Aging Subtypes":
        pass
    if title == "Voxelwise Abnormality Maps":
        pass
    
    pass

def card(title, image_path):
    with st.container(border=True):
        st.image(imgfile_to_data(image_path))
        st.markdown(f"## {title}")
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
