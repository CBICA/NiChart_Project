
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

def click_callback(dialog_tile):
    def callback():
        st.session_state.dialog_tile = dialog_tile
    return callback

def dismiss_callback():
    st.session_state.dialog_tile = None
    return

@st.dialog("Details", on_dismiss=dismiss_callback)
def info_dialog(dialog_tile):
    
    if dialog_tile == 'NiChart':
        st.markdown('Hello NiChart')
    
    elif dialog_tile == 'AI Biomarkers':
        st.markdown('Predictive scores for quantification of brain aging or neurodegeneration from MRI images')

    st.session_state.dialog_tile = None

# State
if "dialog_tile" not in st.session_state:
    st.session_state.dialog_tile = None
else:
    info_dialog(st.session_state.dialog_tile)


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


with st.container(horizontal_alignment="center"):
    st.markdown("## What can I do with NiChart?")
with st.container(horizontal=True, horizontal_alignment="center"):
    cols = st.columns(5)

    data = imgfile_to_data(os.path.join(imgdir, 'nichart_logo_v2_img1_v2.png'))
    with cols[0]:
        clicked0 = card(
            title="NiChart",
            text="",
            image = data,
            styles={
                "card": my_card,
                "text": my_text,
                "filter": my_filter,
                "title": my_text,
            },
            key = 'card0',
            on_click=click_callback("NiChart")
        )

    data = imgfile_to_data(os.path.join(imgdir, 'nichart_logo_v2_img3_v2.png'))
    with cols[1]:
        clicked1 = card(
            title="AI Biomarkers",
            text="",
            image = data,
            styles={
                "card": my_card,
                "filter": my_filter,
                "text": my_text,
                "title": my_text,
            },
            key = 'card1',
            on_click=click_callback("AI Biomarkers")
        )

    data = imgfile_to_data(os.path.join(imgdir, 'nichart_logo_v2_img4_v2.png'))
    with cols[2]:
        clicked2 = card(
            title="DL Segmentation",
            text="",
            image = data,
            styles={
                "card": my_card,
                "filter": my_filter,
                "text": my_text,
                "title": my_text,
            },
            key = 'card2',
            on_click=click_callback("DL Segmentation")
        )

    data = imgfile_to_data(os.path.join(imgdir, 'nichart_logo_v2_img5_v2.png'))
    with cols[3]:
        clicked3 = card(
            title="Brain Aging Subtypes",
            text="",
            image = data,
            styles={
                "card": my_card,
                "filter": my_filter,
                "text": my_text,
                "title": my_text,
            },
            key = 'card3',
            on_click=click_callback("Brain Aging Subtypes")
        )

    data = imgfile_to_data(os.path.join(imgdir, 'nichart_logo_v2_img6_v2.png'))
    with cols[4]:
        clicked4 = card(
            title="Voxelwise Abnormality Maps",
            text="",
            image = data,
            styles={
                "card": my_card,
                "filter": my_filter,
                "text": my_text,
                "title": my_text,
            },
            key = 'card4',
            on_click=click_callback("Voxelwise Abnormality Maps")
        )

#if clicked0:
#    st.session_state.dialog_tile = 'NiChart'
#    clicked0 = False

#if clicked1:
#    st.session_state.dialog_tile = 'AI Biomarkers'
#    clicked1 = False

# ---- Open dialog if set ----
#if st.session_state.dialog_tile is not None:
#    info_dialog()
    
#st.write(clicked0)
#st.write(clicked1)
utilnav.main_navig(
    None, None,
    'Home', 'pages/nichart_home.py',
)
