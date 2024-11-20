import argparse
import os

import pandas as pd
import streamlit as st
import utils.utils_rois as utilroi
import utils.utils_st as utilst

from menu import menu
from init_session_state import init_session_state

from st_pages import add_page_title, get_nav_from_toml

from PIL import Image

nicon = Image.open("../resources/nichart1.png")
st.set_page_config(
    page_title="NiChart",
    page_icon=nicon,
    layout="wide",
    #layout="centered",
    menu_items={
        "Get help": "https://neuroimagingchart.com/",
        "Report a bug": "https://neuroimagingchart.com/",
        "About": "https://neuroimagingchart.com/",
    },
)

# Read user arg to select cloud / desktop
parser = argparse.ArgumentParser(description='NiChart Application Server')
parser.add_argument('--cloud', action='store_true', default=False,
                    help="If passed, set the session type to cloud")
try:
    args = parser.parse_args()
except SystemExit as e:
    # This exception will be raised if --help or invalid command line arguments
    # are used. Currently streamlit prevents the program from exiting normally
    # so we have to do a hard exit.
    os._exit(e.code)
if args.cloud:
    st.session_state.app_type = "CLOUD"

# Initialize session state variables
st.switch_page("pages/home.py")
