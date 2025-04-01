import os

import pandas as pd
import streamlit as st
import utils.utils_rois as utilroi
import utils.utils_st as utilst
from PIL import Image
from st_pages import add_page_title, get_nav_from_toml

nicon = Image.open("../resources/nichart1.png")
st.set_page_config(
    page_title="NiChart",
    page_icon=nicon,
    layout="wide",
    # layout="centered",
    menu_items={
        "Get help": "https://neuroimagingchart.com/",
        "Report a bug": "https://neuroimagingchart.com/",
        "About": "https://neuroimagingchart.com/",
    },
)

# If you want to use the no-sections version, this
# defaults to looking in .streamlit/pages.toml, so you can
# just call `get_nav_from_toml()`

# nav = get_nav_from_toml(".streamlit/pages_sections_home.toml")
nav = get_nav_from_toml(".streamlit/pages_sections.toml")
pg = st.navigation(nav)
add_page_title(pg)
pg.run()
