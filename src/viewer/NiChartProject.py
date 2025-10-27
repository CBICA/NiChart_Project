import argparse
import os

import streamlit as st
import utils.utils_pages as utilpg
import utils.utils_session as utilss

from PIL import Image
from st_pages import add_page_title, get_nav_from_toml

nicon = Image.open("../resources/nichart1.png")

# Init session state
utilss.init_session_state()

# utilpg.config_page()

print("--- RERUN: HOME PAGE STARTING ---") 

# Read user arg to select cloud / desktop
parser = argparse.ArgumentParser(description="NiChart Application Server")
parser.add_argument(
    "--cloud",
    action="store_true",
    default=False,
    help="If passed, set the session type to cloud",
)
try:
    args = parser.parse_args()
except SystemExit as e:
    # This exception will be raised if --help or invalid command line arguments
    # are used. Currently streamlit prevents the program from exiting normally
    # so we have to do a hard exit.
    exit_code = e.code
    if exit_code is None:
        exit_code = 0  # Default exit code (success)
    elif isinstance(exit_code, str):
        try:
            exit_code = int(exit_code)
        except ValueError:
            exit_code = 1  # Default error code if conversion fails
    print(f"Exiting with code: {exit_code}")
    os._exit(exit_code)

if args.cloud:
    st.session_state.app_type = "CLOUD"
    st.session_state.forced_cloud = True

pages = {
    "Home": [
        st.Page("pages/home.py", title="Home"),
    ],
    "Info": [
        st.Page("pages/info.py", title="Info"),
    ],
    "Modes": [
        st.Page("pages/single_subject.py", title="Single Subject"),
        st.Page("pages/multi_subject.py", title="Multi Subject"),
        st.Page("pages/no_user_mri.py", title="No User MRI"),
    ],    
    "Pipelines": [
        st.Page("pages/sel_pipelines.py", title="Select Pipelines"),
        st.Page("pages/run_pipelines.py", title="Run Pipelines"),
    ],
}

pg = st.navigation(pages, position="top")
pg.run()

