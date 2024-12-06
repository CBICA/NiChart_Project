import argparse
import os

import streamlit as st
import utils.utils_session as utilss

utilss.config_page()

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
    os._exit(e.code)
if args.cloud:
    st.session_state.app_type = "CLOUD"
    st.session_state.forced_cloud = True

# Initialize session state variables
st.switch_page("pages/home.py")
