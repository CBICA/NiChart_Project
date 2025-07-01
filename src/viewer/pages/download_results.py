import streamlit as st
import utils.utils_pages as utilpg
import utils.utils_session as utilses
import os
from utils.utils_logger import setup_logger

logger = setup_logger()
logger.debug('Page: Download Results')



# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()

st.markdown(
    """
    ### Download Results 
    """
)

st.info('Coming Soon!')

# Show selections
utilses.disp_selections()

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()
