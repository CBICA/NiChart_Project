import streamlit as st
import utils.utils_menu as utilmenu
import utils.utils_pages as utilpg
import utils.utils_session as utilss
from streamlit_extras.stylable_container import stylable_container
from streamlitextras.webutils import stxs_javascript

# Page config should be called for each page
utilpg.config_page()
utilpg.select_main_menu()
utilpg.select_pipeline()
utilpg.select_pipeline_step()

