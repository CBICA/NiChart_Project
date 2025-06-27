import streamlit as st
import utils.utils_pages as utilpg
#import utils.utils_doc as utildoc
import utils.utils_io as utilio
import utils.utils_session as utilses
import utils.utils_cmaps as utilcmap
import os

from utils.utils_logger import setup_logger

import streamlit_antd_components as sac

logger = setup_logger()
logger.debug('Page: Settings')

# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()

def panel_models_path():
    """
    Panel for selecting models
    """
    with st.container(border=True):
        st.write('Work in progress!')

def panel_resources_path():
    """
    Panel for selecting resource directories
    """
    with st.container(border=True):
        sel_res = st.pills(
            "Select resource type",
            ["process definitions", "roi lists"],
            selection_mode="single",
            default=None,
            label_visibility="collapsed",
        )

        if sel_res is None:
            return
        
        if sel_res == "process definitions":
            out_dir = st.session_state.paths["proc_def"]

            # Browse output folder
            if st.button('Browse path'):
                sel_dir = utilio.browse_folder(out_dir)
                utilss.update_out_dir(sel_dir)
                st.rerun()

            # Enter output folder
            sel_dir = st.text_input(
                'Enter path',
                value=out_dir,
                # label_visibility='collapsed',
            )
            if sel_dir != out_dir:
                utilss.update_out_dir(sel_dir)
                st.rerun()

            if st.session_state.flags["out_dir"]:
                st.success(
                    f"Output directory: {st.session_state.paths['out_dir']}",
                    icon=":material/thumb_up:",
                )

            #utildoc.util_help_dialog(utildoc.title_out, utildoc.def_out)

def panel_misc() -> None:
    """
    Panel for setting various parameters
    """
    with st.container(border=True):

        is_cloud_mode = st.session_state.app_type == "cloud"
        if st.checkbox("Switch to cloud?", value=is_cloud_mode):
            st.session_state.app_type = "cloud"
        else:
            st.session_state.app_type = "desktop"

def panel_debug_options():
    sel_debug = st.pills(
        "Select Debug Options",
        ["Session State", "Output Files"],
        selection_mode="single",
        default=None,
        label_visibility="collapsed",
    )

    if sel_debug == "Session State":
        with st.container(border=True):
            disp_session_state()

    elif sel_debug == "Output Files":
        with st.container(border=True):
            st.markdown('##### '+ st.session_state.navig['task'] + ':')
            disp_folder_tree()

def panel_plot_colors():
    utilcmap.panel_update_cmaps()

#st.info(
st.markdown(
    """
    ### Configuration Options
    """
)

tab = sac.tabs(
    items=[
        sac.TabsItem(label='Paths'),
        sac.TabsItem(label='Plot Colors'),
        sac.TabsItem(label='Debug'),
        sac.TabsItem(label='Misc'),
    ],
    size='lg',
    align='left'
)

if tab == 'Paths':
    panel_resources_path()
    panel_models_path()

elif tab == 'Plot Colors':
    panel_plot_colors()

elif tab == 'Debug':
    panel_debug_options()

elif tab == 'Misc':
    panel_misc()

# Show session state vars
if st.session_state.mode == 'debug':
    utilses.disp_session_state()

