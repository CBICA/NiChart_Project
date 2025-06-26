import streamlit as st
import utils.utils_pages as utilpg
#import utils.utils_doc as utildoc
import utils.utils_io as utilio
import utils.utils_session as utilses
import utils.utils_cmaps as utilcmap    
import os

from utils.utils_logger import setup_logger
logger = setup_logger()

logger.debug('Start of Config Screen!')

# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()

def disp_folder_tree(allowed_extensions=None):
    root_path = st.session_state.paths['task']
    curr_path = st.session_state.paths['task_curr_path']

    # Prevent access outside root
    def is_within_root(path):
        return os.path.commonpath([root_path, path]) == root_path

    entries = sorted(os.listdir(curr_path))
    folders = [e for e in entries if os.path.isdir(os.path.join(curr_path, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(curr_path, e))]

    if allowed_extensions:
        files = [f for f in files if any(f.endswith(ext) for ext in allowed_extensions)]
    
    st.markdown(f"##### 📂 `{curr_path}`")

    with st.container(border=True):
        # Show subfolders
        for folder in folders:
            folder_path = os.path.join(curr_path, folder)
            if is_within_root(folder_path):
                if st.button(f"📁 {folder}", key=f'_key_folder_{folder}'):
                    st.session_state.paths['task_curr_path'] = folder_path
                    st.rerun()

        # Show files
        selected_file = None
        for f in files:
            st.write(f"📝 {f}")

    # Go Up Button (only if not already at root)
    parent_path = os.path.abspath(os.path.join(curr_path, ".."))
    if is_within_root(parent_path) and curr_path != root_path:
        if st.button("⬆️ Go Up", key='_key_btn_up'):
            st.session_state.paths['task_curr_path'] = parent_path
            st.rerun()

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
    sel_cmap = utilcmap.panel_update_cmap(
        st.session_state.plot_settings['cmaps3'],
        st.session_state.plot_settings['alphas']
    )
    st.session_state.plot_settings['cmaps3'] = sel_cmap


#st.info(
st.markdown(
    """
    ### Configuration Options
    """
)

tabs = st.tabs(
    ['Paths', 'Plot Colors', 'Debug', 'Misc']
)

with tabs[0]:
    panel_resources_path()
    panel_models_path()

with tabs[1]:
    panel_plot_colors()

with tabs[2]:
    panel_debug_options()

with tabs[3]:
    panel_misc()

# Show session state vars
if st.session_state.mode == 'debug':
    if st.sidebar.button('Show Session State'):
        utilses.disp_session_state()

