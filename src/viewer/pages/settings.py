import streamlit as st
import utils.utils_pages as utilpg
import utils.utils_panels as utilpn
import utils.utils_doc as utildoc
import utils.utils_io as utilio
import utils.utils_session as utilss
import os

from utils.utils_logger import setup_logger
logger = setup_logger()

logger.debug('Start of Config Screen!')

# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()

def disp_session_state():
    sel_ssvars = st.pills(
        "Select Session State Variable(s) to View",
        sorted(st.session_state.keys()),
        selection_mode="multi",
        default=None,
        #label_visibility="collapsed",
    )
    for sel_var in sel_ssvars:
        st.markdown('➤ ' + sel_var + ':')
        st.write(st.session_state[sel_var])

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

            utildoc.util_help_dialog(utildoc.title_out, utildoc.def_out)

def panel_out_dir():
    """
    Panel for selecting output dir
    """
    with st.container(border=True):
        out_dir = st.session_state.paths["out_dir"]

        st.markdown(
            """
                - Output data folder with consolidated data files for each study.
                - Output data is organized in subfolders with the study name and process name
            """
        )
        
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

        utildoc.util_help_dialog(utildoc.title_out, utildoc.def_out)

def update_task() -> None:
    """
    Panel for updating task name
    """
    with st.container(border=True):

        if st.session_state.user['btn_update']:
            out_dir = st.session_state.paths["out_dir"]
            curr_task = st.session_state.navig['task']            
            sel_mode = st.radio(
                'Sel mode',
                options=['Select Existing', 'Enter Manually'],
                horizontal = True,
                index = st.session_state.user['radio_mode']
            )
            if sel_mode == 'Select Existing':
                st.session_state.user['radio_mode'] = 0
                list_tasks = utilio.get_subfolders(out_dir)
                if len(list_tasks) > 0:
                    sel_ind = list_tasks.index(curr_task)
                    sel_task = st.selectbox(
                        "Select Existing Task:",
                        options = list_tasks,
                        index = sel_ind,
                        label_visibility = 'collapsed',
                    )
            elif sel_mode == 'Enter Manually':
                st.session_state.user['radio_mode'] = 1
                sel_task = st.text_input(
                    "Task name:",
                    None,
                    placeholder="My_new_study",
                    label_visibility = 'collapsed'
                )   
                
            if st.button("Submit"):
                if sel_task is not None and sel_task != curr_task:
                    utilss.update_task(sel_task)
                st.session_state.user['btn_update'] = False
                st.rerun()
        
        else:
            st.success(
                f"Task name: {st.session_state.navig['task']}",
                icon=":material/thumb_up:",
            )
            if st.button('Update'):
                st.session_state.user['btn_update'] = True
                st.rerun()

        #utildoc.util_help_dialog(utildoc.title_exp, utildoc.def_exp)

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


st.markdown(
    """
    ### Configuration Options
    - Select configuration options here.
    """
)

sel_config_cat = st.radio(
    "Select Config Category",
    ["Basic", "Advanced", "Debugging"],
    horizontal = True,
    #selection_mode="single",
    #default=None,
    label_visibility="collapsed",
)


if sel_config_cat == "Basic":
    sel_config = st.selectbox(
        "Select Basic Config",
        ["Output Dir", "Task Name", "Misc"],
        index=None,
        #selection_mode="single",
        #default=None,
        key = '_sel_config_cat',
        label_visibility="collapsed",
    )

    if sel_config == "Output Dir":
        panel_out_dir()

    if sel_config == "Task Name":
        update_task()

    if sel_config == "Misc":
        panel_misc()

elif sel_config_cat == "Advanced":
    del st.session_state["_sel_config_cat"]
    sel_config = st.pills(
        "Select Advanced Config",
        ["Resources", "Models"],
        selection_mode="single",
        default=None,
        label_visibility="collapsed",
    )

    if sel_config == "Resources":
        panel_resources_path()

    elif sel_config == "Models":
        panel_models_path()

if sel_config_cat == "Debugging":
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
        
