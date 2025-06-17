import streamlit as st
import utils.utils_pages as utilpg
import utils.utils_doc as utildoc
import utils.utils_io as utilio
import utils.utils_session as utilss
import os

from utils.utils_logger import setup_logger
logger = setup_logger()

logger.debug('Start of setup!')

# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()

def update_out_dir():
    """
    Panel for selecting output dir
    """
    with st.container(border=True):
        
        st.markdown(
            """
            The ***:red[designated output directory]*** where all generated results will be stored.
            """
        )

        if st.session_state.app_type == 'cloud':
            st.warning('Output directory selection isn’t needed in the cloud version!')
            return
            

        if 'key_setup_out_btn' not in st.session_state:
            st.session_state.key_setup_out_btn = False

        if st.session_state.key_setup_out_btn:
            
            curr_dir = st.session_state.paths["out_dir"]

            if 'key_setup_out_mode' not in st.session_state:
                st.session_state.key_setup_out_mode = 'Browse Path'
            
            sel_mode = st.radio(
                'Sel mode',
                options=['Browse Path', 'Enter Manually'],
                horizontal = True,
                label_visibility = 'collapsed',                
                key = 'key_setup_out_mode'
            )
            
            if sel_mode == 'Browse Path':
                # Browse output folder
                if st.button('Browse'):
                    sel_dir = utilio.browse_folder(curr_dir)
                    if sel_dir is not None and sel_dir != curr_dir:
                        utilss.update_out_dir(sel_dir)
                    st.session_state['key_setup_out_btn'] = False
                    st.rerun()

            elif sel_mode == 'Enter Manually':
                # Enter output folder
                sel_dir = st.text_input(
                    'Enter path',
                    value=curr_dir,
                    label_visibility='collapsed',
                )
                
                if st.button("Submit"):
                    if sel_dir is not None and sel_dir != curr_dir:
                        utilss.update_out_dir(sel_dir)
                    st.session_state['key_setup_out_btn'] = False
                    st.rerun()


        else:
            st.success(
                f"Output directory: {st.session_state.paths['out_dir']}",
                icon=":material/thumb_up:",
            )

            if st.button('Update'):
                st.session_state['key_setup_out_btn'] = True
                st.rerun()

def update_task() -> None:
    """
    Panel for updating task name
    """
    with st.container(border=True):

        st.markdown(
            """
                - Task Name serves as a ***:red[unique identifier for your analysis]***.
                - All results will be organized and saved under a folder named after the Task Name, following a predefined nested folder structure.
                - You can also ***:red[choose a demo dataset]*** or ***:red[revisit a previous task]*** here.
            """
        )

        if 'key_setup_task_btn' not in st.session_state:
            st.session_state.key_setup_task_btn = False

        if st.session_state.key_setup_task_btn:
            out_dir = st.session_state.paths["out_dir"]
            curr_task = st.session_state.navig['task']            

            if 'key_setup_task_mode' not in st.session_state:
                st.session_state.key_setup_task_mode = 'Select Existing'

            sel_mode = st.radio(
                'Sel mode',
                options=['Select Existing', 'Enter Manually'],
                horizontal = True,
                label_visibility = 'collapsed',
                key = 'key_setup_task_mode'
            )
            
            if sel_mode == 'Select Existing':
                list_tasks = utilio.get_subfolders(out_dir)
                if len(list_tasks) > 0:

                    if 'key_setup_task_list' not in st.session_state:
                        st.session_state.key_setup_task_list = None
                        
                    sel_task = st.selectbox(
                        "Select Existing Task:",
                        options = list_tasks,
                        label_visibility = 'collapsed',
                        key = 'key_setup_task_list'
                    )
            elif sel_mode == 'Enter Manually':

                if 'key_setup_task_text' not in st.session_state:
                    st.session_state.key_setup_task_text = None

                sel_task = st.text_input(
                    "Task name:",
                    None,
                    placeholder="My_new_study",
                    label_visibility = 'collapsed',
                    key = 'key_setup_task_text'
                )   
                
            if st.button("Submit"):
                if sel_task is not None and sel_task != curr_task:
                    utilss.update_task(sel_task)
                st.session_state['key_setup_task_btn'] = False
                st.rerun()
        
        else:
            st.success(
                f"Task name: {st.session_state.navig['task']}",
                icon=":material/thumb_up:",
            )
            if st.button('Update'):
                st.session_state['key_setup_task_btn'] = True
                st.rerun()

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


st.info(
    """
    ### User Configuration
    - To help you better organize your work, please select a ***:red[few important settings]*** below.
    """
)

if 'key_setup_sel_item' not in st.session_state:
    st.session_state.key_setup_sel_item = None

sel_item = st.pills(
        "Select Config Item",
    ["Output Directory", "Task Name"],
    selection_mode="single",
    key='key_setup_sel_item',
    label_visibility="collapsed",
)

## Required to make sure that state of widget is consistent with returned value
#if st.session_state._setup_sel_item != sel_item:
    #st.session_state._setup_sel_item = sel_item
    #st.rerun()    

if sel_item == 'Output Directory':
    update_out_dir()

if sel_item == 'Task Name':
    update_task()

