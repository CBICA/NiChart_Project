import streamlit as st
import utils.utils_pages as utilpg
import os

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


# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()

st.markdown(
    """
    ### Debugging Options (Dev)
    - Select debugging options here.
    """
)

sel_config = st.pills(
    "Select Debug",
    ["Session State", "Output Files"],
    selection_mode="single",
    default=None,
    label_visibility="collapsed",
)

if sel_config == "Session State":
    with st.container(border=True):
        disp_session_state()
if sel_config == "Output Files":
    with st.container(border=True):
        st.markdown('##### '+ st.session_state.navig['task'] + ':')
        disp_folder_tree()

