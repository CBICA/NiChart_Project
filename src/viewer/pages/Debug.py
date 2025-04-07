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


def disp_folder(path: str, indent: int = 0):
    """Recursively display the contents of a folder with subfolders."""
    try:
        items = sorted(os.listdir(path))
    except Exception as e:
        st.error(f"Error reading directory: {e}")
        return

    for item in items:
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path):
            st.markdown(">" * indent + "📁 " + item)
            disp_folder(full_path, indent + 1)
        else:
            st.markdown(">" * indent + "📄 " + item, unsafe_allow_html=True)



# Page config should be called for each page
utilpg.config_page()
utilpg.select_main_menu()

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
        disp_folder(st.session_state.paths['task'])

