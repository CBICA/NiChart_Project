import streamlit as st
import tkinter as tk
import os
from tkinter import filedialog

if "instantiated" not in st.session_state:
    st.session_state.odir='/home/guraylab/GitHub/CBICA/NiChart_Project/output_folder'
    st.session_state.instantiated = True

def browse_folder(path_init: str):
    """
    Folder selector
    Returns the folder name selected by the user
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    out_path = filedialog.askdirectory(initialdir=path_init)
    root.destroy()
    if len(out_path) == 0:
        return None
    return out_path

st.write('**Output path**')
tmp_sel=None
tab1, tab2 = st.tabs(['Enter', 'Select'])
with tab1:
    tmp_sel = st.text_input(
        '',
        key="_text_input_folder",
        #value=st.session_state.paths["dir_out"],
        value=st.session_state.odir,
        label_visibility='visible',
        help="Enter the output path. A new folder will be created if it doesn't exist."
    )

with tab2:
    if st.button(
        'Browse Folders',
        key='_btn_seldirout',
        help='Select the output path on your system'
    ):
        #tmp_sel = browse_folder(st.session_state.paths["dir_out"])
        tmp_sel = browse_folder(st.session_state.odir)
    
        if tmp_sel is not None and os.path.exists(tmp_sel):
            #st.session_state.paths["dir_out"] = os.path.abspath(tmp_sel)
            st.session_state.odir = os.path.abspath(tmp_sel)
            st.rerun()

#with st.container():
    #st.write(st.session_state)
    
    
