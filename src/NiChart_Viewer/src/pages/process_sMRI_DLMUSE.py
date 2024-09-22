import plotly.express as px
import os
import streamlit as st
import tkinter as tk
from tkinter import filedialog

def browse_file(path_input):
    '''
    File selector
    Returns the file name selected by the user and the parent folder
    '''
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    out_path = filedialog.askopenfilename(initialdir = path_input)
    path_output = os.path.dirname(out_path)
    root.destroy()
    return out_path, path_output

def browse_folder(path_input):
    '''
    Folder selector
    Returns the folder name selected by the user
    '''
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    out_path = filedialog.askdirectory(initialdir = path_input)
    root.destroy()
    return out_path



st.markdown(
        """
    - NiChart sMRI segmentation pipeline using DLMUSE.
    - DLMUSE segments raw T1 images into 145 regions of interest (ROIs) + 105 composite ROIs.

    ### Want to learn more?
    - Visit [DLMUSE GitHub](https://github.com/CBICA/NiChart_DLMUSE)
        """
)

with st.container(border=True):

    # Dataset name: Used to create a main folder for all outputs
    tmpcol = st.columns((1,8))
    with tmpcol[0]:
        dset_name = st.text_input("Dataset name", value = st.session_state.study_name,
                                  help = "Each dataset's results are organized in a dedicated folder named after the dataset")
        st.session_state.study_name = dset_name

    # In folder name
    tmpcol = st.columns((8,1))
    with tmpcol[1]:
        if st.button("Select the input folder"):
            st.session_state.path_output = browse_folder(st.session_state.path_root)
    with tmpcol[0]:
        dir_input = st.text_input("Input folder", value = st.session_state.path_output,
                                  help = 'DLMUSE will be applied to .nii/.nii.gz images directly in the input folder.')

    # Out folder name
    tmpcol = st.columns((8,1))
    with tmpcol[1]:
        if st.button("Select the output folder", help = 'Hello'):
            st.session_state.path_output = browse_folder(st.session_state.path_root)
    with tmpcol[0]:
        dir_output = st.text_input("Output folder", value = st.session_state.path_output,
                                   help = 'DLMUSE results will be saved into the output folder, in a subfolder named "DLMUSE".\n\nDLMUSE will generate a segmented image for each input image, and a csv file with the volumes of ROIs for the complete dataset.')

    # Device type
    tmpcol = st.columns((1,8))
    with tmpcol[0]:
        device = st.selectbox("Device", ['cuda', 'cpu', 'mps'], key = 'dlmuse_sel_device', help = "Choose 'cuda' if your computer has an NVIDIA GPU, 'mps' if you have an Apple M-series chip, and 'cpu' if you have a standard CPU.")

    # Check input files
    flag_files = 1
    if not os.path.exists(dir_input):
        st.warning("Path to input folder doesn't exist")
        flag_files = 0

    if not os.path.exists(dir_output):
        st.warning("Path to output folder doesn't exist")
        flag_files = 0

    run_dir = os.path.join(st.session_state.path_root, 'src', 'NiChart_DLMUSE')

    # Run workflow
    if flag_files == 1:
        if st.button("Run w_DLMUSE"):
            st.write("Pipeline is running, please wait!")
            dir_out_dlmuse = os.path.join(dir_output, dset_name, 'DLMUSE')
            os.system(f"NiChart_DLMUSE -i {dir_input} -o {dir_out_dlmuse} -d {device}")
            st.write("Run completed!")

            # Set the output file as the input for the related viewers
            out_csv = f"{dir_output}/DLMUSE/DLMUSE_All.csv"
            if os.path.exists(out_csv):
                st.session_state.path_dlmuse = out_csv

# FIXME: this is for debugging; will be removed
with st.expander('session_state: All'):
    st.write(st.session_state)

