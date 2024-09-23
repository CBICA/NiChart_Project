import plotly.express as px
import os
import streamlit as st
import tkinter as tk
from tkinter import filedialog
import utils_st as utilst

def browse_file(path_init):
    '''
    File selector
    Returns the file name selected by the user and the parent folder
    '''
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    out_path = filedialog.askopenfilename(initialdir = path_init)
    path_out = os.path.dirname(out_path)
    root.destroy()
    return out_path, path_out

def browse_folder(path_init):
    '''
    Folder selector
    Returns the folder name selected by the user
    '''
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    out_path = filedialog.askdirectory(initialdir = path_init)
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

    # Dataset name: All results will be saved in a main folder named by the dataset name 
    helpmsg = "Each dataset's results are organized in a dedicated folder named after the dataset"
    dset_name = utilst.user_input_text("Dataset name", 
                                        st.session_state.dset_name, 
                                        helpmsg)
    st.session_state.dset_name = dset_name

    # Input T1 image folder
    helpmsg = 'DLMUSE will be applied to .nii/.nii.gz images directly in the input folder.\n\nChoose the path by typing it into the text field or using the file browser to browse and select it'
    path_t1 = utilst.user_input_folder("Select folder",
                                       'btn_indir_t1',
                                       "Input folder",
                                       st.session_state.path_last_sel,
                                       st.session_state.path_t1,
                                       helpmsg)
    st.session_state.path_t1 = path_t1

    # Out folder
    helpmsg = 'DLMUSE will generate a segmented image for each input image, and a csv file with the volumes of ROIs for the complete dataset.\n\nChoose the path by typing it into the text field or using the file browser to browse and select it'
    path_out = utilst.user_input_folder("Select folder",
                                       'btn_out_dir',
                                        "Output folder",
                                        st.session_state.path_last_sel,
                                        st.session_state.path_out,
                                        helpmsg)
    st.session_state.path_out = path_out

    # Device type
    helpmsg = "Choose 'cuda' if your computer has an NVIDIA GPU, 'mps' if you have an Apple M-series chip, and 'cpu' if you have a standard CPU."
    device = utilst.user_input_select('Device',
                                      ['cuda', 'cpu', 'mps'],
                                      'dlmuse_sel_device',
                                      helpmsg)

    # Check input files
    flag_files = 1
    if not os.path.exists(path_t1):
        flag_files = 0

    if not os.path.exists(path_out):
        flag_files = 0

    run_dir = os.path.join(st.session_state.path_root, 'src', 'NiChart_DLMUSE')

    # Run workflow
    if flag_files == 1:
        if st.button("Run w_DLMUSE"):
            import time
            path_out_dlmuse = os.path.join(path_out, dset_name, 'DLMUSE')
            st.info(f"Running: NiChart_DLMUSE -i {path_t1} -o {path_out_dlmuse} -d {device}", icon = ":material/manufacturing:")
            with st.spinner('Wait for it...'):
                time.sleep(15)
                os.system(f"NiChart_DLMUSE -i {path_t1} -o {path_out_dlmuse} -d {device}")
                st.success("Run completed!", icon = ":material/thumb_up:")

            # Set the dlmuse output as input for other modules
            out_csv = f"{path_out_dlmuse}/{dset_name}_DLMUSE.csv"
            if os.path.exists(out_csv):
                st.session_state.path_csv_dlmuse = out_csv
                st.session_state.path_dlmuse = path_out_dlmuse

# FIXME: this is for debugging; will be removed
with st.expander('session_state: All'):
    st.write(st.session_state)

