import plotly.express as px
import os
import streamlit as st
import tkinter as tk
from tkinter import filedialog
import utils_st as utilst
import utils_dicom as utildcm

st.markdown(
        """
    - Extract sMRI Nifti scans from raw dicom filess.
        """
)

with st.container(border=True):

    # Dataset name: All results will be saved in a main folder named by the dataset name 
    helpmsg = "Each dataset's results are organized in a dedicated folder named after the dataset"
    dset_name = utilst.user_input_text("Dataset name", 
                                        st.session_state.dset_name, 
                                        helpmsg)
    st.session_state.dset_name = dset_name

    # Input dicom image folder
    helpmsg = 'Input folder with dicom files (.dcm).\n\nChoose the path by typing it into the text field or using the file browser to browse and select it'
    path_dicom = utilst.user_input_folder("Select folder",
                                       'btn_indir_dicom',
                                       "Input folder",
                                       st.session_state.path_last_sel,
                                       st.session_state.path_dicom,
                                       helpmsg)
    st.session_state.path_dicom = path_dicom

    # Check input files
    flag_files = 1
    if not os.path.exists(path_dicom):
        flag_files = 0

    run_dir = os.path.join(st.session_state.path_root, 'src', 'NiChart_DLMUSE')

    # Run workflow
    if flag_files == 1:
        if st.button("Detect Series"):

            with st.spinner('Wait for it...'):
                df_dicoms, list_series = utildcm.detect_series(path_dicom)
                st.success("Run completed!", icon = ":material/thumb_up:")                
                if len(list_series) == 0:
                    st.warning('Could not detect any dicom series!')
                st.session_state.list_series = list_series

if len(st.session_state.list_series) > 0:

    with st.container(border=True):

        list_sel = st.multiselect("Select T1 Series:", st.session_state.list_series, [])

                    



    #path_out_t1 = os.path.join(path_out, dset_name, 'Images', 'T1')

            #if not os.path.exists(path_out_t1):
                #os.makedirs(path_out_t1)
            

    ## Out folder
    #helpmsg = 'Nifti images will be extracted to the output folder.\n\nChoose the path by typing it into the text field or using the file browser to browse and select it'
    #path_out = utilst.user_input_folder("Select folder",
                                        #'btn_out_dir',
                                        #"Output folder",
                                        #st.session_state.path_last_sel,
                                        #st.session_state.path_out,
                                        #helpmsg)
    #st.session_state.path_out = path_out

        ## Set the nifti output as input for other modules
        #if os.path.exists(path_out_t1):
            #st.session_state.path_t1 = path_out_t1

# FIXME: this is for debugging; will be removed
with st.expander('session_state: All'):
    st.write(st.session_state)

