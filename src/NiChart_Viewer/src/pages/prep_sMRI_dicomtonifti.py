import plotly.express as px
import os
import streamlit as st
import tkinter as tk
from tkinter import filedialog
import utils_st as utilst
import utils_dicom as utildcm
import pandas as pd

st.markdown(
        """
    1. Select Input Folder: Choose the directory containing your raw DICOM files.
    2. Detect Series: The application will automatically identify different imaging series within the selected folder.
    3. Choose Series: Select the specific series you want to extract. You can pick multiple series if necessary.
    4. Extract Nifti Scans: Click the "Extract" button to convert the selected DICOM series into Nifti format. The extracted Nifti files will be saved in the specified output folder.
        """
)

# Panel for output (dataset name + out_dir)
with st.container(border=True):
    # Dataset name: All results will be saved in a main folder named by the dataset name
    helpmsg = "Each dataset's results are organized in a dedicated folder named after the dataset"
    dset_name = utilst.user_input_text("Dataset name", st.session_state.dset_name, helpmsg)

    # Out folder
    helpmsg = 'Extracted Nifti images will be saved to the output folder.\n\nChoose the path by typing it into the text field or using the file browser to browse and select it'
    path_out = utilst.user_input_folder("Select folder",
                                        'btn_sel_out_dir',
                                        "Output folder",
                                        st.session_state.path_last_sel,
                                        st.session_state.path_out,
                                        helpmsg)
    if dset_name != '' and path_out != '':
        st.session_state.dset_name = dset_name
        st.session_state.path_out = path_out
        st.session_state.path_dset = os.path.join(path_out, dset_name)
        st.session_state.path_nifti = os.path.join(path_out, dset_name, 'Nifti')

        if st.session_state.path_nifti != '':
            if not os.path.exists(st.session_state.path_nifti):
                os.makedirs(st.session_state.path_nifti)
            st.success(f'Results will be saved to: {st.session_state.path_nifti}')

# Panel for detecting dicom series
if st.session_state.dset_name != '':
    with st.container(border=True):
        # Input dicom image folder
        helpmsg = 'Input folder with dicom files (.dcm).\n\nChoose the path by typing it into the text field or using the file browser to browse and select it'
        path_dicom = utilst.user_input_folder("Select folder",
                                        'btn_indir_dicom',
                                        "Input dicom folder",
                                        st.session_state.path_last_sel,
                                        st.session_state.path_dicom,
                                        helpmsg)
        st.session_state.path_dicom = path_dicom

        flag_btn = os.path.exists(st.session_state.path_dicom)

        # Detect dicom series
        btn_detect = st.button("Detect Series", disabled = not flag_btn)
        if btn_detect:
            with st.spinner('Wait for it...'):
                df_dicoms, list_series = utildcm.detect_series(path_dicom)
                if len(list_series) == 0:
                    st.warning('Could not detect any dicom series!')
                else:
                    st.success(f"Detected {len(list_series)} dicom series!", icon = ":material/thumb_up:")
                st.session_state.list_series = list_series
                st.session_state.df_dicoms = df_dicoms

# Panel for selecting and extracting dicom series
if len(st.session_state.list_series) > 0:
    with st.container(border=True):

        # Selection of img modality
        helpmsg = 'Modality of the extracted images'
        st.session_state.sel_mod = utilst.user_input_select('Image Modality',
                                                            ['T1', 'T2', 'FL', 'DTI', 'rsfMRI'], 'T1',
                                                            helpmsg)
        # Selection of dicom series
        st.session_state.sel_series = st.multiselect("Select series:",
                                                     st.session_state.list_series,
                                                     [])
        # Create out folder for the selected modality
        if len(st.session_state.sel_series) > 0:
            if st.session_state.path_nifti != '':
                st.session_state.path_selmod = os.path.join(st.session_state.path_nifti,
                                                            st.session_state.sel_mod)
                if not os.path.exists(st.session_state.path_selmod):
                    os.makedirs(st.session_state.path_selmod)

        # Button for extraction
        flag_btn = st.session_state.df_dicoms.shape[0] > 0 and len(st.session_state.sel_series) > 0
        btn_convert = st.button("Convert Series", disabled = not flag_btn)
        if btn_convert:
            with st.spinner('Wait for it...'):
                utildcm.convert_sel_series(st.session_state.df_dicoms,
                                        st.session_state.sel_series,
                                        st.session_state.path_selmod)
                st.session_state.list_input_nifti = [f for f in os.listdir(st.session_state.path_selmod) if f.endswith('nii.gz')]
                if len(st.session_state.list_input_nifti) == 0:
                    st.warning(f'Could not extract any nifti images')
                else:
                    st.success(f'Extracted {len(st.session_state.list_input_nifti)} nifti images')

            # utilst.display_folder(st.session_state.path_selmod)

# Panel for viewing extracted nifti images
if len(st.session_state.list_input_nifti)) > 0:
    with st.container(border=True):


# FIXME: this is for debugging; will be removed
with st.expander('session_state: All'):
    st.write(st.session_state.df_dicoms)
    st.write(st.session_state.sel_series)

