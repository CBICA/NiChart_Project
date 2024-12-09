import os
import sys

import pandas as pd
import streamlit as st
import utils.utils_io as utilio
import utils.utils_menu as utilmenu
import utils.utils_session as utilss
import utils.utils_st as utilst

run_dir = os.path.join(st.session_state.paths["root"], "src", "workflows", "w_sMRI")
sys.path.append(run_dir)
import w_mlscores as w_mlscores

# Page config should be called for each page
utilss.config_page()

utilmenu.menu()

st.write("# Machine Learning (ML)-Based Imaging Biomarkers")

st.markdown(
    """
    - Application of pre-trained  machine learning (ML) models to derive imaging biomarkers.
    - ML biomarkers quantify brain changes related to aging and disease.
    - ML models were trained on ISTAGING reference data using DLMUSE ROIs after statistical harmonization with COMBAT.
    """
)

# Update status of checkboxes
if '_check_ml_wdir' in st.session_state:
    st.session_state.checkbox['ml_wdir'] = st.session_state._check_ml_wdir
if '_check_ml_in' in st.session_state:
    st.session_state.checkbox['ml_in'] = st.session_state._check_ml_in
if '_check_ml_run' in st.session_state:
    st.session_state.checkbox['ml_run'] = st.session_state._check_ml_run
if '_check_ml_download' in st.session_state:
    st.session_state.checkbox['ml_download'] = st.session_state._check_ml_download

def panel_wdir() -> None:
    """
    Panel for selecting the working dir
    """
    icon = st.session_state.icon_thumb[st.session_state.flags["dir_out"]]
    show_panel_wdir = st.checkbox(
        f":material/folder_shared: Working Directory {icon}",
        key='_check_ml_wdir',
        value=st.session_state.checkbox['ml_wdir']
    )
    if not st.session_state._check_ml_wdir:
        return

    with st.container(border=True):
        utilst.util_panel_workingdir(st.session_state.app_type)
        if os.path.exists(st.session_state.paths["dset"]):
            st.success(
                f"All results will be saved to: {st.session_state.paths['dset']}",
                icon=":material/thumb_up:",
            )
            st.session_state.flags["dir_out"] = True

        utilst.util_workingdir_get_help()

def panel_indata() -> None:
    """
    Panel for uploading input files
    """
    msg = st.session_state.app_config[st.session_state.app_type]["msg_infile"]
    icon = st.session_state.icon_thumb[st.session_state.flags["csv_dlmuse+demog"]]
    show_panel_indata = st.checkbox(
        f":material/upload: {msg} Data {icon}",
        disabled=not st.session_state.flags["dir_out"],
        key='_check_ml_in',        
        value=st.session_state.checkbox['ml_in']
    )
    if not st.session_state._check_ml_in:
        return

    show_panel_int1 = st.checkbox(
        f":material/upload: {msg} T1 Images {icon}",
        disabled=not st.session_state.flags["dir_out"],
        key='_check_ml_in',
        value=st.session_state.checkbox['ml_in']
    )
    if not st.session_state._check_ml_in:
        return

    with st.container(border=True):
        if st.session_state.app_type == "cloud":
            utilst.util_upload_file(
                st.session_state.paths["csv_dlmuse"],
                "DLMUSE csv",
                "uploaded_dlmuse_file",
                False,
                "visible",
            )
            if os.path.exists(st.session_state.paths["csv_dlmuse"]):
                p_dlmuse = st.session_state.paths["csv_dlmuse"]
                st.success(f"Data is ready ({p_dlmuse})", icon=":material/thumb_up:")

            utilst.util_upload_file(
                st.session_state.paths["csv_demog"],
                "Demographics csv",
                "uploaded_demog_file",
                False,
                "visible",
            )
            if os.path.exists(st.session_state.paths["csv_demog"]):
                p_demog = st.session_state.paths["csv_dlmuse"]
                st.success(f"Data is ready ({p_demog})", icon=":material/thumb_up:")

        else:  # st.session_state.app_type == 'desktop'
            utilst.util_select_file(
                "selected_dlmuse_file",
                "DLMUSE csv",
                st.session_state.paths["csv_dlmuse"],
                st.session_state.paths["file_search_dir"],
            )
            if os.path.exists(st.session_state.paths["csv_dlmuse"]):
                p_dlmuse = st.session_state.paths["csv_dlmuse"]
                st.success(f"Data is ready ({p_dlmuse})", icon=":material/thumb_up:")

            utilst.util_select_file(
                "selected_demog_file",
                "Demographics csv",
                st.session_state.paths["csv_demog"],
                st.session_state.paths["file_search_dir"],
            )
            if os.path.exists(st.session_state.paths["csv_demog"]):
                p_demog = st.session_state.paths["csv_demog"]
                st.success(f"Data is ready ({p_demog})", icon=":material/thumb_up:")
        
        # Check the input data
        if st.button('Verify input data'):
            [f_check, m_check] = w_mlscores.check_input(
                st.session_state.paths["csv_dlmuse"],
                st.session_state.paths["csv_demog"],
            )
            if f_check == 0:
                st.session_state.flags["csv_dlmuse+demog"] = True
                st.success(m_check, icon=":material/thumb_up:")
            else:
                st.session_state.flags["csv_dlmuse+demog"] = False
                st.error(m_check, icon=":material/thumb_down:")

        # Check the input data
        @st.dialog("Input data requirements")  # type:ignore
        def help_input_data():
            df_muse = pd.DataFrame(
                columns=['MRID', '702', '701', '600', '601', '...'],
                data=[
                    ['Subj1', '...', '...', '...', '...', '...'],
                    ['Subj2', '...', '...', '...', '...', '...'],
                    ['Subj3', '...', '...', '...', '...', '...'],
                    ['...', '...', '...', '...', '...', '...']
                ]
            )
            st.markdown(
                """
                ### DLMUSE File:
                The DLMUSE CSV file contains volumes of ROIs (Regions of Interest) segmented by the DLMUSE algorithm. This file is generated as output when DLMUSE is applied to a set of images.
                """
            )
            st.write('Example MUSE data file:')
            st.dataframe(df_muse)

            df_demog = pd.DataFrame(
                columns=['MRID', 'Age', 'Sex'],
                data=[
                    ['Subj1', '57', 'F'],
                    ['Subj2', '65', 'M'],
                    ['Subj3', '44', 'F'],
                    ['...', '...', '...']
                ]
            )
            st.markdown(
                """
                ### Demographics File:
                The DEMOGRAPHICS CSV file contains demographic information for each subject in the study.
                - **Required Columns:**
                    - **MRID:** Unique subject identifier.
                    - **Age:** Age of the subject.
                    - **Sex:** Sex of the subject (e.g., M, F).
                - **Matching MRIDs:** Ensure the MRID values in this file match the corresponding MRIDs in the DLMUSE file for merging the data files.
                """
            )
            st.write('Example demographic data file:')
            st.dataframe(df_demog)

        col1, col2 = st.columns([0.5, 0.1])
        with col2:
            if st.button('Get help 🤔', key='key_btn_help_mlinput', use_container_width=True):
                help_input_data()


def panel_run() -> None:
    """
    Panel for running ml score calculation
    """
    icon = st.session_state.icon_thumb[st.session_state.flags["csv_mlscores"]]
    st.checkbox(
        f":material/new_label: Run DLMUSE {icon}",
        disabled=not st.session_state.flags["dir_t1"],
        key='_check_ml_run',
        value=st.session_state.checkbox['ml_run']
    )
    if not st.session_state._check_ml_run:
        return

    with st.container(border=True):

        btn_mlscore = st.button("Run MLScore", disabled=False)
        if btn_mlscore:

            if not os.path.exists(st.session_state.paths["mlscores"]):
                os.makedirs(st.session_state.paths["mlscores"])

            # Check flag for harmonization
            flag_harmonize = True
            df_tmp = pd.read_csv(st.session_state.paths["csv_demog"])
            if df_tmp.shape[0] < st.session_state.harm_min_samples:
                flag_harmonize = False
                st.warning("Sample size is small. The data will not be harmonized!")

            if 'SITE' not in df_tmp.columns:
                st.warning("SITE column missing, assuming all samples are from the same site!")
                df_tmp['SITE'] = 'SITE1'

            with st.spinner("Wait for it..."):
                st.info("Running: mlscores_workflow ", icon=":material/manufacturing:")

                try:
                    if flag_harmonize:
                        w_mlscores.run_workflow(
                            st.session_state.dset,
                            st.session_state.paths["root"],
                            st.session_state.paths["csv_dlmuse"],
                            st.session_state.paths["csv_demog"],
                            st.session_state.paths["mlscores"],
                        )
                    else:
                        w_mlscores.run_workflow_noharmonization(
                            st.session_state.dset,
                            st.session_state.paths["root"],
                            st.session_state.paths["csv_dlmuse"],
                            st.session_state.paths["csv_demog"],
                            st.session_state.paths["mlscores"],
                        )
                except:
                    st.warning(':material/thumb_up: ML scores calculation failed!')

        # Check out file
        if os.path.exists(st.session_state.paths["csv_mlscores"]):
            st.success(
                f"Data is ready ({st.session_state.paths['csv_mlscores']}).\n\n Output data includes harmonized ROIs, SPARE scores (AD, Age, Diabetes, Hyperlipidemia, Hypertension, Obesity, Smoking) and SurrealGAN subtype indices (R1-R5)",
                icon=":material/thumb_up:",
            )
            st.session_state.flags["csv_mlscores"] = True

            # Copy output to plots
            if not os.path.exists(st.session_state.paths["plots"]):
                os.makedirs(st.session_state.paths["plots"])
            os.system(
                f"cp {st.session_state.paths['csv_mlscores']} {st.session_state.paths['csv_plot']}"
            )
            st.session_state.flags["csv_plot"] = True
            p_plot = st.session_state.paths["csv_plot"]
            print(f"Data copied to {p_plot}")

            with st.expander('View output data with ROIs and ML scores'):
                df_ml=pd.read_csv(st.session_state.paths["csv_mlscores"])
                st.dataframe(df_ml)

        s_title="ML Biomarkers"
        s_text="""
        - DLMUSE ROI volumes are harmonized to reference data.
        - SPARE scores are calculated using harmonized ROI values and pre-trained models
        - SurrealGAN scores are calculated using harmonized ROI values and pre-trained models
        - Final results, ROI values and ML scores, are saved in the result csv file
        """
        utilst.util_get_help(s_title, s_text)

def panel_download() -> None:
    """
    Panel for downloading results
    """
    st.checkbox(
        ":material/new_label: Download Scans",
        disabled=not st.session_state.flags["csv_mlscores"],
        key='_check_ml_download',
        value=st.session_state.checkbox['ml_download']
    )
    if not st.session_state._check_ml_download:
        return

    with st.container(border=True):
        out_zip = bytes()
        if not os.path.exists(st.session_state.paths["download"]):
            os.makedirs(st.session_state.paths["download"])
        f_tmp = os.path.join(st.session_state.paths["download"], "MLScores.zip")
        out_zip = utilio.zip_folder(st.session_state.paths["mlscores"], f_tmp)

        st.download_button(
            "Download ML Scores",
            out_zip,
            file_name=f"{st.session_state.dset}_MLScores.zip",
            disabled=False,
        )

st.divider()
panel_wdir()
panel_indata()
panel_run()
if st.session_state.app_type == "cloud":
    panel_download()

# FIXME: For DEBUG
utilst.add_debug_panel()
