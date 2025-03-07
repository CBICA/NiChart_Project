import os
import sys

import pandas as pd
import streamlit as st
import utils.utils_cloud as utilcloud
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

def panel_experiment() -> None:
    """
    Panel for selecting the working dir
    """
    with st.container(border=True):
        utilst.util_panel_workingdir(st.session_state.app_type)

        if os.path.exists(st.session_state.paths["experiment"]):
            list_subdir = utilio.get_subfolders(st.session_state.paths["experiment"])
            st.success(
                f"Working directory is set to: {st.session_state.paths['experiment']}",
                icon=":material/thumb_up:",
            )
            if len(list_subdir) > 0:
                st.info(
                    "Working directory already includes the following folders: "
                    + ", ".join(list_subdir)
                )
            st.session_state.flags["dir_out"] = True

        utilst.util_workingdir_get_help()


def panel_inrois() -> None:
    """
    Panel for uploading input rois
    """
    with st.container(border=True):
        if st.session_state.app_type == "cloud":
            utilst.util_upload_file(
                st.session_state.paths["csv_dlmuse"],
                "DLMUSE csv",
                "uploaded_dlmuse_file",
                False,
                "visible",
            )

        else:  # st.session_state.app_type == 'desktop'
            utilst.util_select_file(
                "selected_dlmuse_file",
                "DLMUSE csv",
                st.session_state.paths["csv_dlmuse"],
                st.session_state.paths["file_search_dir"],
            )

        if os.path.exists(st.session_state.paths["csv_dlmuse"]):
            p_dlmuse = st.session_state.paths["csv_dlmuse"]
            st.session_state.flags["csv_dlmuse"] = True
            st.success(f"Data is ready ({p_dlmuse})", icon=":material/thumb_up:")

            df_rois = pd.read_csv(st.session_state.paths["csv_dlmuse"])
            with st.expander("Show ROIs", expanded=False):
                st.dataframe(df_rois)

        # Check the input data
        @st.dialog("Input data requirements")  # type:ignore
        def help_inrois_data():
            df_muse = pd.DataFrame(
                columns=["MRID", "702", "701", "600", "601", "..."],
                data=[
                    ["Subj1", "...", "...", "...", "...", "..."],
                    ["Subj2", "...", "...", "...", "...", "..."],
                    ["Subj3", "...", "...", "...", "...", "..."],
                    ["...", "...", "...", "...", "...", "..."],
                ],
            )
            st.markdown(
                """
                ### DLMUSE File:
                The DLMUSE CSV file contains volumes of ROIs (Regions of Interest) segmented by the DLMUSE algorithm. This file is generated as output when DLMUSE is applied to a set of images.
                """
            )
            st.write("Example MUSE data file:")
            st.dataframe(df_muse)

        col1, col2 = st.columns([0.5, 0.1])
        with col2:
            if st.button(
                "Get help 🤔", key="key_btn_help_mlinrois", use_container_width=True
            ):
                help_inrois_data()


def panel_indemog() -> None:
    """
    Panel for uploading demographics
    """
    with st.container(border=True):
        flag_manual = st.checkbox("Enter data manually", False)
        if flag_manual:
            st.info("Please enter values for your sample")
            df_rois = pd.read_csv(st.session_state.paths["csv_dlmuse"])
            df_tmp = pd.DataFrame({"MRID": df_rois["MRID"], "Age": None, "Sex": None})
            df_user = st.data_editor(df_tmp)

            if st.button("Save data"):
                if not os.path.exists(
                    os.path.dirname(st.session_state.paths["csv_demog"])
                ):
                    os.makedirs(os.path.dirname(st.session_state.paths["csv_demog"]))

                df_user.to_csv(st.session_state.paths["csv_demog"], index=False)
                st.success(f"Data saved to {st.session_state.paths['csv_demog']}")

        else:
            if st.session_state.app_type == "cloud":
                utilst.util_upload_file(
                    st.session_state.paths["csv_demog"],
                    "Demographics csv",
                    "uploaded_demog_file",
                    False,
                    "visible",
                )

            else:  # st.session_state.app_type == 'desktop'
                utilst.util_select_file(
                    "selected_demog_file",
                    "Demographics csv",
                    st.session_state.paths["csv_demog"],
                    st.session_state.paths["file_search_dir"],
                )

        if os.path.exists(st.session_state.paths["csv_demog"]):
            p_demog = st.session_state.paths["csv_demog"]
            st.session_state.flags["csv_demog"] = True
            st.success(f"Data is ready ({p_demog})", icon=":material/thumb_up:")

            df_demog = pd.read_csv(st.session_state.paths["csv_demog"])
            with st.expander("Show demographics data", expanded=False):
                st.dataframe(df_demog)

        # Check the input data
        if os.path.exists(st.session_state.paths["csv_demog"]):
            if st.button("Verify input data"):
                [f_check, m_check] = w_mlscores.check_input(
                    st.session_state.paths["csv_dlmuse"],
                    st.session_state.paths["csv_demog"],
                )
                if f_check == 0:
                    st.session_state.flags["csv_dlmuse+demog"] = True
                    st.success(m_check, icon=":material/thumb_up:")
                    st.session_state.flags["csv_mlscores"] = True
                else:
                    st.session_state.flags["csv_dlmuse+demog"] = False
                    st.error(m_check, icon=":material/thumb_down:")
                    st.session_state.flags["csv_mlscores"] = False

        # Help
        @st.dialog("Input data requirements")  # type:ignore
        def help_indemog_data():
            df_demog = pd.DataFrame(
                columns=["MRID", "Age", "Sex"],
                data=[
                    ["Subj1", "57", "F"],
                    ["Subj2", "65", "M"],
                    ["Subj3", "44", "F"],
                    ["...", "...", "..."],
                ],
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
            st.write("Example demographic data file:")
            st.dataframe(df_demog)

        col1, col2 = st.columns([0.5, 0.1])
        with col2:
            if st.button(
                "Get help 🤔", key="key_btn_help_mlindemog", use_container_width=True
            ):
                help_indemog_data()


def panel_run() -> None:
    """
    Panel for running ml score calculation
    """
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

            if "SITE" not in df_tmp.columns:
                st.warning(
                    "SITE column missing, assuming all samples are from the same site!"
                )
                df_tmp["SITE"] = "SITE1"

            with st.spinner("Wait for it..."):
                st.info("Running: mlscores_workflow ", icon=":material/manufacturing:")
                fcount = df_tmp.shape[0]
                if st.session_state.has_cloud_session:
                    utilcloud.update_stats_db(
                        st.session_state.cloud_user_id, "MLScores", fcount
                    )

                try:
                    if flag_harmonize:
                        w_mlscores.run_workflow(
                            st.session_state.experiment,
                            st.session_state.paths["root"],
                            st.session_state.paths["csv_dlmuse"],
                            st.session_state.paths["csv_demog"],
                            st.session_state.paths["mlscores"],
                        )
                    else:
                        w_mlscores.run_workflow_noharmonization(
                            st.session_state.experiment,
                            st.session_state.paths["root"],
                            st.session_state.paths["csv_dlmuse"],
                            st.session_state.paths["csv_demog"],
                            st.session_state.paths["mlscores"],
                        )
                except:
                    st.warning(":material/thumb_up: ML scores calculation failed!")

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

            with st.expander("View output data with ROIs and ML scores"):
                df_ml = pd.read_csv(st.session_state.paths["csv_mlscores"])
                st.dataframe(df_ml)

        s_title = "ML Biomarkers"
        s_text = """
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
    with st.container(border=True):
        out_zip = bytes()
        if not os.path.exists(st.session_state.paths["download"]):
            os.makedirs(st.session_state.paths["download"])
        f_tmp = os.path.join(st.session_state.paths["download"], "MLScores.zip")
        out_zip = utilio.zip_folder(st.session_state.paths["mlscores"], f_tmp)

        st.download_button(
            "Download ML Scores",
            out_zip,
            file_name=f"{st.session_state.experiment}_MLScores.zip",
            disabled=False,
        )

# Call all steps
t1, t2, t3, t4 =  st.tabs(
    ['Working Dir', 'Input Data', 'In ROIS', 'In Demog']
)
if st.session_state.app_type == "cloud":
    t1, t2, t3, t4, t5 =  st.tabs(
        ['Working Dir', 'Input Data', 'In ROIS', 'In Demog', 'Download']
    )

with t1:
    panel_experiment()
with t2:
    panel_inrois()
with t3:
    panel_indemog()
with t4:
    panel_run()
if st.session_state.app_type == "cloud":
    with t5:
        panel_download()

# FIXME: For DEBUG
utilst.add_debug_panel()
