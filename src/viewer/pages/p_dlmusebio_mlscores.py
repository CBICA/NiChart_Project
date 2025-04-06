import os
import sys

import pandas as pd
import streamlit as st
import utils.utils_cloud as utilcloud
import utils.utils_io as utilio
import utils.utils_pages as utilpg
import utils.utils_panels as utilpn
import utils.utils_st as utilst

run_dir = os.path.join(st.session_state.paths["root"], "src", "workflows", "w_sMRI")
sys.path.append(run_dir)
import w_mlscores as w_mlscores

# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()

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
            df_tmp = pd.read_csv(st.session_state.paths["demog_csv"])
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
                            st.session_state.paths["dlmuse_csv"],
                            st.session_state.paths["demog_csv"],
                            st.session_state.paths["mlscores"],
                        )
                    else:
                        w_mlscores.run_workflow_noharmonization(
                            st.session_state.experiment,
                            st.session_state.paths["root"],
                            st.session_state.paths["dlmuse_csv"],
                            st.session_state.paths["demog_csv"],
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
        utilst.util_help_dialog(s_title, s_text)


# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()

st.markdown(
    """
    ### Machine Learning (ML)-Based Imaging Biomarkers
    - Application of pre-trained  machine learning (ML) models to derive imaging biomarkers.
    - ML biomarkers quantify brain changes related to aging and disease.
    - ML models were trained on ISTAGING reference data using DLMUSE ROIs after statistical harmonization with COMBAT.
    """
)

st.markdown("##### Select Task")
list_tasks = ["DLMUSE", "View Scans", "Download"]
sel_task = st.pills(
    "Select Task", list_tasks, selection_mode="single", label_visibility="collapsed"
)
if sel_task == "ML Scores":
    panel_dlmuse()
elif sel_task == "Download":
    utilpn.util_panel_download("MLScores")
