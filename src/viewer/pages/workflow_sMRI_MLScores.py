import os
import sys
import streamlit as st
import utils.utils_io as utilio
import utils.utils_menu as utilmenu
import utils.utils_session as utilss
import utils.utils_st as utilst
import pandas as pd

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


st.divider()

# Panel for selecting the working dir
icon = st.session_state.icon_thumb[st.session_state.flags["dir_out"]]
show_panel_wdir = st.checkbox(
    f":material/folder_shared: Working Directory {icon}", value=False
)
if show_panel_wdir:
    with st.container(border=True):
        utilst.util_panel_workingdir(st.session_state.app_type)
        if os.path.exists(st.session_state.paths["dset"]):
            st.success(
                f"All results will be saved to: {st.session_state.paths['dset']}",
                icon=":material/thumb_up:",
            )
            st.session_state.flags["dir_out"] = True

# Panel for uploading input data csv
msg = st.session_state.app_config[st.session_state.app_type]["msg_infile"]
icon = st.session_state.icon_thumb[st.session_state.flags["csv_dlmuse+demog"]]
show_panel_indata = st.checkbox(
    f":material/upload: {msg} Data {icon}",
    disabled=not st.session_state.flags["dir_out"],
    value=False,
)
if show_panel_indata:
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
        if os.path.exists(st.session_state.paths["csv_dlmuse"]) and os.path.exists(
            st.session_state.paths["csv_demog"]
        ):
            st.session_state.flags["csv_dlmuse+demog"] = True


# Panel for running MLScore
icon = st.session_state.icon_thumb[st.session_state.flags["csv_mlscores"]]
show_panel_runml = st.checkbox(
    f":material/upload: Run MLScores {icon}",
    disabled=not st.session_state.flags["csv_dlmuse+demog"],
    value=False,
)
if show_panel_runml:
    with st.container(border=True):

        btn_mlscore = st.button("Run MLScore", disabled=False)
        if btn_mlscore:
            run_dir = os.path.join(
                st.session_state.paths["root"], "src", "workflows", "w_sMRI"
            )

            if not os.path.exists(st.session_state.paths["mlscores"]):
                os.makedirs(st.session_state.paths["mlscores"])

            # Check flag for harmonization
            flag_harmonize = True
            df_tmp = pd.read_csv(st.session_state.paths['csv_demog'])
            if df_tmp.shape[0] < st.session_state.harm_min_samples:
                flag_harmonize = False
                st.warning('Sample size is small. The data will not be harmonized!')

            with st.spinner("Wait for it..."):
                st.info("Running: mlscores_workflow ", icon=":material/manufacturing:")
                sys.path.append(run_dir)
                import w_mlscores as w_mlscores
                
                if flag_harmonize:
                    w_mlscores.run_workflow(
                        st.session_state.dset,
                        st.session_state.paths['root'],
                        st.session_state.paths['csv_dlmuse'],
                        st.session_state.paths['csv_demog'],
                        st.session_state.paths['mlscores'],
                    )
                else:
                    w_mlscores.run_workflow_noharmonization(
                        st.session_state.dset,
                        st.session_state.paths['root'],
                        st.session_state.paths['csv_dlmuse'],
                        st.session_state.paths['csv_demog'],
                        st.session_state.paths['mlscores'],
                    )
                    
        # Check out file
        if os.path.exists(st.session_state.paths["csv_mlscores"]):
            st.success(
                f"Data is ready ({st.session_state.paths['csv_mlscores']})",
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


# Panel for downloading results
if st.session_state.app_type == "cloud":
    show_panel_download = st.checkbox(
        ":material/new_label: Download Scans",
        disabled=not st.session_state.flags["csv_mlscores"],
        value=False,
    )
    if show_panel_download:
        with st.container(border=True):
            out_zip = bytes()
            if not os.path.exists(st.session_state.paths["OutZipped"]):
                os.makedirs(st.session_state.paths["OutZipped"])
            f_tmp = os.path.join(st.session_state.paths["OutZipped"], "MLScores.zip")
            out_zip = utilio.zip_folder(st.session_state.paths["MLScores"], f_tmp)

            st.download_button(
                "Download ML Scores",
                out_zip,
                file_name=f"{st.session_state.dset}_MLScores.zip",
                disabled=False,
            )

# FIXME: For DEBUG
utilst.add_debug_panel()
