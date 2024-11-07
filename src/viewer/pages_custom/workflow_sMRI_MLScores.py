import os

import streamlit as st
import utils.utils_io as utilio
import utils.utils_st as utilst

st.markdown(
    """
    - Pipeline for calculating machine learning (ML) based imaging biomarkers from DLMUSE ROIs
    - ML imaging signatures quantify the progression of brain changes related to aging and disease
    - Input data
      - DLMUSE ROI volumes and demographic data
    - Processing steps:
      - COMBAT harmonization to reference data
      - SPARE scores: Supervised ML models for disease prediction
      - SurrealGAN indices: Semi-supervised ML models for disease subtype identification
    - Output
      - ML biomarker panel 

    ### Want to learn more?
    - Visit [SPARE GitHub](https://github.com/CBICA/spare_scores)
        """
)


st.divider()

# Panel for selecting the working dir
show_panel_wdir = st.checkbox(
    f":material/folder_shared: Working Directory {st.session_state.icons['out_dir']}",
    value = False
)
if show_panel_wdir:
    with st.container(border=True):
        utilst.util_panel_workingdir(st.session_state.app_type)
        if os.path.exists(st.session_state.paths["dset"]):
            st.success(
                f"All results will be saved to: {st.session_state.paths['dset']}",
                icon=":material/thumb_up:",
            )
            st.session_state.flags["out_dir"] = True
            st.session_state.icons['out_dir'] = ':material/thumb_up:'

# Panel for uploading input data csv
show_panel_indata = st.checkbox(
    f":material/upload: Select/Upload Data {st.session_state.icons['csv_dlmuse']}",
    disabled = not st.session_state.flags['out_dir'],
    value = False
)
if show_panel_indata:
    with st.container(border=True):
        if st.session_state.app_type == "CLOUD":
            utilst.util_upload_file(
                st.session_state.paths["csv_dlmuse"],
                "DLMUSE csv",
                "uploaded_dlmuse_file",
                False,
                "visible",
            )
            if os.path.exists(st.session_state.paths["csv_dlmuse"]):
                st.success(f"Data is ready ({st.session_state.paths["csv_dlmuse"]})", icon=":material/thumb_up:")

            utilst.util_upload_file(
                st.session_state.paths["csv_demog"],
                "Demographics csv",
                "uploaded_demog_file",
                False,
                "visible",
            )
            if os.path.exists(st.session_state.paths["csv_demog"]):
                st.success(f"Data is ready ({st.session_state.paths["csv_demog"]})", icon=":material/thumb_up:")

        else:  # st.session_state.app_type == 'DESKTOP'
            utilst.util_select_file(
                "selected_dlmuse_file",
                "DLMUSE csv",
                st.session_state.paths["csv_dlmuse"],
                st.session_state.paths["last_in_dir"],
                False,
            )
            if os.path.exists(st.session_state.paths["csv_dlmuse"]):
                st.success(f"Data is ready ({st.session_state.paths["csv_dlmuse"]})", icon=":material/thumb_up:")

            utilst.util_select_file(
                "selected_demog_file",
                "Demographics csv",
                st.session_state.paths["csv_demog"],
                st.session_state.paths["last_in_dir"],
                False,
            )
            if os.path.exists(st.session_state.paths["csv_demog"]):
                st.success(
                    f"Data is ready ({st.session_state.paths["csv_demog"]})",
                    icon=":material/thumb_up:"
                )
        if os.path.exists(st.session_state.paths["csv_demog"]) and os.path.exists(st.session_state.paths["csv_demog"]):
            st.session_state.icons['csv_dlmuse'] = ':material/thumb_up:'


# Panel for running MLScore
show_panel_runml = st.checkbox(
    f":material/upload: Run MLScores {st.session_state.icons['mlscores']}",
    disabled = not st.session_state.flags['csv_dlmuse'],
    value = False
)
if show_panel_runml:
    with st.container(border=True):

        btn_mlscore = st.button("Run MLScore", disabled = flag_disabled)
        if btn_mlscore:
            run_dir = os.path.join(
                st.session_state.paths["root"], "src", "workflow", "workflows", "w_sMRI"
            )

            if not os.path.exists(st.session_state.paths["MLScores"]):
                os.makedirs(st.session_state.paths["MLScores"])

            with st.spinner("Wait for it..."):
                os.system(f"cd {run_dir}")
                st.info("Running: mlscores_workflow ", icon=":material/manufacturing:")

                # cmd = f"python3 {run_dir}/call_snakefile.py --run_dir {run_dir} --dset_name {st.session_state.dset} --input_rois {csv_dlmuse} --input_demog {csv_demog} --dir_out {st.session_state.paths['MLScores']}"

                cmd = f"python3 {run_dir}/workflow_mlscores.py --root_dir {st.session_state.paths['root']} --run_dir {run_dir} --dset_name {st.session_state.dset} --input_rois {st.session_state.paths['csv_dlmuse']} --input_demog {st.session_state.paths['csv_demog']} --dir_out {st.session_state.paths['MLScores']}"
                print(f"About to run: {cmd}")
                os.system(cmd)

        # Check out file
        if os.path.exists(st.session_state.paths["csv_mlscores"]):
            st.success(
                f"Data is ready ({st.session_state.paths['csv_mlscores']})",
                icon=":material/thumb_up:",
            )

# Panel for downloading results
if st.session_state.app_type == "CLOUD":
    show_panel_download = st.checkbox(
        f":material/new_label: Download Scans {st.session_state.icons['out_zip']}",
        disabled = not st.session_state.flags['csv_plot'],
        value = False
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

if st.session_state.debug_show_state:
    with st.expander("DEBUG: Session state - all variables"):
        st.write(st.session_state)

if st.session_state.debug_show_paths:
    with st.expander("DEBUG: Session state - paths"):
        st.write(st.session_state.paths)

if st.session_state.debug_show_flags:
    with st.expander("DEBUG: Session state - flags"):
        st.write(st.session_state.flags)
