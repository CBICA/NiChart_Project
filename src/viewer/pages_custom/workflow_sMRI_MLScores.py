import os

import streamlit as st
import utils.utils_io as utilio
import utils.utils_st as utilst

st.markdown(
    """
    NiChart sMRI ML pipeline using COMBAT harmonization, SPARE
    scores, and SurrealGAN indices.
    - Input data is a csv file with the DLMUSE ROI volumes and
    a csv file with demographic info (Age, Sex, DX, Site).

    ### Want to learn more?
    - Visit [SPARE GitHub](https://github.com/CBICA/spare_scores)
        """
)


# def save_dlmuse_file() -> None:
#     # Save dlmuse file to local storage
#     if len(st.session_state["uploaded_dlmuse"]) > 0:
#         utilio.copy_uploaded_file(
#             st.session_state["uploaded_dlmuse"],
#             os.path.join(st.session_state.paths["DLMUSE"], "DLMUSE.csv"),
#         )


# Panel for output (dataset name + out_dir)
utilst.util_panel_workingdir(st.session_state.app_type)

# Panel for selecting input dlmuse csv
flag_disabled = not st.session_state.flags['dset']

if st.session_state.app_type == "CLOUD":
    with st.expander(":material/upload: Upload data", expanded=False):  # type:ignore
        utilst.util_upload_file(
            st.session_state.paths["csv_dlmuse"],
            "DLMUSE csv",
            "uploaded_dlmuse_file",
            flag_disabled,
            "visible",
        )
        if not flag_disabled and os.path.exists(st.session_state.paths["csv_dlmuse"]):
            st.success(f"Data is ready ({st.session_state.paths["csv_dlmuse"]})", icon=":material/thumb_up:")

        utilst.util_upload_file(
            st.session_state.paths["csv_demog"],
            "Demographics csv",
            "uploaded_demog_file",
            flag_disabled,
            "visible",
        )
        if not flag_disabled and os.path.exists(st.session_state.paths["csv_demog"]):
            st.success(f"Data is ready ({st.session_state.paths["csv_demog"]})", icon=":material/thumb_up:")

else:  # st.session_state.app_type == 'DESKTOP'
    with st.expander(":material/upload: Select data", expanded=False):  # type:ignore
        utilst.util_select_file(
            "selected_dlmuse_file",
            "DLMUSE csv",
            st.session_state.paths["csv_dlmuse"],
            st.session_state.paths["last_in_dir"],
            flag_disabled,
        )
        if not flag_disabled and os.path.exists(st.session_state.paths["csv_dlmuse"]):
            st.success(f"Data is ready ({st.session_state.paths["csv_dlmuse"]})", icon=":material/thumb_up:")

        utilst.util_select_file(
            "selected_demog_file",
            "Demographics csv",
            st.session_state.paths["csv_demog"],
            st.session_state.paths["last_in_dir"],
            flag_disabled,
        )
        if not flag_disabled and os.path.exists(st.session_state.paths["csv_demog"]):
            st.success(f"Data is ready ({st.session_state.paths["csv_demog"]})", icon=":material/thumb_up:")

# Panel for running MLScore
with st.expander(":material/model_training: Run MLScore", expanded=False):

    flag_disabled = not (os.path.exists(st.session_state.paths["csv_demog"]) and os.path.exists(
        st.session_state.paths["csv_dlmuse"])
    )

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
    with st.expander(":material/download: Download Results", expanded=False):

        flag_disabled = not os.path.exists(st.session_state.paths["csv_mlscores"])

        out_zip = bytes()
        if not flag_disabled:
            if not os.path.exists(st.session_state.paths["OutZipped"]):
                os.makedirs(st.session_state.paths["OutZipped"])
            f_tmp = os.path.join(st.session_state.paths["OutZipped"], "MLScores.zip")
            out_zip = utilio.zip_folder(st.session_state.paths["MLScores"], f_tmp)

        st.download_button(
            "Download ML Scores",
            out_zip,
            file_name=f"{st.session_state.dset}_MLScores.zip",
            disabled=flag_disabled,
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
