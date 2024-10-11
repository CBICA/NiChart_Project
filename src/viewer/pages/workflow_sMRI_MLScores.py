import os

import streamlit as st
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

# Panel for output (dataset name + out_dir)
utilst.util_panel_workingdir()

# Panel for running MLScore
with st.expander("Run MLScore", expanded=False):

    # DLMUSE file name
    helpmsg = "Input csv file with DLMUSE ROI volumes.\n\nChoose the file by typing it into the text field or using the file browser to browse and select it"
    csv_dlmuse, csv_path = utilst.user_input_file(
        "Select file",
        "btn_input_dlmuse",
        "DLMUSE ROI file",
        st.session_state.paths["last_sel"],
        st.session_state.paths["csv_dlmuse"],
        helpmsg,
    )
    if os.path.exists(csv_dlmuse):
        st.session_state.paths["csv_dlmuse"] = csv_dlmuse
        st.session_state.paths["last_sel"] = csv_path

    # Demog file name
    helpmsg = "Input csv file with demographic values.\n\nChoose the file by typing it into the text field or using the file browser to browse and select it"
    csv_demog, csv_path = utilst.user_input_file(
        "Select file",
        "btn_input_demog",
        "Demographics file",
        st.session_state.paths["last_sel"],
        st.session_state.paths["csv_demog"],
        helpmsg,
    )
    if os.path.exists(csv_demog):
        st.session_state.paths["csv_demog"] = csv_demog
        st.session_state.paths["last_sel"] = csv_path

    # Button to run MLScore
    flag_btn = os.path.exists(st.session_state.paths["csv_demog"]) and os.path.exists(
        st.session_state.paths["csv_dlmuse"]
    )
    btn_mlscore = st.button("Run MLScore", disabled=not flag_btn)

    if btn_mlscore:
        run_dir = os.path.join(
            st.session_state.paths["root"], "src", "workflow", "workflows", "w_sMRI"
        )

        if not os.path.exists(st.session_state.paths["MLScores"]):
            os.makedirs(st.session_state.paths["MLScores"])

        with st.spinner("Wait for it..."):
            os.system(f"cd {run_dir}")
            st.info("Running: mlscores_workflow ", icon=":material/manufacturing:")

            # cmd = f"python3 {run_dir}/call_snakefile.py --run_dir {run_dir} --dset_name {st.session_state.dset_name} --input_rois {csv_dlmuse} --input_demog {csv_demog} --dir_out {st.session_state.paths['MLScores']}"

            cmd = f"python3 {run_dir}/workflow_mlscores.py --root_dir {st.session_state.paths['root']} --run_dir {run_dir} --dset_name {st.session_state.dset_name} --input_rois {csv_dlmuse} --input_demog {csv_demog} --dir_out {st.session_state.paths['MLScores']}"

            os.system(cmd)

            # Set the output file as the input for the related viewers
            if os.path.exists(st.session_state.paths["csv_mlscores"]):
                st.success(
                    f"Run completed! Out file: {st.session_state.paths['csv_mlscores']}",
                    icon=":material/thumb_up:",
                )

with st.expander("FIXME: TMP - Session state"):
    st.write(st.session_state)
with st.expander("TMP: session vars - paths"):
    st.write(st.session_state.paths)
