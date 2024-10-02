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
flag_expanded = st.session_state.paths["dset"] == ""
with st.expander("Select output", expanded=flag_expanded):
    # Dataset name: All results will be saved in a main folder named by the dataset name
    helpmsg = "Each dataset's results are organized in a dedicated folder named after the dataset"
    dset_name = utilst.user_input_text(
        "Dataset name", st.session_state.dset_name, helpmsg
    )

    # Out folder
    helpmsg = "DLMUSE images will be saved to the output folder.\n\nChoose the path by typing it into the text field or using the file browser to browse and select it"
    path_out = utilst.user_input_folder(
        "Select folder",
        "btn_sel_out_dir",
        "Output folder",
        st.session_state.paths["last_sel"],
        st.session_state.paths["out"],
        helpmsg,
    )
    if dset_name != "" and path_out != "":
        st.session_state.dset_name = dset_name
        st.session_state.paths["out"] = path_out
        st.session_state.paths["dset"] = os.path.join(path_out, dset_name)
        st.session_state.paths["mlscore"] = os.path.join(
            path_out, dset_name, "MLScores"
        )
        st.success(f'Results will be saved to: {st.session_state.paths['mlscore']}')

# Panel for running MLScore
flag_expanded = st.session_state.paths["mlscores"] == ""
with st.expander("Run MLScore", expanded=flag_expanded):

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
    flag_btn = os.path.exists(
        st.session_state.paths["csv_demog"]
    ) and os.path.exists(st.session_state.paths["csv_dlmuse"])
    btn_mlscore = st.button("Run MLScore", disabled=not flag_btn)

    if btn_mlscore:
        run_dir = os.path.join(
            st.session_state.paths["root"], "src", "workflow", "workflows", "w_sMRI"
        )

        if not os.path.exists(st.session_state.paths["mlscore"]):
            os.makedirs(st.session_state.paths["mlscore"])

        with st.spinner("Wait for it..."):
            os.system(f"cd {run_dir}")
            st.info("Running: mlscores_workflow ", icon=":material/manufacturing:")
            cmd = f"python3 {run_dir}/call_snakefile.py --run_dir {run_dir} --dset_name {dset_name} --input_rois {csv_dlmuse} --input_demog {csv_demog} --dir_out {st.session_state.paths['mlscore']}"
            os.system(cmd)
            st.success("Run completed!", icon=":material/thumb_up:")

            # Set the output file as the input for the related viewers
            csv_mlscores = f"{st.session_state.paths['mlscore']}/{dset_name}_DLMUSE+MLScores.csv"
            if os.path.exists(csv_mlscores):
                st.session_state.paths["csv_mlscores"] = csv_mlscores

            st.success(f"Out file: {csv_mlscores}")
