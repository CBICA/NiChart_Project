import os

import streamlit as st
import utils.utils_st as utilst
import utils.utils_io as utilio

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

def save_dlmuse_file():
    # Save dlmuse file to local storage
    if len(st.session_state['uploaded_dlmuse']) > 0:
        utilio.copy_uploaded_file(
            st.session_state['uploaded_dlmuse'],
            os.path.join(st.session_state.paths["DLMUSE"], 'DLMUSE.csv')
        )


# Panel for output (dataset name + out_dir)
utilst.util_panel_workingdir(st.session_state.app_type)

# Panel for selecting input dlmuse csv
flag_disabled = os.path.exists(st.session_state.paths["dset"]) == False
if st.session_state.app_type == 'CLOUD':
    msg_txt = 'Upload DLMUSE csv file'
    utilst.util_upload_file(
        st.session_state.paths['csv_seg'],
        'uploaded_dlmuse_file',
        flag_disabled,
        msg_txt
    )

else:   # st.session_state.app_type == 'DESKTOP'
    msg_txt = 'Select DLMUSE csv file'
    utilst.util_select_file(
        st.session_state.paths['csv_seg'],
        st.session_state.paths['last_in_dir'],
        flag_disabled,
        msg_txt
    )

# Panel for selecting input demog csv
flag_disabled = os.path.exists(st.session_state.paths["dset"]) == False
if st.session_state.app_type == 'CLOUD':
    msg_txt = 'Upload demographic csv file'
    utilst.util_upload_file(
        st.session_state.paths['csv_demog'],
        'uploaded_demog_file',
        flag_disabled,
        msg_txt
    )

else:   # st.session_state.app_type == 'DESKTOP'
    msg_txt = 'Select demographic csv file'
    utilst.util_select_file(
        st.session_state.paths['csv_demog'],
        st.session_state.paths['last_in_dir'],
        flag_disabled,
        msg_txt
    )

# Panel for running MLScore
with st.expander(":material/model_training: Run MLScore", expanded=False):

    # Button to run MLScore
    flag_btn = os.path.exists(st.session_state.paths["csv_demog"]) and os.path.exists(
        st.session_state.paths["csv_seg"]
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

            # cmd = f"python3 {run_dir}/call_snakefile.py --run_dir {run_dir} --dset_name {st.session_state.dset_name} --input_rois {csv_seg} --input_demog {csv_demog} --dir_out {st.session_state.paths['MLScores']}"

            cmd = f"python3 {run_dir}/workflow_mlscores.py --root_dir {st.session_state.paths['root']} --run_dir {run_dir} --dset_name {st.session_state.dset_name} --input_rois {st.session_state.paths['csv_seg']} --input_demog {st.session_state.paths['csv_demog']} --dir_out {st.session_state.paths['MLScores']}"
            print(f'About to run: {cmd}')
            os.system(cmd)

    # Check out file
    if os.path.exists(st.session_state.paths["csv_mlscores"]):
        st.success(
            f"Out file created: {st.session_state.paths['csv_mlscores']}",
            icon=":material/thumb_up:",
        )




# Panel for downloading results
if st.session_state.app_type == "CLOUD":
    with st.expander(":material/download: Download Results", expanded=False):

        # Zip results and download
        flag_btn = os.path.exists(st.session_state.paths['MLScores'])
        out_zip = bytes()
        if flag_btn:
            if not os.path.exists(st.session_state.paths["OutZipped"]):
                os.makedirs(st.session_state.paths["OutZipped"])
            f_tmp = os.path.join(st.session_state.paths["OutZipped"], "MLScores.zip")
            out_zip = utilio.zip_folder(st.session_state.paths["MLScores"], f_tmp)

        st.download_button(
            "Download ML Scores",
            out_zip,
            file_name="MLScores.zip",
            disabled=not flag_btn,
        )


with st.expander("FIXME: TMP - Session state"):
    st.write(st.session_state)
with st.expander("TMP: session vars - paths"):
    st.write(st.session_state.paths)
