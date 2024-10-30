from typing import Any
import streamlit as st
import os

def update_default_paths() -> None:
    """
    Update default paths in session state if the working dir changed
    """
    for d_tmp in st.session_state.dict_paths.keys():
        st.session_state.paths[d_tmp] = os.path.join(
            st.session_state.paths["dset"],
            st.session_state.dict_paths[d_tmp][0],
            st.session_state.dict_paths[d_tmp][1],
        )
        print(f"setting {st.session_state.paths[d_tmp]}")

    st.session_state.paths["csv_dlmuse"] = os.path.join(
        st.session_state.paths["dset"], "DLMUSE", "DLMUSE_Volumes.csv"
    )

    st.session_state.paths["csv_mlscores"] = os.path.join(
        st.session_state.paths["dset"],
        "MLScores",
        f"{st.session_state.dset}_DLMUSE+MLScores.csv",
    )

    st.session_state.paths["csv_demog"] = os.path.join(
        st.session_state.paths["dset"], "Lists", "Demog.csv"
    )

    st.session_state.paths["csv_plot"] = os.path.join(
        st.session_state.paths["dset"], "Plots", "Data.csv"
    )

def reset_flags() -> None:
    """
    Resets flags if the working dir changed
    """
    for tmp_key in st.session_state.flags.keys():
        st.session_state.flags[tmp_key] = False
    st.session_state.flags['dset'] = True

