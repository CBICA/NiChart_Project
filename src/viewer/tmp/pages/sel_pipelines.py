
import streamlit as st
from menu import menu

# Initialize st.session_state.pipeline
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

st.session_state._pipeline = st.session_state.pipeline

def set_pipeline():
    # Callback function to save the pipeline selection to Session State
    st.session_state.pipeline = st.session_state._pipeline

# Selectbox to choose role
st.selectbox(
    "Select pipeline:",
    [None, "dlmuse", "dlwmls"],
    key="_pipeline",
    on_change=set_pipeline,
)
menu()
