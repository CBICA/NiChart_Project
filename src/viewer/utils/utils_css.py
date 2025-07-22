import streamlit as st

def load_css():
    st.markdown(
        """
        <style>
        div.stButton > button {
            white-space: normal;
            word-break: normal;
            overflow-wrap: break-word;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
