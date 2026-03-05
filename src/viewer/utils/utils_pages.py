import streamlit as st
from PIL import Image

def set_global_style():
    #st.markdown("""
        #<style>
        #body, html, .stMarkdown, .stText, .stTextInput > label {
            #font-size: 18px !important;
        #}
        #h1, h2, h3 {
            #font-size: 28px;
        #}
        #</style>
    #""", unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        html, body, [class*="css"]  {
            font-size: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def config_page() -> None:
    nicon = Image.open("../resources/nichart1.png")
    st.set_page_config(
        page_title="NiChart",
        page_icon=nicon,
        layout="wide",
        #layout="centered",
        menu_items={
            "Get help": "https://neuroimagingchart.com/",
            "Report a bug": "https://github.com/CBICA/NiChart_Project/issues/new?assignees=&labels=&projects=&template=bug_report.md&title=%5BBUG%5D+",
            "About": "https://neuroimagingchart.com/",
        },
    )

