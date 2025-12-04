import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

###################################
# Hard-coded menu items for NiChart
dict_menu = {
    "Home": "pages/home.py",
    "Explore NiChart": "pages/explore_nichart.py",
    "Select Pipeline": "pages/sel_pipelines.py",
    "Select Project": "pages/upload_data.py",
    "Run Pipeline": "pages/run_pipelines.py",
    "View Your Brain Chart": "pages/view_results.py",
    "Download Results": "pages/download_results.py",
    "Settings": "pages/settings.py",
}

dict_workflow = {
}

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
    if st.session_state.has_cloud_session:
        user_email = st.session_state.cloud_user_email
    else: 
        user_email = "place@holder.com"
    if True:
        logout_url = 'https://cbica-nichart.auth.us-east-1.amazoncognito.com/logout?client_id=4shr6mm2h0p0i4o9uleqpu33fj&logout_uri=https://neuroimagingchart.com'
        st.markdown(
            f"""
            <div style="
                position: absolute;
                top: 0.75rem;
                right: 1.2rem;
                background-color: #EEE;
                padding: 4px 10px;
                border-radius: 8px;
                font-size: 0.85rem;">
                {user_email} |
                <a href="{logout_url}" style="text-decoration:none;">Logout</a>
            </div>
            """,
            unsafe_allow_html=True
        )

def show_menu() -> None:
    with st.sidebar:
        list_options = list(dict_menu.keys())
        if 'sel_menu' not in st.session_state:
            st.session_state.sel_menu = list_options[0]
            sel_ind = 0
        else:
            sel_ind = list_options.index(st.session_state.sel_menu)
        sel_menu = option_menu(
            'NiChart',
            list_options,
            icons=['house', 'map', 'check2-square', 'upload', 'rocket-takeoff', 'graph-up', 'download', 'sliders'],
            menu_icon='cast',
            default_index=sel_ind
        )

        if sel_menu is None:
            return
        
        if sel_menu == st.session_state.sel_menu:
            return
        
        sel_page = dict_menu[sel_menu]
        st.session_state.sel_menu = sel_menu
        st.switch_page(sel_page)
        
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

def add_sidebar_options():
    with st.sidebar:

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(
                "[![GitHub](https://img.shields.io/badge/GitHub-Repo-8DA1EE?style=for-the-badge&logo=github&logoColor=white)](https://github.com/CBICA/NiChart_Project)"
            )
        with col2:
            st.markdown(
                "[![ISTAGING](https://img.shields.io/badge/NiChart-Web-C744C2?style=for-the-badge&logoColor=white)](https://neuroimagingchart.com)"
            )
