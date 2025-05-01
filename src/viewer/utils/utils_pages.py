import streamlit as st
import utils.utils_st as utilst
from streamlit_option_menu import option_menu

###################################
# Hard-coded menu items for NiChart
dict_menu = {
    "Home": "pages/home.py",
    "Quick Setup": "pages/setup.py",
    "Load Data": "pages/data.py",
    "Process Data": "pages/pipelines.py",
    "View Results": "pages/results.py",
    "Settings": "pages/settings.py",
}

dict_workflow = {
}

def show_menu() -> None:
    with st.sidebar:
        list_options = list(dict_menu.keys())
        sel_ind = list_options.index(st.session_state.sel_menu)
        sel_menu = option_menu(
            'NiChart',
            list_options,
            icons=['house', 'sliders', 'clipboard-data', 'rocket-takeoff', 'graph-up', 'gear'],
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
    st.set_page_config(
        page_title="NiChart",
        page_icon=st.session_state.nicon,
        layout="wide",
        # layout="centered",
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
