import streamlit as st
import utils.utils_st as utilst
from streamlit_option_menu import option_menu

###################################
# Hard-coded menu items for NiChart
dict_menu = {
    "Home": "pages/home.py",
    "Data": "pages/data.py",
    "Pipelines": "pages/pipelines.py",
    "Results": "pages/results.py",
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
            icons=['house', 'clipboard-data', 'rocket-takeoff', 'graph-up', 'gear'],
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
