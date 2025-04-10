import streamlit as st

###################################
# Hard-coded menu items for NiChart

dict_main_menu = {
    "Home": "pages/Home.py",
    "Config": "pages/Config.py",
    "Workflow": "pages/Menu.py",
    "Debug": "pages/Debug.py",
}

dict_workflow = {
    "Load Input Data": "pages/data_input.py",
    "Select Pipeline(s)": "pages/select_workflow.py",
    "View Results": "pages/Menu.py",
}

def select_main_menu() -> None:
    """
    Select main menu page from a list and switch to it
    """
    with st.sidebar:
        sel_main_menu = st.pills(
            "Select Main Menu",
            dict_main_menu.keys(),
            selection_mode="single",
            default=st.session_state.navig['main_menu'],
            label_visibility="collapsed",
        )

        if sel_main_menu is None:
            return

        if sel_main_menu == st.session_state.navig['main_menu']:
            return

        # Reset selection in next steps
        st.session_state.navig['workflow'] = None
        st.session_state.navig['pipeline_step'] = None

        st.session_state.navig['main_menu'] = sel_main_menu
        sel_page = dict_main_menu[sel_main_menu]
        st.switch_page(sel_page)

def select_workflow() -> None:
    """
    Select pipeline from a list and switch to pipeline page
    """
    if st.session_state.navig['main_menu'] != "Pipelines":
        return

    with st.sidebar:
        # st.markdown('##### ')
        st.markdown("### Pipeline:")
        sel_workflow = st.pills(
            "Workflow",
            dict_workflow.keys(),
            selection_mode="single",
            default=st.session_state.navig['workflow'],
            label_visibility="collapsed",
        )
        if sel_workflow is None:
            return
        if sel_workflow == st.session_state.navig['workflow']:
            return

        st.session_state.navig['workflow'] = sel_workflow
        sel_page = dict_workflow[sel_workflow]
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


def show_menu() -> None:
    select_main_menu()
    select_workflow()
